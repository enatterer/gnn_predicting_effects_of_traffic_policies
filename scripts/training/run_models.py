'''
Run GNN model training with configurable architecture and hyperparameters.

'dataset_path' and 'base_dir' need to be adjusted to the correct paths.
All the other parameters can be passed as command line arguments. Run `python run_models.py --help` to see the list of available arguments.

Example usage with default architecture, dropout, and most significant features found using ablation tests:
`python run_models.py --in_channels 5 --use_all_features False --num_epochs 500 --peak_lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3`

Our use case:
python run_models.py --gnn_arch gatv2 --unique_model_description trial13 --in_channels 5 --use_all_features True --num_epochs 2 --peak_lr 0.003 --early_stopping_patience 25
python run_models.py --gnn_arch graphSAGE --unique_model_description graphSAGE_5_features_16_cities_retina --in_channels 5 --use_all_features True --num_epochs 2 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3 --use_nested_neighbor_loader True --neighbor_sizes 5,5,5,5,5 
python run_models.py --gnn_arch eign --unique_model_description EIGN_trial_unsigned --in_channels 5 --use_all_features True --num_epochs 2 --peak_lr 0.003 --early_stopping_patience 25 
python run_models.py --gnn_arch trans_encoder --unique_model_description encoder_trial --in_channels 5 --use_all_features True --num_epochs 20 --peak_lr 0.0003 --early_stopping_patience 25
'''

import os
import sys
import json
import argparse
import copy
import random
from pathlib import Path

import torch

# TODO: Check if this helps
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # This is to avoid memory issues in Retina. Comment it out in LRZ AI

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from training.help_functions import *
from gnn.help_functions import GNN_Loss, CityBalancedGNNLoss, select_target_tensor

# Repo root: repo/scripts/training/run_models.py → go two levels up
project_root = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", project_root / "data")).resolve()

# Use universal un-normalized data, any normalization will be handled during training
dataset_path = os.path.join(project_root, 'data','bavaria','inductive_data','training_data','kreisfreistadt')

# Writable local results directory
base_dir = os.path.join(project_root, 'inductive_gnn_data_results', 'transductive') # for saving results

# Possible cities:
# ['wuerzburg','aschaffenburg','regensburg','landshut','bayreuth','erlangen','fuerth','kempten','neuulm','muenchen','augsburg','rosenheim','schweinfurt','bamberg','nuernberg', 'ingolstadt']

train_cities = ['landshut', 'bayreuth', 'schweinfurt', 'wuerzburg', 'bamberg', 'regensburg'] # Good cities, Blue Cluster 'landshut','bayreuth','schweinfurt','wuerzburg','bamberg','regensburg'
val_cities = [] # Non empty implies inductive learning
test_cities = [] # Non empty implies inductive learning

# Fixed defaults for CITY-level CrossTReS-style selective weighting.
DEFAULT_SELECTIVE_TARGET_SUPPORT_FRACTION = 0.5
DEFAULT_SELECTIVE_META_UPDATE_INTERVAL = 10
DEFAULT_SELECTIVE_META_SOURCE_STEPS = 1
DEFAULT_SELECTIVE_META_TARGET_STEPS = 1
DEFAULT_SELECTIVE_WEIGHT_TEMPERATURE = 1.0
DEFAULT_SELECTIVE_WEIGHT_EMA = 0.7
DEFAULT_SELECTIVE_LIMIT_GRAPHS_PER_CITY = 80


def _slice_metadata(data_dict, indices):
    return {
        "path": [data_dict["path"][i] for i in indices],
        "policy_region": [data_dict["policy_region"][i] for i in indices],
        "scenario": [data_dict["scenario"][i] for i in indices],
        "city": [data_dict["city"][i] for i in indices],
    }


class CrossTReSCityWeightingCallback:
    """
    Approximate CrossTReS-style city selection via target-conditioned meta simulation.
    """

    def __init__(
        self,
        source_city_loaders,
        loss_fct,
        device,
        target_type,
        weight_temperature=1.0,
        ema_coef=0.7,
        eval_steps_per_city=1,
        update_interval=1,
    ):
        self.source_city_loaders = source_city_loaders
        self.loss_fct = loss_fct
        self.device = device
        self.target_type = target_type
        self.weight_temperature = max(weight_temperature, 1e-6)
        self.ema_coef = min(max(ema_coef, 0.0), 0.999)
        self.eval_steps_per_city = max(int(eval_steps_per_city), 1)
        self.update_interval = max(int(update_interval), 1)
        self.city_list = sorted(source_city_loaders.keys())
        self.weights = {city: 1.0 for city in self.city_list}
        self.source_iters = {city: iter(loader) for city, loader in source_city_loaders.items()}

    @staticmethod
    def _next_batch(loader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def _estimate_city_loss(self, model, city_name):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(self.eval_steps_per_city):
                source_batch, self.source_iters[city_name] = self._next_batch(
                    self.source_city_loaders[city_name], self.source_iters[city_name]
                )
                source_batch = source_batch.to(self.device)
                source_target = select_target_tensor(source_batch, self.target_type)
                source_target = model.standardize_target(source_target, data=source_batch)
                pred_source = model(source_batch)
                loss_val = self.loss_fct(pred_source, source_target, source_batch, source_batch.batch).item()
                losses.append(float(loss_val))
        model.train()
        if not losses:
            return 1.0
        return float(sum(losses) / len(losses))

    def __call__(self, epoch, model):
        if epoch % self.update_interval != 0:
            return self.weights

        city_losses = {}
        for city in self.city_list:
            city_losses[city] = self._estimate_city_loss(model=model, city_name=city)

        losses_tensor = torch.tensor([city_losses[c] for c in self.city_list], dtype=torch.float32)
        raw_scores = -losses_tensor / self.weight_temperature
        softmax_weights = torch.softmax(raw_scores, dim=0)
        scaled_weights = softmax_weights * len(self.city_list)

        updated = {}
        for i, city in enumerate(self.city_list):
            candidate = float(scaled_weights[i].item())
            updated[city] = self.ema_coef * self.weights[city] + (1.0 - self.ema_coef) * candidate

        self.weights = updated
        print(f"[CrossTReSCityWeighting] epoch={epoch} weights={self.weights} losses={city_losses}")
        return self.weights
    
def main():
    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--gnn_arch", type=str, default="trans_encoder",
                        help="The GNN architecture to use.",
                        choices=["gatv2", "trans_conv", "graphSAGE", "trans_encoder", "crossST"])  # Add more as you implement them
    parser.add_argument("--use_inductive_variant", type=str_to_bool, default=True,
                        help="Whether to perform inductive or transductive training.")
    parser.add_argument("--project_name", type=str, default=None,
                        help="Override for project directory/WandB project. Defaults automatically based on use_inductive_variant.")
    parser.add_argument("--unique_model_description", type=str, default="trans_encoder_5_features_15_cities",
                        help="A unique description for the run.")
    parser.add_argument("--in_channels", type=int, default=5, help="The number of input channels.")
    parser.add_argument("--use_all_features", type=str_to_bool, default=False, help="Whether to use all features or 5 core features.")
    parser.add_argument("--out_channels", type=int, default=1, help="The number of output channels.")
    parser.add_argument("--model_kwargs", type=str, default=None,
                        help='Additional model parameters (as defined in the class) in JSON format (path to the file).' \
                        'If not provided, defaults params will be used.') 
    parser.add_argument("--loss_fct", type=str, default="mse", help="The loss function to use. Supported: mse, l1.")
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False, help="Whether to use weighted loss (based on vol_base_case) or not.")
    parser.add_argument("--use_city_balanced_loss", type=str_to_bool, default=False,
                        help="Optional for inductive variant: Whether to use city-balanced loss function (based on CityBalancedGNNLoss) or not. \
                            For transductive variant use standard node-weighted loss function (based on GNN_Loss).")
    parser.add_argument("--use_target_standardization", type=str_to_bool, default=False, help="[DEPRECATED] Use --target_normalization instead. Whether to use target standardization during training.")
    parser.add_argument("--target_normalization", type=str, default="None", 
                        help="Target normalization method. Options: 'None' (no normalization), 'relative_to_max_traffic_vol_base_case' (normalize by max vol_base_case per graph), 'relative_standard_scaler' (standardize with mean/std).",
                        choices=["None", "relative_to_max_traffic_vol_base_case", "relative_standard_scaler"])
    parser.add_argument("--target_type", type=str, default="abs_vol_car", help="Which target to use for training.", 
                        choices=["abs_vol_car", "abs_vol_car_percentage", "vol_car_signed_log", "vol_car_percentage_signed_log", "vol_car_mean_std", "vol_car_percentage_mean_std", "vol_car_min_max", "vol_car_percentage_min_max"])
    parser.add_argument("--use_weighted_batches", type=str_to_bool, default=False, help="Whether to use weighted random sampling for training batches.")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    
    #parameters for the learning rate scheduler
    parser.add_argument("--peak_lr", type=float, default=0.001, help="The peak learning rate (after warmup) from which decay will occur.")
    parser.add_argument("--initial_lr", type=float, default=0.0005, help="The initial learning rate from which training will start (used during warmup).")
    parser.add_argument("--warmup_fraction", type=float, default=0.1, help="Fraction of total training steps to use for linear warmup (0.0 to 1.0, e.g., 0.15 = 15%%).")
    parser.add_argument("--cosine_decay_rate", type=float, default=0.5, help="The rate at which the learning rate decays after warmup.")
    parser.add_argument("--min_lr_fraction", type=float, default=0.01, help="The minimum learning rate fraction of the initial learning rate to which the learning rate decays after warmup.")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="The early stopping patience.")
    
    parser.add_argument("--use_dropout", type=str_to_bool, default=False, help="Whether to use dropout.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="After how many steps the gradient should be updated.")
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True, help="Whether to use gradient clipping.")
    parser.add_argument("--device_nr", type=int, default=0, help="The device number (0 or 1 for Retina Roaster's two GPUs).")
    parser.add_argument("--continue_training", type=str_to_bool, default=False, help="Whether to continue training from a checkpoint.")
    parser.add_argument("--base_checkpoint_path", type=str, default=None, help="Path to the checkpoint to continue training from.")
    
    # Parameters for the GraphSAGE
    parser.add_argument("--use_nested_neighbor_loader", type=str_to_bool, default=False, help="Whether to use nested neighbor loader.")
    parser.add_argument("--neighbor_sizes", type=str, default="7,7,7", help="The neighbor sizes for the nested neighbor loader (comma-separated).")
    parser.add_argument("--subgraphs_per_graph", type=int, default=1, help="The number of subgraphs to sample per graph.")
    parser.add_argument("--seed_size", type=int, default=1000, help="The number of seed nodes in each subgraph.")
    parser.add_argument("--sampling_strategy", type=str, default="neighbor_sampling", help="The sampling strategy to use for the nested neighbor loader.",
                        choices=["neighbor_sampling", "random_walk"])
    parser.add_argument("--min_subgraph_nodes", type=int, default=5000, help="The minimum number of nodes in a subgraph.")
    parser.add_argument("--max_subgraph_nodes", type=int, default=50000, help="The maximum number of nodes in a subgraph.")
    
    # Parameters for Data Augmentation
    parser.add_argument("--aug_pos_rotation", type=str_to_bool, default=False, help="Whether to use Position Rotation augmentation.")
    parser.add_argument("--aug_feature_noise", type=str_to_bool, default=False, help="Whether to use Gaussian noise addition to node features as data augmentation.")
    parser.add_argument("--aug_node_masking_probability", type=float, default=0.0, help="The probability of masking all features of a node to 0 during training. 0.0 means no node masking.")

    # Fast-iteration: optionally cap dataset sizes per split (random subsample)
    parser.add_argument("--limit_available_graphs", type=int, default=2000, help="If >0, randomly keep only this many available graphs after reading metadata (applies before splitting into train/val/test).")
    parser.add_argument("--apply_source_city_weighting_crosstres", type=str_to_bool, default=False,
                        help="Enable CrossTReS-style CITY-level selective source weighting during pretraining.")
    parser.add_argument("--crossst_alpha", type=float, default=0.3,
                        help="CrossST-inspired temporal distillation weight (used in finetuning mode).")
    parser.add_argument("--crossst_beta", type=float, default=0.3,
                        help="CrossST-inspired spatial distillation weight (used in finetuning mode).")

    args = vars(parser.parse_args())
    
    # Parse neighbor_sizes from string to list
    if isinstance(args['neighbor_sizes'], str):
        args['neighbor_sizes'] = [int(x.strip()) for x in args['neighbor_sizes'].split(',')]
    
    # Convert "None" string to None
    if args.get('target_normalization') == "None":
        args['target_normalization'] = None

    # Backward compatibility: map use_target_standardization to target_normalization
    if args.get('target_normalization') is None and args.get('use_target_standardization', False):
        args['target_normalization'] = "relative_standard_scaler"
        print("Warning: --use_target_standardization is deprecated. Using --target_normalization='relative_standard_scaler'")
    
    set_random_seeds()

    if args.get("apply_source_city_weighting_crosstres", False):
        run_desc = str(args.get("unique_model_description", "") or "")
        if "crosstres" not in run_desc.lower():
            args["unique_model_description"] = f"{run_desc}_crosstres" if run_desc else "crosstres"
    
    if args.get('project_name') is None:
        args['project_name'] = "GNN_Inductive" if args['use_inductive_variant'] else "GNN_Transductive"
    
    
    try:
        
        # Continue with GPU setup and model training
        gpus = get_available_gpus()
        best_gpu = select_best_gpu(gpus)
        set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create directory for the run
        unique_run_dir = os.path.join(base_dir, args['project_name'], args['unique_model_description'])
        os.makedirs(unique_run_dir, exist_ok=True)
        
        model_save_path, path_to_save_dataloader = get_paths(base_dir=os.path.join(base_dir, args['project_name']),
                                                             unique_model_description=args['unique_model_description'],
                                                             model_save_path='trained_model/model.pth')
        
        # Start data prepration
        train_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city':list()}    
        for city in sorted(train_cities):
            load_metadata_from_disk(train_data, os.path.join(dataset_path, city, 'metadata.json'))

        if args['use_inductive_variant']:
            if len(val_cities) > 0:
                val_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city':list()}
                for city in sorted(val_cities):
                    load_metadata_from_disk(val_data, os.path.join(dataset_path, city, 'metadata.json'))
            else:
                val_data = None

            if len(test_cities) > 0:
                test_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city':list()}
                for city in sorted(test_cities):
                    load_metadata_from_disk(test_data, os.path.join(dataset_path, city, 'metadata.json'))
            else:
                test_data = None
            
            # For inductive variant: if limit_available_graphs is set, limit each city group separately
            # to preserve city-based separation (no mixing of train/val/test cities)
            if args.get('limit_available_graphs', 0) and args['limit_available_graphs'] > 0:
                # Calculate proportional limits for each split to maintain train/val/test ratio
                total_original = len(train_data['path']) + (len(val_data['path']) if val_data else 0) + (len(test_data['path']) if test_data else 0)
                if total_original > 0:
                    train_ratio_actual = len(train_data['path']) / total_original
                    val_ratio_actual = len(val_data['path']) / total_original if val_data else 0
                    test_ratio_actual = len(test_data['path']) / total_original if test_data else 0
                    
                    # Limit each split proportionally
                    train_limit = max(1, int(args['limit_available_graphs'] * train_ratio_actual))
                    val_limit = max(1, int(args['limit_available_graphs'] * val_ratio_actual)) if val_data else 0
                    test_limit = max(1, int(args['limit_available_graphs'] * test_ratio_actual)) if test_data else 0
                    
                    # Apply limits to each city group separately to preserve city separation
                    train_data = balanced_subset_by_city(train_data, train_limit)
                    if val_data:
                        val_data = balanced_subset_by_city(val_data, val_limit)
                    if test_data:
                        test_data = balanced_subset_by_city(test_data, test_limit)
                    
                    print(f"[DEBUG] Limited data while preserving city separation: train={len(train_data['path'])}, val={len(val_data['path']) if val_data else 0}, test={len(test_data['path']) if test_data else 0}")
            
            # Verify no data leakage: ensure validation/test cities don't appear in training
            if args['use_inductive_variant']:
                train_cities_set = set(train_data['city'])
                val_cities_set = set(val_data['city']) if val_data else set()
                test_cities_set = set(test_data['city']) if test_data else set()
                
                leakage_val = train_cities_set & val_cities_set
                leakage_test = train_cities_set & test_cities_set
                
                if leakage_val:
                    raise ValueError(f"DATA LEAKAGE DETECTED: Validation cities {leakage_val} appear in training data!")
                if leakage_test:
                    raise ValueError(f"DATA LEAKAGE DETECTED: Test cities {leakage_test} appear in training data!")
                
                print(f"[VERIFICATION] ✓ No data leakage confirmed:")
                print(f"  Training cities ({len(train_cities_set)}): {sorted(train_cities_set)}")
                print(f"  Validation cities ({len(val_cities_set)}): {sorted(val_cities_set)}")
                print(f"  Test cities ({len(test_cities_set)}): {sorted(test_cities_set)}")
                print(f"  Training graphs: {len(train_data['path'])}, Validation graphs: {len(val_data['path']) if val_data else 0}, Test graphs: {len(test_data['path']) if test_data else 0}")
        else:
            # Transductive variant: validation/test splits derived from training cities.
            val_data = None
            test_data = None
            
            # Optional: Subsample training graphs for faster iterations
            if args.get("apply_source_city_weighting_crosstres", False):
                # In CrossTReS benchmark mode, always use all source-city graphs for pretraining,
                # regardless of the global fast-iteration default.
                pass
            elif args['limit_available_graphs'] > 0:
                train_data = balanced_subset_by_city(train_data, args['limit_available_graphs'])

        print(f"Using {'INDUCTIVE' if args['use_inductive_variant'] else 'TRANSDUCTIVE'} data preparation!")
        transductive_val_ratio = 0.15
        transductive_test_ratio = 0.05
        if (not args["use_inductive_variant"]) and args.get("apply_source_city_weighting_crosstres", False):
            # For benchmark fairness in CrossTReS mode: use all source-city graphs for train+val,
            # and keep no internal test holdout during pretraining.
            transductive_val_ratio = 0.2
            transductive_test_ratio = 0.0

        train_dl, valid_dl, scalers_train = prepare_data_with_graph_features(train_data=train_data,
                                                                             val_data=val_data,
                                                                             test_data=test_data,
                                                                             use_inductive_variant=args['use_inductive_variant'], # Conditional (Transductive/Inductive)
                                                                             batch_size=args['batch_size'],
                                                                             path_to_save_dataloader=path_to_save_dataloader,
                                                                             use_all_features=args['use_all_features'],
                                                                             use_weighted_batches=args['use_weighted_batches'],
                                                                             use_nested_neighbor_loader=args['use_nested_neighbor_loader'],
                                                                             neighbor_sizes=args['neighbor_sizes'],
                                                                             subgraphs_per_graph=args['subgraphs_per_graph'],
                                                                             seed_size=args['seed_size'],
                                                                             sampling_strategy=args['sampling_strategy'],
                                                                             min_subgraph_nodes=args['min_subgraph_nodes'],
                                                                             max_subgraph_nodes=args['max_subgraph_nodes'],
                                                                             aug_pos_rotation=args['aug_pos_rotation'],
                                                                             aug_feature_noise=args['aug_feature_noise'],
                                                                             aug_node_masking_probability=args['aug_node_masking_probability'],
                                                                             transductive_val_ratio=transductive_val_ratio,
                                                                             transductive_test_ratio=transductive_test_ratio)

        # Create WandB config
        config = setup_wandb(args)

        if args["model_kwargs"] is not None:
            with open(args["model_kwargs"], 'r') as f:
                model_kwargs = json.load(f)
        else:
            model_kwargs = {}

        # CRITICAL: Get actual data feature count from a batch (after collate_fn filtering)
        # The collate_fn filters features, so we need to check the batch, not the raw dataset
        sample_batch = next(iter(train_dl))
        actual_feature_count = sample_batch.x.shape[1]
        
        # Also check raw dataset for comparison
        raw_dataset_feature_count = train_dl.dataset[0].x.shape[1] if hasattr(train_dl, 'dataset') else None
        
        print(f"\n{'='*60}")
        print(f"Data feature analysis:")
        print(f"  Raw dataset feature count: {raw_dataset_feature_count}")
        print(f"  DataLoader batch feature count (after filtering): {actual_feature_count}")
        print(f"  Config in_channels parameter: {config.in_channels}")
        print(f"  use_all_features setting: {args['use_all_features']}")
        if args['use_all_features']:
            print(f"  Expected: All features (typically 20 without destination activity, or 28 with)")
        else:
            print(f"  Expected: 5 base features (VOL_BASE_CASE, CAPACITY_BASE_CASE, CAPACITY_REDUCTION, FREESPEED, LENGTH)")
        print(f"{'='*60}\n")
        
        # For trans_encoder, the model adds positional encoding in forward()
        # So in_channels should match the base feature count (actual_feature_count)
        # The model internally calculates effective_in_channels = in_channels + pos_dim
        if config.gnn_arch == 'trans_encoder':
            if config.in_channels != actual_feature_count:
                print(f"⚠️  Batch has {actual_feature_count} base features, but config.in_channels={config.in_channels}")
                print(f"   For trans_encoder, in_channels should match base feature count (before positional encoding)")
                print(f"   Will override in_channels to {actual_feature_count} when creating model")
                # Update wandb config with allow_val_change
                try:
                    config.update({'in_channels': actual_feature_count}, allow_val_change=True)
                except Exception as e:
                    print(f"   Warning: Could not update wandb config: {e}")
                    print(f"   Will pass in_channels={actual_feature_count} directly to model")
                # Store in model_kwargs to override
                model_kwargs['in_channels'] = actual_feature_count
        else:
            # For other architectures, check if they match
            if config.in_channels != actual_feature_count:
                print(f"⚠️  Batch has {actual_feature_count} features, but config.in_channels={config.in_channels}")
                print(f"   Will override in_channels to {actual_feature_count} when creating model")
                # Update wandb config with allow_val_change
                try:
                    config.update({'in_channels': actual_feature_count}, allow_val_change=True)
                except Exception as e:
                    print(f"   Warning: Could not update wandb config: {e}")
                    print(f"   Will pass in_channels={actual_feature_count} directly to model")
                # Store in model_kwargs to override
                model_kwargs['in_channels'] = actual_feature_count
        
        # Create model instance
        gnn_instance = create_gnn_model(gnn_arch=config.gnn_arch,
                                        config=config,
                                        model_kwargs=model_kwargs,
                                        device=device)

        # LOSS FUNCTION
        if args.get('use_city_balanced_loss', False):
            loss_fct = CityBalancedGNNLoss(loss_fct=config.loss_fct, 
                                           device=device, 
                                           weighted=config.use_weighted_loss,
                                           num_nodes=train_dl.dataset[0].x.shape[0])
            print("Using city-balanced loss function- INDUCTIVE VARIANT")
        else:
            loss_fct = GNN_Loss(loss_fct=config.loss_fct,
                                device=device, 
                                weighted=config.use_weighted_loss,
                                num_nodes=train_dl.dataset[0].x.shape[0])
            print("Using standard loss function - TRANSDUCTIVE VARIANT")

        early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)

        print(f"Training method: {'INDUCTIVE' if args['use_inductive_variant'] else 'TRANSDUCTIVE'}")

        selective_weight_callback = None
        dynamic_city_weights = {}
        use_crosstres_city_weighting = args.get("apply_source_city_weighting_crosstres", False)
        if use_crosstres_city_weighting:
            forbidden_targets = set(test_cities or [])
            overlapping_cities = forbidden_targets.intersection(set(train_cities or []))
            if overlapping_cities:
                raise ValueError(
                    f"CrossTReSCityWeighting requires strict source-only pretraining. "
                    f"Target city/cities found in train_cities: {sorted(overlapping_cities)}"
                )

            source_city_loaders = {}
            meta_loader_dir = os.path.join(path_to_save_dataloader, "cross_tres_city_weighting")
            os.makedirs(meta_loader_dir, exist_ok=True)

            for city in sorted(train_cities):
                city_meta = {"path": [], "policy_region": [], "scenario": [], "city": []}
                load_metadata_from_disk(city_meta, os.path.join(dataset_path, city, "metadata.json"))
                if DEFAULT_SELECTIVE_LIMIT_GRAPHS_PER_CITY > 0:
                    city_meta = balanced_subset_by_city(
                        city_meta, min(DEFAULT_SELECTIVE_LIMIT_GRAPHS_PER_CITY, len(city_meta["path"]))
                    )
                city_loader_dir = os.path.join(meta_loader_dir, f"{city}_source")
                os.makedirs(city_loader_dir, exist_ok=True)
                city_train_loader, _, _ = prepare_data_with_graph_features(
                    train_data=city_meta,
                    val_data=None,
                    test_data=None,
                    use_inductive_variant=False,
                    batch_size=args["batch_size"],
                    path_to_save_dataloader=city_loader_dir + "/",
                    use_all_features=args["use_all_features"],
                    use_weighted_batches=False,
                    use_nested_neighbor_loader=False,
                    neighbor_sizes=args["neighbor_sizes"],
                    subgraphs_per_graph=args["subgraphs_per_graph"],
                    seed_size=args["seed_size"],
                    sampling_strategy=args["sampling_strategy"],
                    min_subgraph_nodes=args["min_subgraph_nodes"],
                    max_subgraph_nodes=args["max_subgraph_nodes"],
                    aug_pos_rotation=False,
                    aug_feature_noise=False,
                    aug_node_masking_probability=0.0,
                    transductive_val_ratio=0.0,
                    transductive_test_ratio=0.0,
                )
                source_city_loaders[city] = city_train_loader
                dynamic_city_weights.setdefault(city, 1.0)

            selective_weight_callback = CrossTReSCityWeightingCallback(
                source_city_loaders=source_city_loaders,
                loss_fct=loss_fct,
                device=device,
                target_type=config.target_type,
                weight_temperature=DEFAULT_SELECTIVE_WEIGHT_TEMPERATURE,
                ema_coef=DEFAULT_SELECTIVE_WEIGHT_EMA,
                eval_steps_per_city=DEFAULT_SELECTIVE_META_SOURCE_STEPS,
                update_interval=DEFAULT_SELECTIVE_META_UPDATE_INTERVAL,
            )

        best_val_loss, best_epoch = gnn_instance.train_model(config=config,
                                                             loss_fct=loss_fct,
                                                             optimizer=torch.optim.AdamW(gnn_instance.parameters(), lr=config.peak_lr, weight_decay=1e-3) if config.gnn_arch != "xgboost" else None,
                                                             train_dl=train_dl,
                                                             valid_dl=valid_dl,
                                                             device=device,
                                                             early_stopping=early_stopping,
                                                             model_save_path=model_save_path,
                                                             apply_source_city_weights=use_crosstres_city_weighting,
                                                             source_city_weights=dynamic_city_weights,
                                                             city_weight_callback=selective_weight_callback)
        
        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   
        print_model_info(gnn_instance)

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""


if __name__ == '__main__':
    main()
