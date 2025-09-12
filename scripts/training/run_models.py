'''
Run GNN model training with configurable architecture and hyperparameters.

'dataset_path' and 'base_dir' need to be adjusted to the correct paths.
All the other parameters can be passed as command line arguments. Run `python run_models.py --help` to see the list of available arguments.

Example usage with default architecture, dropout, and most significant features found using ablation tests:
`python run_models.py --in_channels 5 --use_all_features False --num_epochs 500 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3`

Our use case:
python run_models.py --gnn_arch gatv2 --unique_model_description trial13 --in_channels 5 --use_all_features True --num_epochs 2 --lr 0.003 --early_stopping_patience 25
python run_models.py --gnn_arch graphSAGE --unique_model_description graphSAGE_5_features_16_cities_retina --in_channels 5 --use_all_features True --num_epochs 2 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3 --use_nested_neighbor_loader True --neighbor_sizes 5,5,5,5,5 
python run_models.py --gnn_arch eign --unique_model_description EIGN_trial_unsigned --in_channels 5 --use_all_features True --num_epochs 2 --lr 0.003 --early_stopping_patience 25 
python run_models.py --gnn_arch trans_encoder --unique_model_description encoder_trial --in_channels 5 --use_all_features True --num_epochs 20 --lr 0.0003 --early_stopping_patience 25
'''


from html import parser
import os
import sys
import json
import argparse

import torch
#from torchinfo import summary
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # This is to avoid memory issues in Retina. Comment it out in LRZ AI

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from training.help_functions import *
from gnn.help_functions import GNN_Loss, compute_baseline_of_mean_target, compute_baseline_of_no_policies

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Please adjust as needed
base_dir = os.path.join(project_root, 'inductive_gnn_data_results','transductive') # for saving results
#cities = ['wuerzburg','aschaffenburg','regensburg','landshut','bayreuth','erlangen','fuerth','kempten','neuulm','muenchen','augsburg','rosenheim','schweinfurt','bamberg','nuernberg', 'ingolstadt']
train_cities = ['wuerzburg','aschaffenburg','regensburg','bayreuth']
val_cities =['rosenheim','landshut'] # Non empty implies inductive learning
test_cities = ['schweinfurt'] # Non empty implies inductive learning
    
def main():
    
    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--gnn_arch", type=str, default="trans_conv",
                        help="The GNN architecture to use.",
                        choices=["point_net_transf_gat", "gat", "gatv2", "gatv3", "gcn", "gcn2", "trans_conv", "pnc", "fc_nn", "graphSAGE", "eign", "xgboost","trans_encoder"])  # Add more as you implement them
    parser.add_argument("--project_name", type=str, default="GNN_Transductive",
                        help="The name of the project, used for saving the corresponding runs, and as the WandB project name.")
    parser.add_argument("--unique_model_description", type=str, default="trans_conv_5_features_16_cities",
                        help="A unique description for the run.")
    parser.add_argument("--in_channels", type=int, default=5, help="The number of input channels.")
    parser.add_argument("--use_all_features", type=str_to_bool, default=True, help="Whether to use all features(True) or a subset of features(False).")
    parser.add_argument("--out_channels", type=int, default=1, help="The number of output channels.")
    parser.add_argument("--model_kwargs", type=str, default=None,
                        help='Additional model parameters (as defined in the class) in JSON format (path to the file).' \
                        'If not provided, defaults params will be used.') 
    parser.add_argument("--loss_fct", type=str, default="mse", help="The loss function to use. Supported: mse, l1.")
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False, help="Whether to use weighted loss (based on vol_base_case) or not.")
    parser.add_argument("--target_normalization", type=str_to_bool, default=False, help="Whether targets are normalized during preprocessing.")
    parser.add_argument("--predict_mode_stats", type=str_to_bool, default=False, help="Whether to predict mode stats or not.")
    parser.add_argument("--target_type", type=str, default="abs_vol_car", help="Which target to use for training.", 
                        choices=["abs_vol_car", "abs_vol_car_percentage", "vol_car_signed_log", "vol_car_percentage_signed_log", "vol_car_mean_std", "vol_car_percentage_mean_std", "vol_car_min_max", "vol_car_percentage_min_max"])
    parser.add_argument("--use_bootstrapping", type=str_to_bool, default=False, help="Whether to use bootstrapping for train-validation split.")
    parser.add_argument("--use_weighted_sampling", type=str_to_bool, default=False, help="Whether to use weighted random sampling for training.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate for the model.")
    parser.add_argument("--early_stopping_patience", type=int, default=25, help="The early stopping patience.")
    parser.add_argument("--use_dropout", type=str_to_bool, default=False, help="Whether to use dropout.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True, help="Whether to use gradient clipping.")
    parser.add_argument("--device_nr", type=int, default=0, help="The device number (0 or 1 for Retina Roaster's two GPUs).")
    parser.add_argument("--continue_training", type=str_to_bool, default=False, help="Whether to continue training from a checkpoint.")
    parser.add_argument("--base_checkpoint_path", type=str, default=None, help="Path to the checkpoint to continue training from.")
    #parameters for the GraphSAGE
    parser.add_argument("--use_nested_neighbor_loader", type=str_to_bool, default=False, help="Whether to use nested neighbor loader.") # TODO: New for GraphSAGE
    parser.add_argument("--neighbor_sizes", type=str, default="5,5,5", help="The neighbor sizes for the nested neighbor loader (comma-separated).") # TODO: New for GraphSAGE
    parser.add_argument("--subgraphs_per_graph", type=int, default=2, help="The number of subgraphs to sample per graph.") # TODO: New for GraphSAGE
    parser.add_argument("--seed_size", type=int, default=10, help="The number of seed nodes in each subgraph.") # TODO: New for GraphSAGE
    parser.add_argument("--sampling_strategy", type=str, default="neighbor_sampling", help="The sampling strategy to use for the nested neighbor loader.",
                        choices=["neighbor_sampling", "random_walk"]) # TODO: New for GraphSAGE
    parser.add_argument("--min_subgraph_nodes", type=int, default=500, help="The minimum number of nodes in a subgraph.") # TODO: New for GraphSAGE
    parser.add_argument("--max_subgraph_nodes", type=int, default=50000, help="The maximum number of nodes in a subgraph.") # TODO: New for GraphSAGE
    #parameter for Data Augmentation
    parser.add_argument("--use_data_augmentation", type=str_to_bool, default=False, help="Whether to use data augmentation.")
    parser.add_argument("--use_edge_perturbation_probability", type=float, default=0.0, help="The probability of edge perturbation (random dropout on line graph edges) during training. 0.0 means no perturbation.")
    #for Gaussian noise addition to node features
    parser.add_argument("--augment_feature_noise_prob", type=float, default=0.0, help="Probability of applying Gaussian noise to features (0.0-1.0)")

    #parser for DANN
    parser.add_argument("--use_dann", type=str_to_bool, default=False, help="Whether to use Domain Adversarial Neural Network.")
    parser.add_argument("--domain_lambda", type=float, default=0.0, help="Weight for domain adversarial loss.")
    parser.add_argument("--use_target_standardization", type=str_to_bool, default=False, help="Whether to use target standardization during training.")

    args = vars(parser.parse_args())
    
    # Parse neighbor_sizes from string to list
    if isinstance(args['neighbor_sizes'], str):
        args['neighbor_sizes'] = [int(x.strip()) for x in args['neighbor_sizes'].split(',')]
    
    set_random_seeds()
    
    if args['gnn_arch'] == "eign":
        dataset_path = os.path.join(project_root, 'data','inductive_data','training_data_eign','kreisfreistadt')
        base_dir = os.path.join(project_root, 'inductive_gnn_data_results','transductive_eign') # for saving results
    else:
        dataset_path = os.path.join(project_root, 'data','inductive_data','training_data','kreisfreistadt')
        base_dir = os.path.join(project_root, 'inductive_gnn_data_results','transductive') # for saving results
    
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

        if args['project_name'] == 'GNN_Transductive':
            train_dl, valid_dl, scalers_train = prepare_data_with_graph_features(train_data=train_data,
                                                                             val_data=val_data,
                                                                             test_data=test_data,
                                                                             variant=args['project_name'],
                                                                             batch_size=args['batch_size'],
                                                                             path_to_save_dataloader=path_to_save_dataloader,
                                                                             use_all_features=args['use_all_features'],
                                                                             use_bootstrapping=args['use_bootstrapping'],
                                                                             use_weighted_sampling=args['use_weighted_sampling'],
                                                                             use_nested_neighbor_loader=args['use_nested_neighbor_loader'],
                                                                             neighbor_sizes=args['neighbor_sizes'],
                                                                             subgraphs_per_graph=args['subgraphs_per_graph'],
                                                                             seed_size=args['seed_size'],
                                                                             sampling_strategy=args['sampling_strategy'],
                                                                             min_subgraph_nodes=args['min_subgraph_nodes'],
                                                                             max_subgraph_nodes=args['max_subgraph_nodes'],
                                                                             is_eign=(args['gnn_arch'] == "eign"),
                                                                             use_data_augmentation=args['use_data_augmentation'],
                                                                             use_edge_perturbation_probability=args['use_edge_perturbation_probability'],
                                                                             use_feature_noise_probability=args['augment_feature_noise_prob'])
        else: #for 'GNN_Inductive' as project name
            train_dl, valid_dl, scalers_train, scalers_validation = prepare_data_with_graph_features(train_data=train_data,
                                                                                                        val_data=val_data,
                                                                                                        test_data=test_data,
                                                                                                        variant=args['project_name'],
                                                                                                        batch_size=args['batch_size'],
                                                                                                        path_to_save_dataloader=path_to_save_dataloader,
                                                                                                        use_all_features=args['use_all_features'],
                                                                                                        use_bootstrapping=args['use_bootstrapping'],
                                                                                                        use_weighted_sampling=args['use_weighted_sampling'],
                                                                                                        use_nested_neighbor_loader=args['use_nested_neighbor_loader'],
                                                                                                        neighbor_sizes=args['neighbor_sizes'],
                                                                                                        subgraphs_per_graph=args['subgraphs_per_graph'],
                                                                                                        seed_size=args['seed_size'],
                                                                                                        sampling_strategy=args['sampling_strategy'],
                                                                                                        min_subgraph_nodes=args['min_subgraph_nodes'],
                                                                                                        max_subgraph_nodes=args['max_subgraph_nodes'],
                                                                                                        is_eign=(args['gnn_arch'] == "eign"),
                                                                                                        use_data_augmentation=args['use_data_augmentation'],
                                                                                                        use_edge_perturbation_probability=args['use_edge_perturbation_probability'],
                                                                                                        use_feature_noise_probability=args['augment_feature_noise_prob'])

        # Create WandB config
        config = setup_wandb(args)

        if args["model_kwargs"] is not None:
            with open(args["model_kwargs"], 'r') as f:
                model_kwargs = json.load(f)
        else:
            model_kwargs = {}
        
        # Create model instance
        gnn_instance = create_gnn_model(gnn_arch=config.gnn_arch,
                                        config=config,
                                        model_kwargs=model_kwargs,
                                        device=device).to(device)
        
        loss_fct = GNN_Loss(loss_fct=config.loss_fct, num_nodes=train_dl.dataset[0].x.shape[0],
                            device=device, weighted=config.use_weighted_loss)
        
        ## Not needed now, Naive MSE doesn't tell anything!
        # baseline_loss_mean_target = compute_baseline_of_mean_target(dataset=train_dl, loss_fct=loss_fct, device=device, scalers=scalers_train)
        # baseline_loss = compute_baseline_of_no_policies(dataset=train_dl, loss_fct=loss_fct, device=device, scalers=scalers_train)
        # print("baseline loss mean " + str(baseline_loss_mean_target))
        # print("baseline loss no  " + str(baseline_loss) )

        early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)
        if args['project_name'] == 'GNN_Transductive':
            best_val_loss, best_epoch = gnn_instance.train_model(config=config,
                                                                 loss_fct=loss_fct,
                                                                 optimizer=torch.optim.AdamW(gnn_instance.parameters(), lr=config.lr, weight_decay=1e-4) if config.gnn_arch != "xgboost" else None,
                                                                 train_dl=train_dl,
                                                                 valid_dl=valid_dl,
                                                             device=device,
                                                             early_stopping=early_stopping,
                                                             model_save_path=model_save_path,
                                                             scalers_train=scalers_train,
                                                             target_normalization=config.target_normalization)
        else:
            best_val_loss, best_epoch = gnn_instance.train_model_inductive(config=config,
                                                             loss_fct=loss_fct,
                                                             optimizer=torch.optim.AdamW(gnn_instance.parameters(), lr=config.lr, weight_decay=1e-3) if config.gnn_arch != "xgboost" else None,
                                                             train_dl=train_dl,
                                                             valid_dl=valid_dl,
                                                             device=device,
                                                             early_stopping=early_stopping,
                                                             model_save_path=model_save_path,
                                                             scalers_train=scalers_train,
                                                             scalers_validation=scalers_train)
        
        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   
        print_model_info(gnn_instance)

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""


if __name__ == '__main__':
    main()
