import os
import sys
import copy
import random
import joblib
import subprocess

import numpy as np
from tqdm import tqdm
import wandb
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from collections import Counter


# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.gnn_io import *
from gnn.models.point_net_transf_gat import PointNetTransfGAT
from gnn.models.gcn import GCN, GCN2
from gnn.models.gat import GAT
from gnn.models.gatv2 import GATv2
from gnn.models.trans_conv import TransConv
from gnn.models.pnc import PNC
from gnn.models.fc_nn import FC_NN
from gnn.models.eign import EIGNLaplacianConv
from gnn.models.graphSAGE import GraphSAGE
from gnn.models.xgboost import XGBoostModel
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures, use_allowed_modes
from gnn.gnn_io import collate_fn

def get_available_gpus():
    command = "nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Error executing nvidia-smi: {result.stderr.decode('utf-8')}")
    gpu_info = result.stdout.decode('utf-8').strip().split('\n')
    gpus = []
    for info in gpu_info:
        index, utilization, memory_free = info.split(', ')
        gpus.append({
            'index': int(index),
            'utilization': int(utilization),
            'memory_free': int(memory_free)
        })
    return gpus
    
def select_best_gpu(gpus):
    # Sort by free memory (descending) and then by utilization (ascending)
    gpus = sorted(gpus, key=lambda x: (-x['memory_free'], x['utilization']))
    return gpus[0]['index']

def set_cuda_visible_device(gpu_index):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    print(f"Using GPU {gpu_index} with CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

def str_to_bool(value):
    if isinstance(value, str):
        if value.lower() in ['true', '1', 'yes', 'y']:
            return True
        elif value.lower() in ['false', '0', 'no', 'n']:
            return False
    raise ValueError(f"Cannot convert {value} to a boolean.")
    
def set_random_seeds(seed_value=42):
    # Set environment variable for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # Set Python built-in random module seed
    random.seed(seed_value)
    
    # Set NumPy random seed
    np.random.seed(seed_value)
    
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed_value)
    
    # Set PyTorch random seed for all GPUs (if available)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If using multi-GPU
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # If using torch.distributed for distributed training, set the seed
    if torch.distributed.is_initialized():
        torch.distributed.manual_seed_all(seed_value)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
        
def get_paths(base_dir: str, unique_model_description: str, run_name: str, model_save_path: str = 'trained_model/model.pth'):
    data_path = os.path.join(base_dir, unique_model_description, run_name)
    os.makedirs(data_path, exist_ok=True)
    model_save_to = os.path.join(data_path, model_save_path)
    path_to_save_dataloader = os.path.join(data_path, 'data_created_during_training/')
    os.makedirs(os.path.dirname(model_save_to), exist_ok=True)
    os.makedirs(path_to_save_dataloader, exist_ok=True)
    return model_save_to, path_to_save_dataloader

def get_memory_info():
    """Get memory information using psutil."""
    import psutil
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
    used_memory = total_memory - available_memory
    return total_memory, available_memory, used_memory

def parse_filepath_metadata(filepath):
    """
    Parse city and hex_size from filepath string.
    Example: 'augsburg/datalist_batch_500_1.pt' -> city='augsburg', hex_size='500'
    Assumes filename format: datalist_batch_{hex_size}_{counter}.pt
    """
    city = os.path.basename(os.path.dirname(filepath))
    hex_size = os.path.basename(filepath).split('_')[2]
    return city, hex_size

def fixed_proportion_sampler(filepaths, batch_size=16, threshold=100):
    """
    Create a sampler that maintains city proportions within each batch.
    
    Steps:
    1. Group filepaths by city
    2. Calculate global weights (city_count / total_count)
    3. For each batch, determine how many graphs from each city based on weights
    4. Sample with/without replacement based on city size threshold
    
    Args:
        filepaths: List of file paths
        batch_size: Target batch size for maintaining proportions
        threshold: If city has > threshold graphs, sample without replacement
    """
    
    # Step 1: Group filepaths by city into dictionary
    city_filepaths = {}
    for filepath in filepaths:
        city = parse_filepath_metadata(filepath)[0]
        if city not in city_filepaths:
            city_filepaths[city] = []
        city_filepaths[city].append(filepath)
    
    # Step 2: Calculate global weights for each city
    total_graphs = len(filepaths)
    city_weights = {}
    city_counts = {}
    
    for city, paths in city_filepaths.items():
        city_counts[city] = len(paths)
        city_weights[city] = len(paths) / total_graphs
    
    print(f"City distribution:")
    for city, count in city_counts.items():
        print(f"  {city}: {count} graphs ({city_weights[city]:.3f})")
    
    # Step 3 & 4: Create sampling weights for each filepath
    weights = []
    
    for filepath in filepaths:
        city = parse_filepath_metadata(filepath)[0]
        
        # Step 3: Calculate how many graphs this city should have in a batch
        target_in_batch = city_weights[city] * batch_size
        
        # Step 4: Determine sampling strategy based on city size
        if city_counts[city] > threshold:
            # Large city: sample without replacement (standard proportional weight)
            weight = city_weights[city]
        else:
            # Small city: sample with replacement (boost weight to allow duplicates)
            # For small cities, we need higher weight to reach target proportion
            weight = city_weights[city]
            # Boost inversely proportional to city size
            if city_counts[city] <= threshold:
                boost = threshold / city_counts[city]  # Smaller cities get bigger boost
                weight = weight * boost
        
        weights.append(weight)
    
    # Normalize weights so they sum to len(filepaths)
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight * len(filepaths) for w in weights]
    else:
        weights = [1.0] * len(filepaths)
    
    print(f"Sampling strategy:")
    for city, count in city_counts.items():
        strategy = "without replacement" if count > threshold else "with replacement"
        target = city_weights[city] * batch_size
        print(f"  {city}: {count} graphs, target {target:.1f} per batch ({strategy})")
    
    return WeightedRandomSampler(weights, num_samples=len(filepaths), replacement=True)

def nested_fixed_proportion_sampler(dataset):
    """
    Create a sampler that ensures fixed proportion of cities and hex_sizes in batches with replacement.
    Uses two-level sampling: first sample cities globally, then sample hex_sizes locally within each city.
    """
    def get_key(graph): return (graph.city, graph.hex_size)
    
    groups = {}
    for graph in dataset:
        key = get_key(graph)
        if key not in groups:
            groups[key] = []
        groups[key].append(graph)
    
    # Calculate global city distribution
    city_counts = {}
    for (city, hex_size), graphs in groups.items():
        if city not in city_counts:
            city_counts[city] = 0
        city_counts[city] += len(graphs)
    
    total_graphs = len(dataset)
    global_weights = []
    for (city, hex_size), graphs in groups.items():
        # Global weight: probability of this city
        global_weight = city_counts[city] / total_graphs
        global_weights.append(global_weight)
    
    # Calculate local hex_size distribution within each city
    city_hex_counts = {}
    for (city, hex_size), graphs in groups.items():
        if city not in city_hex_counts:
            city_hex_counts[city] = {}
        city_hex_counts[city][hex_size] = len(graphs)
    
    local_weights = []
    for (city, hex_size), graphs in groups.items():
        # Local weight: probability of this hex_size given this city
        total_city_graphs = sum(city_hex_counts[city].values())
        local_weight = len(graphs) / total_city_graphs
        local_weights.append(local_weight)
    
    # Combine global and local weights (product of probabilities)
    combined_weights = [global_w * local_w for global_w, local_w in zip(global_weights, local_weights)]
    
    # DEBUG: Print weight calculations
    print("DEBUG SAMPLER: Weight calculations:")
    for i, ((city, hex_size), graphs) in enumerate(groups.items()):
        print(f"  {city}-{hex_size}: {len(graphs)} graphs, global_w={global_weights[i]:.3f}, local_w={local_weights[i]:.3f}, combined_w={combined_weights[i]:.3f}")
    
    # DEBUG: Check if weights sum to 1
    total_weight = sum(combined_weights)
    print(f"DEBUG SAMPLER: Total weight = {total_weight:.3f}")
    
    # DEBUG: Check if any weights are 0
    zero_weights = [i for i, w in enumerate(combined_weights) if w == 0]
    if zero_weights:
        print(f"DEBUG SAMPLER: Zero weights found at indices: {zero_weights}")
    
    sampler = WeightedRandomSampler(weights=combined_weights, num_samples=len(dataset), replacement=True)
    return sampler

def test_sampler_distribution(dataset, sampler, num_samples=100):
    """
    Test the sampler to see if it produces the expected distribution.
    """
    print(f"\n=== Testing Sampler Distribution ({num_samples} samples) ===")
    
    # Create a larger dataset by duplicating the original dataset to test replacement
    # This allows us to sample more than the original dataset size
    from torch.utils.data import ConcatDataset
    import copy
    
    # Create multiple copies of the dataset to allow sampling with replacement
    num_copies = (num_samples // len(dataset)) + 2  # Extra copies to ensure we can sample enough
    datasets = [copy.deepcopy(dataset) for _ in range(num_copies)]
    large_dataset = ConcatDataset(datasets)
    
    # Create a new sampler for the large dataset
    from torch.utils.data import WeightedRandomSampler
    import torch
    
    # Get the original weights and repeat them for the large dataset
    original_weights = sampler.weights
    repeated_weights = []
    for _ in range(num_copies):
        repeated_weights.extend(original_weights)
    
    large_sampler = WeightedRandomSampler(weights=repeated_weights, num_samples=num_samples, replacement=True)
    
    # Create a temporary DataLoader to test the sampler
    test_loader = DataLoader(dataset=large_dataset, batch_size=1, sampler=large_sampler, num_workers=0, collate_fn=collate_fn)
    
    # Sample from the DataLoader
    sampled_graphs = []
    for i, batch in enumerate(test_loader):
        if i >= num_samples:
            break
        sampled_graphs.append(batch)
    
    # Count hex sizes
    hex_counts = {}
    for batch in sampled_graphs:
        # Handle hex_size as list or tensor
        if hasattr(batch.hex_size, 'tolist'):
            hex_size = str(batch.hex_size.tolist()[0])
        else:
            hex_size = str(batch.hex_size[0])
        hex_counts[hex_size] = hex_counts.get(hex_size, 0) + 1
    
    print(f"Sampled distribution: {hex_counts}")
    
    # Calculate percentages
    total = sum(hex_counts.values())
    percentages = {k: f"{(v/total)*100:.1f}%" for k, v in hex_counts.items()}
    print(f"Percentages: {percentages}")
    
    print("=" * 50)
    return hex_counts

def prepare_data_with_graph_features(datalist, batch_size, path_to_save_dataloader, use_all_features, use_bootstrapping, is_eign=False, split_mode="full", split_variant="uniform", sampler_variant="fixed_proportion"):
    print(f"Starting prepare_data_with_graph_features with {len(datalist)} items")
    
    try:

        print("Splitting into subsets...")

        if split_mode == "full":
            # Full train/val/test split for transductive learning
            if use_bootstrapping:
                train_paths, valid_paths, test_paths = split_into_subsets_with_bootstrapping(dataset=datalist, test_ratio=0.1, bootstrap_seed=4)
            else:
                train_paths, valid_paths, test_paths = split_into_subsets_from_filepaths(dataset=datalist, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05, split_variant=split_variant, split_mode=split_mode)
            print(f"Split complete. Train: {len(train_paths)}, Valid: {len(valid_paths)}, Test: {len(test_paths)}")
        else:  # split_mode == "train_val_only"
            # Train/val split only for inductive learning
            if use_bootstrapping:
                train_paths, valid_paths = split_into_subsets_with_bootstrapping_train_val_only(dataset=datalist, bootstrap_seed=4)
            else:
                train_paths, valid_paths = split_into_subsets_from_filepaths(dataset=datalist, train_ratio=0.85, val_ratio=0.15, split_variant=split_variant, split_mode=split_mode)
            test_paths = None
            print(f"Split complete. Train: {len(train_paths)}, Valid: {len(valid_paths)}, Test: None (using unseen data)")

        if use_all_features:
            node_features = [feat.name for feat in EdgeFeatures]
            if not use_allowed_modes:
                node_features = [feat for feat in node_features if "ALLOWED_MODE" not in feat]
        else:
            # Most important features (from ablation study)
            node_features = ["VOL_BASE_CASE",
                             "CAPACITY_BASE_CASE",
                             "CAPACITY_REDUCTION",
                             "FREESPEED",
                             "LENGTH",
                             #"DESTINATION_HOME",
                             #"DESTINATION_WORK",
                             #"DESTINATION_OTHER",
                             #"DESTINATION_EDUCATION",
                             #"DESTINATION_LEISURE",
                             #"DESTINATION_SHOP",
                             #"DESTINATION_OUTSIDE"
                             ]
        
        print("Normalizing train set and fitting global scalers...")
        train_set_normalized_filepaths, scalers_train = normalize_dataset(dataset_input=train_paths, node_features=node_features)
        print("Train set normalized")      
        
        print("Applying global scalers to validation set...")
        valid_set_normalized_filepaths= apply_global_scaler(data_list=valid_paths, scaler=scalers_train['x_scaler'], node_features=node_features)
        print("Validation set normalized")
        
        if split_mode == "full":
            print("Applying global scalers to test set...")
            test_set_normalized_filepaths = apply_global_scaler(data_list=test_paths, scaler=scalers_train['x_scaler'], node_features=node_features)
            print("Test set normalized")
        
        print(f"Creating train loader with {sampler_variant} sampling...")
        if sampler_variant == "fixed_proportion":
            train_sampler = fixed_proportion_sampler(train_set_normalized_filepaths, batch_size=batch_size, threshold=100)
        elif sampler_variant == "nested_fixed_proportion":
            train_sampler = nested_fixed_proportion_sampler(train_set_normalized_filepaths)
            #TODO: Uncomment this to test the sampler distribution for nested_fixed_proportion
            #test_sampler_distribution(train_set_normalized, train_sampler, num_samples=100) 
        else:
            raise ValueError(f"Invalid sampler variant: {sampler_variant}. Please choose from fixed_proportion or nested_fixed_proportion.")
        
        # DEBUG: Print dataset and sampler information
        print(f"DEBUG: Train set length: {len(train_set_normalized_filepaths)}")
        print(f"DEBUG: Batch size: {batch_size}")
        print(f"DEBUG: Sampler num_samples: {train_sampler.num_samples}")
        print(f"DEBUG: Expected batches: {train_sampler.num_samples // batch_size}")
        
        train_loader = DataLoader(dataset=train_set_normalized_filepaths, batch_size=batch_size, sampler=train_sampler, num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker)
        print("Train loader created")
        print(f"DEBUG: Actual train loader length: {len(train_loader)}")
        
        print("Creating validation loader...")
        val_loader = DataLoader(dataset=valid_set_normalized_filepaths, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker)
        print("Validation loader created")
        
        if split_mode == "full":
            print("Creating test loader...")
            test_loader = DataLoader(dataset=test_set_normalized_filepaths, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker)
            print("Test loader created")
            
            save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
            save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
        
        joblib.dump(scalers_train['x_scaler'], os.path.join(path_to_save_dataloader, 'train_x_scaler.pkl'))
        if not is_eign:
            #joblib.dump(scalers_train['pos_scaler'], os.path.join(path_to_save_dataloader, 'train_pos_scaler.pkl'))
            pass
        # joblib.dump(scalers_train['modestats_scaler'], os.path.join(path_to_save_dataloader, 'train_mode_stats_scaler.pkl'))

        #joblib.dump(scalers_validation['x_scaler'], os.path.join(path_to_save_dataloader, 'validation_x_scaler.pkl'))
        #if not is_eign:
            #joblib.dump(scalers_validation['pos_scaler'], os.path.join(path_to_save_dataloader, 'validation_pos_scaler.pkl'))
        # joblib.dump(scalers_validation['modestats_scaler'], os.path.join(path_to_save_dataloader, 'validation_mode_stats_scaler.pkl'))

        #joblib.dump(scalers_test['x_scaler'], os.path.join(path_to_save_dataloader, 'test_x_scaler.pkl'))
        #if not is_eign:
            #joblib.dump(scalers_test['pos_scaler'], os.path.join(path_to_save_dataloader, 'test_pos_scaler.pkl'))
        # joblib.dump(scalers_test['modestats_scaler'], os.path.join(path_to_save_dataloader, 'test_mode_stats_scaler.pkl'))  
        
        print("Dataloaders and scalers saved")
        
        return train_loader, val_loader, scalers_train
    
    except Exception as e:
        print(f"Error in prepare_data_with_graph_features: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def prepare_test_dataloader_from_unseen(unseen_datalist, scalers_train, batch_size, use_all_features, path_to_save_dataloader, is_eign=False): #TODO: for filepaths
    """
    Prepare test dataloader from unseen data using pre-fitted scalers from training data.
    
    Args:
        unseen_datalist: List of data objects from unseen cities
        scalers_train: Dictionary of scalers fitted on training data
        batch_size: Batch size for DataLoader
        use_all_features: Whether to use all features or subset
        path_to_save_dataloader: Path to save dataloader and parameters
        is_eign: Whether using EIGN model architecture
    
    Returns:
        test_loader: DataLoader for unseen test data
    """
    print(f"Preparing test dataloader from {len(unseen_datalist)} unseen items")
    
    try:
        if use_all_features:
            node_features = [feat.name for feat in EdgeFeatures]
            if not use_allowed_modes:
                node_features = [feat for feat in node_features if "ALLOWED_MODE" not in feat]
        else:
            # Most important features (from ablation study)
            node_features = ["VOL_BASE_CASE",
                             "CAPACITY_BASE_CASE", 
                             "CAPACITY_REDUCTION",
                             "FREESPEED",
                             "LENGTH",
                             "DESTINATION_HOME",
                             "DESTINATION_WORK",
                             "DESTINATION_OTHER",
                             "DESTINATION_EDUCATION",
                             "DESTINATION_LEISURE",
                             "DESTINATION_SHOP",
                             "DESTINATION_OUTSIDE"]
        
        print("Applying training scalers to unseen test data...")
        test_set_normalized = apply_global_scaler(data_list=unseen_datalist, 
                                                  scaler=scalers_train['x_scaler'], 
                                                  node_features=node_features)
        print("Unseen test data normalized")
        
        print("Creating test loader for unseen data...")
        test_loader = DataLoader(dataset=test_set_normalized, 
                                batch_size=batch_size, 
                                shuffle=False,  # No shuffling for test evaluation
                                num_workers=4, 
                                pin_memory=True,
                                collate_fn=collate_fn, 
                                worker_init_fn=seed_worker)
        print("Test loader for unseen data created")
        
        # Save test dataloader and parameters (consistent with transductive case)
        save_dataloader(test_loader, path_to_save_dataloader + 'test_dl_unseen.pt')
        save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_unseen_params.json')
        print("Unseen test dataloader and parameters saved")
        
        return test_loader
    
    except Exception as e:
        print(f"Error in prepare_test_dataloader_from_unseen: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
def normalize_dataset(dataset_input, node_features, is_eign=False): #TODO: ROHAN!! How to integrate pos_scaler for paris here
    """
    Normalize dataset - handles both filepaths and actual data objects.
    """
    # Check what type of input we have
    if not is_eign:
        if isinstance(dataset_input, list) and len(dataset_input) > 0:
            # Check if it's a list of filepaths (strings) or Data objects
            if isinstance(dataset_input[0], str):
                # It's a list of filepaths
                print("Detected filepaths - using filepath-based normalization")
                normalized_filepaths, x_scaler = normalize_x_features_global_inductive_from_filepaths(dataset_input, node_features)
                
                # Return in the expected format for filepath-based processing
                scalers_dict = {
                    "x_scaler": x_scaler,
                }
                return normalized_filepaths, scalers_dict
            else:
                # It's a list of Data objects
                print("Detected Data objects - using object-based normalization")
                data_list = [copy.deepcopy(data) for data in dataset_input]
        elif hasattr(dataset_input, 'indices'):  # It's a Subset object
            print("Detected Subset object - using object-based normalization")
            data_list = [copy.deepcopy(dataset_input.dataset[idx]) for idx in dataset_input.indices]
        else:
            raise ValueError(f"Unsupported dataset_input type: {type(dataset_input)}")
        
    else: #TODO: filepath normalization in case of EIGN
        print("Fitting and normalizing x_signed features...")
        normalized_data_list, x_signed_scaler = normalize_x_signed_features_batched(data_list) #TODO: This should be changed for EIGN
        print("x_signed features normalized")
        scalers_dict = {
            "x_signed_scaler": x_signed_scaler,
        }
        return normalized_data_list, scalers_dict



def normalize_pos_features_batched(data_list, batch_size=1000):
    scaler = StandardScaler()
    
    # Get number of nodes in the graph
    num_nodes = data_list[0].x.shape[0]

    # First pass: Fit the scaler
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting scaler"):
        batch = data_list[i:i+batch_size]
        batch_pos = np.vstack([data.pos.numpy().reshape(-1, 6) for data in batch])
        scaler.partial_fit(batch_pos)
    
    # Second pass: Transform the data
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing pos features"):
        batch = data_list[i:i+batch_size]
        for data in batch:
            pos_reshaped = data.pos.numpy().reshape(-1, 6)
            pos_normalized = scaler.transform(pos_reshaped)
            data.pos = torch.tensor(pos_normalized.reshape(num_nodes, 3, 2), dtype=data.pos.dtype)
    
    return data_list, scaler

def normalize_modestats_features_batched(data_list, batch_size=1000):
    scaler = StandardScaler()
    
    # First pass: Fit the scaler
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting scaler"):
        batch = data_list[i:i+batch_size]
        batch_modestats = np.vstack([data.mode_stats.numpy().reshape(1, -1) for data in batch])
        scaler.partial_fit(batch_modestats)
    
    # Second pass: Transform the data
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing modestats features"):
        batch = data_list[i:i+batch_size]
        for data in batch:
            modestats_reshaped = data.mode_stats.numpy().reshape(1, -1)
            modestats_normalized = scaler.transform(modestats_reshaped)
            data.mode_stats = torch.tensor(modestats_normalized.reshape(6, 2), dtype=torch.float32)
    
    return data_list, scaler

def one_hot_highway(datalist, idx):

    """
    One-hot encodes the 'HIGHWAY' feature and removes the original one.
    Cluster into 6 major classes to reduce dimensionality. (defined with n_types and mapping, originaly 10 classes)
    """
    
    n_types = 6
    mapping = {
        -1: 4, # pt
        0: 0, # primary
        1: 0, # primary
        2: 1, # secondary
        3: 2, # tertiary
        4: 3, # residential
        5: 5, # other
        6: 5,
        7: 5,
        8: 5,
        9: 5
    }

    for data in datalist:
        
        highway = data.x[:, idx].numpy()
        mapped_highway = np.vectorize(mapping.get)(highway)
        one_hot = np.eye(n_types)[mapped_highway]

        data.x = torch.cat((data.x[:, :idx], torch.tensor(one_hot, dtype=data.x.dtype), data.x[:, idx+1:]), dim=1)

def normalize_x_features_with_scaler(data_list, node_features, x_scaler, batch_size=100):
    """
    Normalize the continuous node features with a given scaler.
    Categorical features (Allowed Modes) are left as booleans (0 or 1).
    'HIGHWAY' feature is one-hot encoded.

    Finally, features are filtered to only include the ones specified in node_features. 
    """

    # Continuous features to normalize
    continuous_feat = [EdgeFeatures.VOL_BASE_CASE,
                       EdgeFeatures.CAPACITY_BASE_CASE,
                       EdgeFeatures.CAPACITY_REDUCTION,
                       EdgeFeatures.FREESPEED,
                       EdgeFeatures.LENGTH]
    
    # Get number of nodes in the graph
    num_nodes = data_list[0].x.shape[0]
    
    # Second pass: Transform the data
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing x features"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        batch_x_normalized = x_scaler.transform(batch_x)
        for j, data in enumerate(batch):
            data.x[:,continuous_feat] = torch.tensor(batch_x_normalized[j*num_nodes:(j+1)*num_nodes], dtype=data.x.dtype)

    # Filter features
    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
    for data in data_list:
        data.x = data.x[:, node_feature_filter]

    # One-hot encode highway
    if "HIGHWAY" in node_features:
        one_hot_highway(data_list, idx=node_features.index("HIGHWAY"))
    
    return data_list

def normalize_x_signed_features_batched(data_list, batch_size=1000):
    """
    Normalize the x_signed features (0 mean and unit variance).
    x_signed typically has shape (num_nodes, 1) for EIGN models.
    """
    scaler = StandardScaler()

    # Get number of nodes in the graph
    num_nodes = (
        data_list[0].x_signed.shape[0]
        if hasattr(data_list[0], "x_signed") and data_list[0].x_signed is not None
        else 0
    )

    # Skip if no x_signed features
    if num_nodes == 0:
        return data_list, None

    # First pass: Fit the scaler
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting x_signed scaler"):
        batch = data_list[i : i + batch_size]
        batch_x_signed = np.vstack(
            [
                data.x_signed.numpy().reshape(-1, 1)
                for data in batch
                if hasattr(data, "x_signed") and data.x_signed is not None
            ]
        )
        if batch_x_signed.size > 0:
            scaler.partial_fit(batch_x_signed)

    # Second pass: Transform the data
    for i in tqdm(
        range(0, len(data_list), batch_size), desc="Normalizing x_signed features"
    ):
        batch = data_list[i : i + batch_size]
        for data in batch:
            if hasattr(data, "x_signed") and data.x_signed is not None:
                x_signed_reshaped = data.x_signed.numpy().reshape(-1, 1)
                x_signed_normalized = scaler.transform(x_signed_reshaped)
                data.x_signed = torch.tensor(
                    x_signed_normalized.reshape(data.x_signed.shape),
                    dtype=data.x_signed.dtype,
                )

    return data_list, scaler

def normalize_x_signed_features_with_scaler(
    data_list, x_signed_scaler, batch_size=1000
):
    """
    Normalize the x_signed features with a given scaler.
    x_signed typically has shape (num_nodes, 1) for EIGN models.
    """
    # Skip if no scaler provided or no x_signed features
    if x_signed_scaler is None:
        return data_list

    # Check if any data has x_signed features
    has_x_signed = any(
        hasattr(data, "x_signed") and data.x_signed is not None for data in data_list
    )

    if not has_x_signed:
        return data_list

    # Transform the data using the provided scaler
    for i in tqdm(
        range(0, len(data_list), batch_size), desc="Normalizing x_signed features"
    ):
        batch = data_list[i : i + batch_size]
        for data in batch:
            if hasattr(data, "x_signed") and data.x_signed is not None:
                x_signed_reshaped = data.x_signed.numpy().reshape(-1, 1)
                x_signed_normalized = x_signed_scaler.transform(x_signed_reshaped)
                data.x_signed = torch.tensor(
                    x_signed_normalized.reshape(data.x_signed.shape),
                    dtype=data.x_signed.dtype,
                )
    return data_list

def setup_wandb(args):
    wandb.login()
    wandb.init(project=args['project_name'], name=args['unique_model_description'],
               config={k: v for k, v in args.items() if k not in ['project_name', 'unique_model_description', 'model_kwargs']})
    return wandb.config

def setup_wandb_metrics(predict_mode_stats=False):

    wandb.define_metric("epoch") # Custom X-axis
    wandb.define_metric("batch_step") # Custom X-axis
    
    wandb.define_metric("batch_train_loss", step_metric="batch_step")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    wandb.define_metric("r^2", step_metric="epoch")
    wandb.define_metric("spearman", step_metric="epoch")
    wandb.define_metric("pearson", step_metric="epoch")

    if predict_mode_stats:
        wandb.define_metric("batch_train_loss-node_predictions", step_metric="batch_step")
        wandb.define_metric("batch_train_loss-mode_stats", step_metric="batch_step")
        wandb.define_metric("train_loss-node_predictions", step_metric="epoch")
        wandb.define_metric("train_loss-mode_stats", step_metric="epoch")
        wandb.define_metric("val_loss-node_predictions", step_metric="epoch")
        wandb.define_metric("val_loss-mode_stats", step_metric="epoch")

def create_gnn_model(gnn_arch: str, config: object, model_kwargs: dict, device: torch.device):
    """
    Factory function to create the specified model architecture.
    
    Args:
    - gnn_arch (str): The architecture of the GNN model to create.
    - config (object): WandB config with run arguments.
    - device (torch.device): The device to which the model should be moved (CPU or GPU).
    
    Returns:
    - Initialized model on the specified device
    """

    common_kwargs = {
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "use_dropout": config.use_dropout,
        "dropout": config.dropout,
        "predict_mode_stats": config.predict_mode_stats,
        "dtype": torch.float32,
        "log_to_wandb": True} # During training, yes

    if gnn_arch == "point_net_transf_gat":
        return PointNetTransfGAT(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "graphSAGE":
        return GraphSAGE(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gcn":
        return GCN(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gcn2":
        return GCN2(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gat":
        return GAT(**common_kwargs, **model_kwargs).to(device)

    elif gnn_arch == "gatv2":
        return GATv2(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "trans_conv":
        return TransConv(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "pnc":
        return PNC(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "fc_nn":
        return FC_NN(**common_kwargs, **model_kwargs).to(device)

    elif gnn_arch == "eign":
        return EIGNLaplacianConv(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "xgboost":
        return XGBoostModel(**common_kwargs, **model_kwargs)
    else:
        raise ValueError(f"Unknown architecture: {gnn_arch}")

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def normalize_x_features_global_inductive_from_filepaths(filepaths, node_features, batch_size=100):
    """
    Global normalization across ALL cities for inductive learning with batching.
    Works with filepaths instead of loaded graph objects for memory efficiency.
    """
    scaler = StandardScaler()
    
    # Continuous features to normalize - only the ones that will be used
    continuous_feat = [EdgeFeatures[feature].value for feature in node_features]
    
    # First pass: Fit the scaler on ALL cities using batching
    print("Fitting global scaler across all cities from filepaths...")
    
    processed_graphs = 0
    for i in tqdm(range(0, len(filepaths), batch_size), desc="Fitting scaler"):
        batch_filepaths = filepaths[i:i+batch_size]
        
        # Load graphs from this batch of filepaths
        batch_data = []
        for filepath in batch_filepaths:
            try:
                # Load the batch file
                loaded_data = torch.load(filepath, map_location='cpu')
                if isinstance(loaded_data, list):
                    batch_data.extend(loaded_data)
                else:
                    batch_data.append(loaded_data)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
                continue
        
        if batch_data:
            # Extract features for scaler fitting
            batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch_data])
            scaler.partial_fit(batch_x)
            processed_graphs += len(batch_data)
        
        # Clear memory
        del batch_data
    
    print(f"Global scaler fitted on {processed_graphs} graphs from all cities")
    print(f"Feature means: {scaler.mean_}")
    print(f"Feature scales: {scaler.scale_}")
    
    # Second pass: Transform and save the data using batching
    print("Transforming and saving normalized data...")
    normalized_filepaths = []
    
    for i in tqdm(range(0, len(filepaths), batch_size), desc="Normalizing x features"):
        batch_filepaths = filepaths[i:i+batch_size]
        
        for filepath in batch_filepaths:
            try:
                # Load the batch file
                loaded_data = torch.load(filepath, map_location='cpu')
                if isinstance(loaded_data, list):
                    data_list = loaded_data
                else:
                    data_list = [loaded_data]
                
                # Normalize this batch of graphs
                normalized_graphs = []
                for data in data_list:
                    # Transform features
                    data_x = data.x[:,continuous_feat].numpy()
                    data_x_normalized = scaler.transform(data_x)
                    data.x[:,continuous_feat] = torch.tensor(data_x_normalized, dtype=data.x.dtype)
                    
                    # Filter features to keep only selected ones
                    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
                    data.x = data.x[:, node_feature_filter]
                    
                    # One-hot encode highway if needed
                    if "HIGHWAY" in node_features:
                        # Apply one-hot encoding to this single graph
                        highway_idx = node_features.index("HIGHWAY")
                        one_hot_highway([data], idx=highway_idx)
                    
                    normalized_graphs.append(data)
                
                # Save normalized data back to the same file (or new location)
                torch.save(normalized_graphs if len(normalized_graphs) > 1 else normalized_graphs[0], filepath)
                normalized_filepaths.append(filepath)
                
                # Clear memory
                del data_list, normalized_graphs
                
            except Exception as e:
                print(f"Warning: Could not process {filepath}: {e}")
                continue
    
    if normalized_filepaths:
        print(f'Shape of features (from first file): checking...')
        # Check shape from first file
        try:
            sample_data = torch.load(normalized_filepaths[0], map_location='cpu')
            if isinstance(sample_data, list):
                sample_graph = sample_data[0]
            else:
                sample_graph = sample_data
            print(f'Shape of features: {sample_graph.x.shape}')
        except:
            print('Could not determine feature shape')
    
    return normalized_filepaths, scaler

def normalize_x_features_global_inductive(data_list, node_features, batch_size=100):
    """
    Global normalization across ALL cities for inductive learning with batching.
    Model learns to handle the natural variation between cities.
    """
    scaler = StandardScaler()
    
    # Continuous features to normalize - only the ones that will be used
    continuous_feat = [EdgeFeatures[feature].value for feature in node_features]
    
    # First pass: Fit the scaler on ALL cities using batching
    print("Fitting global scaler across all cities...")
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting scaler"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        scaler.partial_fit(batch_x)
    
    print(f"Global scaler fitted on features from all cities")
    print(f"Feature means: {scaler.mean_}")
    print(f"Feature scales: {scaler.scale_}")
    
    # Second pass: Transform the data using batching
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing x features"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        batch_x_normalized = scaler.transform(batch_x)
        
        # Put normalized features back with correct indexing for different graph sizes
        start_idx = 0
        for data in batch:
            num_nodes = data.x.shape[0]  # Actual size of THIS graph
            end_idx = start_idx + num_nodes
            data.x[:,continuous_feat] = torch.tensor(
                batch_x_normalized[start_idx:end_idx], dtype=data.x.dtype)
            start_idx = end_idx  # Move to next graph's position
    
    # Filter features to keep only selected ones
    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
    for data in data_list:
        data.x = data.x[:, node_feature_filter]
    
    # One-hot encode highway if needed
    if "HIGHWAY" in node_features:
        one_hot_highway(data_list, idx=node_features.index("HIGHWAY"))

    print(f'Shape of features: {data_list[0].x.shape}')
    
    return data_list, scaler

def normalize_x_features_with_scaler_global_inductive(data_list, node_features, x_scaler, batch_size=100):
    """
    Normalize the continuous node features with a given scaler.
    Categorical features (Allowed Modes) are left as booleans (0 or 1).
    'HIGHWAY' feature is one-hot encoded.

    Finally, features are filtered to only include the ones specified in node_features.
    
    FIXED: Properly handles graphs with different numbers of nodes.
    """

    # Continuous features to normalize - only the ones that will be used
    continuous_feat = [EdgeFeatures[feature].value for feature in node_features]
    
    # Transform the data using batching with proper indexing
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing x features"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        batch_x_normalized = x_scaler.transform(batch_x)
        
        # Put normalized features back with correct indexing for different graph sizes
        start_idx = 0
        for data in batch:
            num_nodes = data.x.shape[0]  # Actual size of THIS graph
            end_idx = start_idx + num_nodes
            data.x[:,continuous_feat] = torch.tensor(
                batch_x_normalized[start_idx:end_idx], dtype=data.x.dtype)
            start_idx = end_idx  # Move to next graph's position

    # Filter features to keep only selected ones
    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
    for data in data_list:
        data.x = data.x[:, node_feature_filter]

    # One-hot encode highway if needed
    if "HIGHWAY" in node_features:
        one_hot_highway(data_list, idx=node_features.index("HIGHWAY"))
    
    print(f'Shape of features: {data_list[0].x.shape}')
    
    return data_list


def apply_global_scaler(data_input, scaler, node_features, batch_size=100):
    """
    Apply the global scaler to the data.
    Handles both Data objects and filepaths.
    - If data_input contains strings (filepaths), processes files in batches
    - If data_input contains Data objects, uses the original logic
    """
    # Check if input is filepaths (strings) or Data objects
    if len(data_input) > 0 and isinstance(data_input[0], str):
        # Handle filepaths case
        return apply_global_scaler_from_filepaths(data_input, scaler, node_features, batch_size)
    else:
        # Handle Data objects case
        # Convert Subset to list if needed
        if hasattr(data_input, 'indices'):  # It's a Subset object
            data_list_actual = [data_input[i] for i in range(len(data_input))]
        else:
            data_list_actual = data_input
        
        return normalize_x_features_with_scaler_global_inductive(data_list_actual, node_features, scaler)


def apply_global_scaler_from_filepaths(filepaths, scaler, node_features, batch_size=100):
    """
    Apply a fitted global scaler to data stored in filepaths.
    Works with filepaths instead of loaded graph objects for memory efficiency.
    
    Args:
        filepaths: List of file paths containing graph data
        scaler: Pre-fitted StandardScaler object
        node_features: List of feature names to use
        batch_size: Number of files to process at once
    
    Returns:
        List of processed filepaths
    """
    # Continuous features to normalize - only the ones that will be used
    continuous_feat = [EdgeFeatures[feature].value for feature in node_features]
    
    print("Applying fitted global scaler to data from filepaths...")
    processed_filepaths = []
    
    for i in tqdm(range(0, len(filepaths), batch_size), desc="Applying scaler to files"):
        batch_filepaths = filepaths[i:i+batch_size]
        
        for filepath in batch_filepaths:
            try:
                # Load the batch file
                loaded_data = torch.load(filepath, map_location='cpu')
                if isinstance(loaded_data, list):
                    data_list = loaded_data
                else:
                    data_list = [loaded_data]
                
                # Apply scaler to this batch of graphs
                normalized_graphs = []
                for data in data_list:
                    # Transform features using the fitted scaler
                    data_x = data.x[:,continuous_feat].numpy()
                    data_x_normalized = scaler.transform(data_x)
                    data.x[:,continuous_feat] = torch.tensor(data_x_normalized, dtype=data.x.dtype)
                    
                    # Filter features to keep only selected ones
                    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
                    data.x = data.x[:, node_feature_filter]
                    
                    # One-hot encode highway if needed
                    if "HIGHWAY" in node_features:
                        # Apply one-hot encoding to this single graph
                        highway_idx = node_features.index("HIGHWAY")
                        one_hot_highway([data], idx=highway_idx)
                    
                    normalized_graphs.append(data)
                
                # Save normalized data back to the same file
                torch.save(normalized_graphs if len(normalized_graphs) > 1 else normalized_graphs[0], filepath)
                processed_filepaths.append(filepath)
                
                # Clear memory
                del data_list, normalized_graphs
                
            except Exception as e:
                print(f"Warning: Could not process {filepath}: {e}")
                continue
    
    if processed_filepaths:
        print(f'Applied scaler to {len(processed_filepaths)} files')
        # Check shape from first file
        try:
            sample_data = torch.load(processed_filepaths[0], map_location='cpu')
            if isinstance(sample_data, list):
                sample_graph = sample_data[0]
            else:
                sample_graph = sample_data
            print(f'Shape of features after scaling: {sample_graph.x.shape}')
        except:
            print('Could not determine feature shape')
    
    return processed_filepaths


def verify_batch_distribution(data_container, num_batches_to_check=None):
    """
    Verify that data has balanced hex_size distribution.
    Handles DataLoader, Subset, or list of Data objects.
    If num_batches_to_check is None, checks ALL batches.
    """
    print("\n=== Verifying Data Distribution ===")
    
    # Handle different input types
    if hasattr(data_container, 'dataset'):  # DataLoader
        # Check batches from DataLoader
        for i, batch in enumerate(data_container):
            if num_batches_to_check is not None and i >= num_batches_to_check:
                break
                
            # Handle both tensor and list formats
            if hasattr(batch.hex_size, 'tolist'):
                hex_sizes = batch.hex_size.tolist()
            else:
                hex_sizes = batch.hex_size
                
            hex_distribution = Counter(hex_sizes)
            print(f"Batch {i+1}: {dict(hex_distribution)}")
            
            # Calculate percentages
            total = len(hex_sizes)
            percentages = {k: f"{(v/total)*100:.1f}%" for k, v in hex_distribution.items()}
            print(f"  Percentages: {percentages}")
            print(f"  Total items in batch: {total}")
    
    elif hasattr(data_container, 'indices'):  # Subset
        # Check all items from Subset
        if num_batches_to_check is None:
            sample_size = len(data_container)
        else:
            sample_size = min(num_batches_to_check * 10, len(data_container))
        
        sample_indices = random.sample(range(len(data_container)), sample_size)
        
        hex_sizes = []
        for idx in sample_indices:
            hex_sizes.append(data_container[idx].hex_size)
        
        hex_distribution = Counter(hex_sizes)
        print(f"Subset sample ({sample_size} items): {dict(hex_distribution)}")
        
        # Calculate percentages
        total = len(hex_sizes)
        percentages = {k: f"{(v/total)*100:.1f}%" for k, v in hex_distribution.items()}
        print(f"  Percentages: {percentages}")
    
    else:  # List of Data objects
        # Check all items from list
        if num_batches_to_check is None:
            sample_size = len(data_container)
        else:
            sample_size = min(num_batches_to_check * 10, len(data_container))
        
        sample_indices = random.sample(range(len(data_container)), sample_size)
        
        hex_sizes = []
        for idx in sample_indices:
            hex_sizes.append(data_container[idx].hex_size)
        
        hex_distribution = Counter(hex_sizes)
        print(f"List sample ({sample_size} items): {dict(hex_distribution)}")
        
        # Calculate percentages
        total = len(hex_sizes)
        percentages = {k: f"{(v/total)*100:.1f}%" for k, v in hex_distribution.items()}
        print(f"  Percentages: {percentages}")
    
    print("=" * 40)