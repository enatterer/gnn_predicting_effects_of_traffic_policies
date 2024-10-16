import math
import numpy as np
import wandb
import random
import torch
import torch_geometric
from torch_geometric.data import Data
import sys
import os
from tqdm import tqdm
import signal
import joblib
import argparse
import json
import os
import subprocess
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler

import help_functions as hf

# Add the 'scripts' directory to the Python path
scripts_path = os.path.abspath(os.path.join('..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
    
import gnn_io as gio
import gnn_architectures_improved as garch



# Define parameters
def get_parameters(args):
        project_name = "runs_with_graph_features"
        num_epochs = 3000
        in_channels = 15
        out_channels = 1
        graph_mlp_layer_structure = [int(x) for x in args.graph_mlp_layer_structure.split(',')]
        lr = float(args.lr)
        batch_size = int(args.batch_size)
        point_net_conv_layer_structure_local_mlp = [int(x) for x in args.point_net_conv_layer_structure_local_mlp.split(',')]
        point_net_conv_layer_structure_global_mlp = [int(x) for x in args.point_net_conv_layer_structure_global_mlp.split(',')]
        gat_conv_layer_structure = [int(x) for x in args.gat_conv_layer_structure.split(',')]
        graph_mlp_layer_structure = [int(x) for x in args.graph_mlp_layer_structure.split(',')]
        gradient_accumulation_steps = args.gradient_accumulation_steps
        early_stopping_patience = args.early_stopping_patience
        dropout =args.dropout 
        use_dropout = args.use_dropout
        use_graph_features = args.use_graph_features
        device_nr = args.device_nr
        
        unique_model_description = (
            f"pnc_local_{gio.int_list_to_string(lst = point_net_conv_layer_structure_local_mlp, delimiter='_')}_"
            f"pnc_global_{gio.int_list_to_string(lst = point_net_conv_layer_structure_global_mlp, delimiter='_')}_"
            f"hidden_layer_str_{gio.int_list_to_string(lst = gat_conv_layer_structure, delimiter='_')}_"
            f"dropout_{dropout}_"
            f"use_dropout_{use_dropout}"
            f"use_graph_features_{use_graph_features}"
        )
        return {
            "project_name": project_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "use_graph_features": use_graph_features,
            "point_net_conv_layer_structure_local_mlp": point_net_conv_layer_structure_local_mlp,
            "point_net_conv_layer_structure_global_mlp": point_net_conv_layer_structure_global_mlp,
            "gat_conv_layer_structure": gat_conv_layer_structure,
            "graph_mlp_layer_structure": graph_mlp_layer_structure,
            "lr": lr,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "early_stopping_patience": early_stopping_patience,
            "unique_model_description": unique_model_description,
            "dropout": dropout,
            "use_dropout": use_dropout,
            "device_nr": device_nr
        } 
        
def get_paths(base_dir: str, unique_model_description: str, model_save_path: str = 'trained_model/model.pth'):
    data_path = os.path.join(base_dir, unique_model_description)
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

def get_paths(base_dir: str, unique_model_description: str, model_save_path: str = 'trained_model/model.pth'):
    data_path = os.path.join(base_dir, unique_model_description)
    os.makedirs(data_path, exist_ok=True)
    model_save_to = os.path.join(data_path, model_save_path)
    path_to_save_dataloader = os.path.join(data_path, 'data_created_during_training/')
    os.makedirs(os.path.dirname(model_save_to), exist_ok=True)
    os.makedirs(path_to_save_dataloader, exist_ok=True)
    return model_save_to, path_to_save_dataloader

def normalize_dataset(dataset_input, directory_path):
    data_list = [dataset_input.dataset[idx] for idx in dataset_input.indices]
    print("LEN DATALIST")
    print(len(data_list))
    print("Fitting and normalizing x features...")
    normalized_data_list, x_scaler = normalize_x_features_batched(data_list)
    print("x features normalized")
    print(len(normalized_data_list))
    

    print("Fitting and normalizing pos features...")
    normalized_data_list, pos_scaler = normalize_pos_features_batched(normalized_data_list)
    print("Pos features normalized")
    
    print("Fitting and normalizing modestats features...")
    normalized_data_list, modestats_scaler = normalize_modestats_features_batched(normalized_data_list)
    print("Modestats features normalized")
    
    print("FINAL LEN")
    print(len(normalized_data_list))
    return normalized_data_list, (x_scaler, pos_scaler, modestats_scaler)

def normalize_x_features_batched(data_list, batch_size=100):
    scaler = StandardScaler()
    
    # First pass: Fit the scaler
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting scaler"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x.numpy() for data in batch])
        scaler.partial_fit(batch_x)
    
    # Second pass: Transform the data
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing x features"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x.numpy() for data in batch])
        batch_x_normalized = scaler.transform(batch_x)
        for j, data in enumerate(batch):
            data.x = torch.tensor(batch_x_normalized[j*31140:(j+1)*31140], dtype=torch.float32)
    
    return data_list, scaler

def normalize_pos_features_batched(data_list, batch_size=1000):
    scaler = StandardScaler()
    
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
            data.pos = torch.tensor(pos_normalized.reshape(31140, 3, 2), dtype=torch.float32)
    
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


def prepare_data_with_graph_features(datalist, batch_size, path_to_save_dataloader):
    print(f"Starting prepare_data_with_graph_features with {len(datalist)} items")
    
    try:
        print("Splitting into subsets...")
        train_set, valid_set, test_set = gio.split_into_subsets(dataset=datalist, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)
        print(f"Split complete. Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
        
        print("Normalizing train set...")
        train_set_normalized, scalers_train = normalize_dataset(dataset_input=train_set, directory_path=path_to_save_dataloader + "train_")
        print("Train set normalized")
        
        print("Normalizing validation set...")
        valid_set_normalized, scalers_validation = normalize_dataset(dataset_input=valid_set, directory_path=path_to_save_dataloader + "valid_")
        print("Validation set normalized")
        print(len(valid_set_normalized))
        
        print("Creating train loader...")
        train_loader = DataLoader(dataset=train_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
        print("Train loader created")
        
        print("Creating validation loader...")
        val_loader = DataLoader(dataset=valid_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
        print("Validation loader created")
        
        return train_loader, val_loader, scalers_train, scalers_validation
    except Exception as e:
        print(f"Error in prepare_data_with_graph_features: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    try:
        dataset_path = '../../data/train_data/sim_output_1pm_capacity_reduction_10k_15_10_2024/'
        datalist = []
        batch_num = 1
        while True:
            print(f"Processing batch number: {batch_num}")
            # total_memory, available_memory, used_memory = get_memory_info()
            # print(f"Total Memory: {total_memory:.2f} GB")
            # print(f"Available Memory: {available_memory:.2f} GB")
            # print(f"Used Memory: {used_memory:.2f} GB")
            batch_file = os.path.join(dataset_path, f'datalist_batch_{batch_num}.pt')
            if not os.path.exists(batch_file):
                break
            batch_data = torch.load(batch_file, map_location='cpu')
            if isinstance(batch_data, list):
                datalist.extend(batch_data)
            batch_num += 1
        print(f"Loaded {len(datalist)} items into datalist")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    
    
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--num_epochs", type=int, default=3000, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--point_net_conv_layer_structure_local_mlp", type=str, default="64, 128", help="Size of hidden layers.")
    parser.add_argument("--point_net_conv_layer_structure_global_mlp", type=str, default="256, 64", help="Size of hidden layers.")
    parser.add_argument("--gat_conv_layer_structure", type=str, default="128, 256, 256, 128", help="Structure of GAT Conv hidden layer sizes (comma-separated).")
    parser.add_argument("--graph_mlp_layer_structure", type=str, default="128, 256, 128", help="Structure of GAT Conv hidden layer sizes (comma-separated).")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate for the model.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--in_channels", type=int, default=15, help="The early stopping patience.")
    parser.add_argument("--out_channels", type=int, default=1, help="The early stopping patience.")
    parser.add_argument("--early_stopping_patience", type=int, default=100, help="The early stopping patience.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--device_nr", type=int, default=0, help="The device that this model should run for. The Retina Roaster has two GPUs, so the values 0 and 1 are allowed here.")
    parser.add_argument("--use_dropout", type=hf.str_to_bool, default=False, help="Whether to use or not use dropout.")
    parser.add_argument("--use_graph_features", type=hf.str_to_bool, default=False, help="Whether to use or not use graph features.")

    args = parser.parse_args()
    
    hf.set_random_seeds()
    
    try:
        gpus = hf.get_available_gpus()
        best_gpu = hf.select_best_gpu(gpus)
        hf.set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = get_parameters(args=args)
        
        # Create base directory for the run
        base_dir = '../../data/' + params['project_name'] + '/'
        unique_run_dir = os.path.join(base_dir, params['unique_model_description'])
        os.makedirs(unique_run_dir, exist_ok=True)
        
        model_save_path, path_to_save_dataloader = get_paths(base_dir=base_dir, unique_model_description= params['unique_model_description'], model_save_path= 'trained_model/model.pth')
        train_dl, valid_dl, scalers_train, scalers_validation = prepare_data_with_graph_features(datalist=datalist, batch_size= params['batch_size'], path_to_save_dataloader= path_to_save_dataloader)
        
        config = hf.setup_wandb(params['project_name'], {
            "epochs": params['num_epochs'],
            "batch_size": params['batch_size'],
            "lr": params['lr'],
            "gradient_accumulation_steps": params['gradient_accumulation_steps'],
            "early_stopping_patience": params['early_stopping_patience'],
            "point_net_conv_local_mlp": params['point_net_conv_layer_structure_local_mlp'],
            "point_net_conv_global_mlp": params['point_net_conv_layer_structure_global_mlp'],
            "gat_conv_layer_structure": params['gat_conv_layer_structure'],
            "graph_mlp_layer_structure": params['graph_mlp_layer_structure'],
            "in_channels": params['in_channels'],
            "out_channels": params['out_channels'],
            "dropout": params['dropout'],
            "use_dropout": params['use_dropout'],
            "use_graph_features": params['use_graph_features'],
            "device_nr": params['device_nr'],
            "unique_model_description": params['unique_model_description'],
        })

        gnn_instance = garch.MyGnn(in_channels=config.in_channels, out_channels=config.out_channels, point_net_conv_layer_structure_local_mlp=config.point_net_conv_local_mlp,
                                   point_net_conv_layer_structure_global_mlp=config.point_net_conv_global_mlp,
                                   gat_conv_layer_structure=config.gat_conv_layer_structure,
                                   dropout=config.dropout, use_dropout=config.use_dropout, use_graph_features=config.use_graph_features)
        model = gnn_instance.to(device)
        loss_fct = torch.nn.MSELoss()
        
        baseline_loss_mean_target = gio.compute_baseline_of_mean_target(dataset=train_dl, loss_fct=loss_fct)
        baseline_loss = gio.compute_baseline_of_no_policies(dataset=train_dl, loss_fct=loss_fct)
        print("baseline loss mean " + str(baseline_loss_mean_target))
        print("baseline loss no  " +str(baseline_loss) )

        early_stopping = gio.EarlyStopping(patience=params['early_stopping_patience'], verbose=True)
        best_val_loss, best_epoch = garch.train(model=model, 
                    config=config, 
                    loss_fct=loss_fct,
                    optimizer=torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4),
                    train_dl=train_dl, 
                    valid_dl=valid_dl,
                    device=device, 
                    early_stopping=early_stopping,
                    accumulation_steps=config.gradient_accumulation_steps,
                    model_save_path=model_save_path,
                    use_gradient_clipping=True,
                    lr_scheduler_warmup_steps=20000,
                    lr_scheduler_cosine_decay_rate=0.2)
        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   
        
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
     
if __name__ == '__main__':
    main()