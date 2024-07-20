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

# Add the 'scripts' directory to the Python path
scripts_path = os.path.abspath(os.path.join('..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
    
import gnn_io as gio
import gnn_architectures as garch

# Define the paths here
def get_paths():
    data_dict_list = torch.load('../../data/train_data/dataset_1pm_0-3500_new.pt')
    model_save_path = '../../data/trained_models/model_last_layer_gcn.pth'
    path_to_save_dataloader = "../../data/data_created_during_training_needed_for_testing/"
    checkpoint_dir = "../../data/checkpoints_batchsize_8/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return data_dict_list, model_save_path, path_to_save_dataloader, checkpoint_dir

# Define parameters
def get_parameters():
        project_name = "run_with_python_script"
        indices_of_datasets_to_use = [0, 1, 3, 4]
        num_epochs = 1000
        batch_size = 8
        output_layer_parameter = 'gat'
        hidden_layer_size = 64
        gat_layer_parameter = 5
        gcn_layer_parameter = 0
        lr = 0.001
        in_channels = len(indices_of_datasets_to_use) + 2
        out_channels = 1
        early_stopping_patience = 10

        unique_model_description = (
            f"datasets_{indices_of_datasets_to_use}_"
            f"batch_{batch_size}_"
            f"output_{output_layer_parameter}_"
            f"hidden_{hidden_layer_size}_"
            f"gat_layers_{gat_layer_parameter}_"
            f"gcn_layers_{gcn_layer_parameter}_"
            f"lr_{lr}_"
            f"in_channels_{in_channels}_"
            f"out_channels_{out_channels}_"
            f"early_stopping_{early_stopping_patience}"
        )
        return {
            "project_name": project_name,
            "indices_of_datasets_to_use": indices_of_datasets_to_use,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "output_layer_parameter": output_layer_parameter,
            "hidden_layer_size": hidden_layer_size,
            "gat_layer_parameter": gat_layer_parameter,
            "gcn_layer_parameter": gcn_layer_parameter,
            "lr": lr,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "early_stopping_patience": early_stopping_patience,
            "unique_model_description": unique_model_description
        }
        
def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_data(data_dict_list, indices_of_datasets_to_use, path_to_save_dataloader):
    datalist = [Data(x=d['x'], edge_index=d['edge_index'], pos=d['pos'], y=d['y']) for d in data_dict_list]
    dataset_only_relevant_dimensions = gio.cut_dimensions(dataset=datalist, indices_of_dimensions_to_keep=indices_of_datasets_to_use)
    dataset_normalized = gio.normalize_dataset(dataset_only_relevant_dimensions, y_scalar=None, x_scalar_list=None, pos_scalar=None, directory_path=path_to_save_dataloader)
    return dataset_normalized

def create_dataloaders_and_save_test_set(dataset_normalized, batch_size, unique_model_description, path_to_save_dataloader):
    train_dl, valid_dl, test_dl = gio.create_dataloaders(batch_size=batch_size, dataset=dataset_normalized, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    gio.save_dataloader(test_dl, path_to_save_dataloader + 'test_dl_' + unique_model_description + '.pt')
    gio.save_dataloader_params(test_dl, path_to_save_dataloader + 'test_loader_params_' + unique_model_description + '.json')
    return train_dl, valid_dl

def setup_wandb(project_name, config):
    wandb.login()
    wandb.init(project=project_name, config=config)
    return wandb.config
        
def train_model(config, train_dl, valid_dl, device, early_stopping, checkpoint_dir, model_save_path):
    gnn_instance = garch.MyGnnHardCoded(in_channels=config.in_channels, out_channels=config.out_channels, hidden_size=config.hidden_layer_size, output_layer=config.output_layer)
    model = gnn_instance.to(device)
    loss_fct = torch.nn.MSELoss()
    garch.train(model=model, 
                config=config, 
                loss_fct=loss_fct,
                optimizer=torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0),
                train_dl=train_dl, 
                valid_dl=valid_dl,
                device=device, 
                early_stopping=early_stopping,
                accumulation_steps=3,
                save_checkpoints=False,
                iteration_save_checkpoint=None,
                use_existing_checkpoint=False, 
                path_existing_checkpoints=checkpoint_dir,
                compute_r_squared=False)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')   

def main():
    set_random_seeds()
    device = get_device()
    data_dict_list, model_save_path, path_to_save_dataloader, checkpoint_dir = get_paths()
    params = get_parameters()
    print("params")
    print(params)
    dataset_normalized = prepare_data(data_dict_list, params['indices_of_datasets_to_use'], path_to_save_dataloader)
    train_dl, valid_dl = create_dataloaders_and_save_test_set(dataset_normalized, params['batch_size'], params['unique_model_description'], path_to_save_dataloader)
    
    config = setup_wandb(params['project_name'], {
        "epochs": params['num_epochs'],
        "batch_size": params['batch_size'],
        "lr": params['lr'],
        "early_stopping_patience": params['early_stopping_patience'],
        "hidden_layer_size": params['hidden_layer_size'],
        "gat_layers": params['gat_layer_parameter'],
        "gcn_layers": params['gcn_layer_parameter'],
        "output_layer": params['output_layer_parameter'],
        "indices_to_use": params['indices_of_datasets_to_use'],
        "dataset_length": len(dataset_normalized), 
        "in_channels": params['in_channels'],
        "out_channels": params['out_channels'],
    })

    early_stopping = gio.EarlyStopping(patience=params['early_stopping_patience'], verbose=True)
    print(config)
    train_model(config, train_dl, valid_dl, device, early_stopping, checkpoint_dir, model_save_path)
     
if __name__ == '__main__':
    main()