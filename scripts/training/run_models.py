import math
import numpy as np
import wandb

import torch
import torch_geometric
from torch_geometric.data import Data

import sys
import os
from tqdm import tqdm

# Add the 'scripts' directory to the Python path
scripts_path = os.path.abspath(os.path.join('..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
    
import joblib

# Now you can import the gnn_io module
import gnn_io as gio

import gnn_architectures as garch
import itertools

def main():
    num_epochs = 1000
    project_name = "test_different_parameters"
    path_to_save_dataloader = "../../data/data_created_during_training_needed_for_testing/"
    indices_of_datasets_to_use = [0, 1, 2, 3]

    loss_fct = torch.nn.MSELoss()
    batch_size = 32
    output_layer_parameter = 'gat'
    hidden_size_parameter = 32
    gat_layer_parameter = 2
    gcn_layer_parameter = 0
    lr = 0.001
    in_channels = len(indices_of_datasets_to_use) + 2 # dimensions of the x vector + 2 (pos)
    out_channels = 1 # we are predicting one value
    early_stopping_patience = 10

    unique_model_description = f"mse_loss_hidden_{hidden_size_parameter}_gat_{gat_layer_parameter}_gcn_{gcn_layer_parameter}_lr_{lr}"

    data_dict_list = torch.load('../../data/train_data/dataset_1pm_0-3100.pt')
    
    # Reconstruct the Data objects
    datalist = [Data(x=d['x'], edge_index=d['edge_index'], pos=d['pos'], y=d['y']) for d in data_dict_list]
    # dataset_only_relevant_dimensions = gio.cut_dimensions(dataset=datalist, indices_of_dimensions_to_keep=indices_of_datasets_to_use)
    dataset_normalized = gio.normalize_dataset(datalist, y_scalar=None, x_scalar_list=None, pos_scalar=None, directory_path=path_to_save_dataloader)
    
    train_dl, valid_dl, test_dl = gio.create_dataloaders(batch_size = batch_size, dataset=dataset_normalized, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    gio.save_dataloader(test_dl, path_to_save_dataloader + 'test_dl_' + unique_model_description + '.pt')
    gio.save_dataloader_params(test_dl, path_to_save_dataloader + 'test_loader_params_' + unique_model_description+ '.json')
    
    print(f"Running with {torch.cuda.device_count()} GPUS")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Name is ", torch.cuda.get_device_name())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.login()
    wandb.init(
        project=project_name,
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "early_stopping_patience": 10,
            "hidden_layer_size": hidden_size_parameter,
            "gat_layers": gat_layer_parameter,
            "gcn_layers": gcn_layer_parameter,
            "output_layer": output_layer_parameter,
            # "dropout": 0.15,
        }
    )
    config = wandb.config

    print("output_layer: ", output_layer_parameter)
    print("hidden_size: ", hidden_size_parameter)
    print("gat_layers: ", gat_layer_parameter)
    print("gcn_layers: ", gcn_layer_parameter)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopping = gio.EarlyStopping(patience=early_stopping_patience, verbose=True)

    gnn_instance = garch.MyGnn(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size_parameter, gat_layers=gat_layer_parameter, gcn_layers=gcn_layer_parameter, output_layer=output_layer_parameter)
    model = gnn_instance.to(device)

    best_val_loss, best_epoch = garch.train(model, config=config, 
                                    loss_fct=loss_fct, 
                                    optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0),
                                    train_dl=train_dl, valid_dl=valid_dl,
                                    device=device, early_stopping=early_stopping)
    
if __name__ == '__main__':
    main()