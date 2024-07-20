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

# Add the 'scripts' directory to the Python path
scripts_path = os.path.abspath(os.path.join('..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
    
import joblib

# Now you can import the gnn_io module
import gnn_io as gio

import gnn_architectures as garch


def main():
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define parameters 
    num_epochs = 1000
    project_name = "run_with_python_script"
    path_to_save_dataloader = "../../data/data_created_during_training_needed_for_testing/"
    indices_of_datasets_to_use = [0, 1, 3, 4]

    loss_fct = torch.nn.MSELoss()
    batch_size = 8
    output_layer_parameter = 'gat'
    hidden_size_parameter = 64
    gat_layer_parameter = 5
    gcn_layer_parameter = 0
    lr = 0.001
    in_channels = len(indices_of_datasets_to_use) + 2 # dimensions of the x vector + 2 (pos)
    out_channels = 1 # we are predicting one value
    early_stopping_patience = 10

    data_dict_list = torch.load('../../data/train_data/dataset_1pm_0-3500_new.pt')

    unique_model_description = (
        f"datasets_{indices_of_datasets_to_use}_"
        f"loss_{loss_fct.__class__.__name__}_"
        f"batch_{batch_size}_"
        f"output_{output_layer_parameter}_"
        f"hidden_{hidden_size_parameter}_"
        f"gat_layers_{gat_layer_parameter}_"
        f"gcn_layers_{gcn_layer_parameter}_"
        f"lr_{lr}_"
        f"in_channels_{in_channels}_"
        f"out_channels_{out_channels}_"
        f"early_stopping_{early_stopping_patience}"
    )

    data_dict_list = torch.load('../../data/train_data/dataset_1pm_0-3500_new.pt')
    
    # Reconstruct the Data objects
    datalist = [Data(x=d['x'], edge_index=d['edge_index'], pos=d['pos'], y=d['y']) for d in data_dict_list]
    dataset_only_relevant_dimensions = gio.cut_dimensions(dataset=datalist, indices_of_dimensions_to_keep=indices_of_datasets_to_use)
    dataset_normalized = gio.normalize_dataset(dataset_only_relevant_dimensions, y_scalar=None, x_scalar_list=None, pos_scalar=None, directory_path=path_to_save_dataloader)
    
    baseline_error = gio.compute_baseline_of_no_policies(dataset=dataset_normalized, loss_fct=loss_fct)
    print(f'Baseline error no policies: {baseline_error}')

    baseline_error = gio.compute_baseline_of_mean_target(dataset=dataset_normalized, loss_fct=loss_fct)
    print(f'Baseline error mean: {baseline_error}')
    
    train_dl, valid_dl, test_dl = gio.create_dataloaders(batch_size = batch_size, dataset=dataset_normalized, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    gio.save_dataloader(test_dl, path_to_save_dataloader + 'test_dl_' + unique_model_description + '.pt')
    gio.save_dataloader_params(test_dl, path_to_save_dataloader + 'test_loader_params_' + unique_model_description+ '.json')
    
    print(f"Running with {torch.cuda.device_count()} GPUS")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Name is ", torch.cuda.get_device_name())
    
    wandb.login()
    wandb.init(
        project=project_name,
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "early_stopping_patience": early_stopping_patience,
            "hidden_layer_size": hidden_size_parameter,
            "gat_layers": gat_layer_parameter,
            "gcn_layers": gcn_layer_parameter,
            "output_layer": output_layer_parameter,
            "indices_to_use": indices_of_datasets_to_use,
            "dataset_length": len(dataset_normalized)
        }
    )
    config = wandb.config

    print("output_layer: ", output_layer_parameter)
    print("hidden_size: ", hidden_size_parameter)
    print("gat_layers: ", gat_layer_parameter)
    print("gcn_layers: ", gcn_layer_parameter)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopping = gio.EarlyStopping(patience=early_stopping_patience, verbose=True)

    gnn_instance = garch.MyGnnHardCoded(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size_parameter, output_layer=output_layer_parameter)
    model = gnn_instance.to(device)
    best_val_loss, best_epoch = garch.train(model=model, config=config, 
                                    loss_fct=loss_fct, 
                                    optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0),
                                    train_dl=train_dl, valid_dl=valid_dl,
                                    device=device, early_stopping=early_stopping,
                                    use_existing_checkpoint=False, path_existing_checkpoints = "../../data/checkpoints_batchsize_8/")
    
if __name__ == '__main__':
    main()