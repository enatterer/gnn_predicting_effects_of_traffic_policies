import math
import numpy as np
import wandb

import torch
import torch_geometric
from torch_geometric.data import Data

from scripts.gnn_architectures import MyGnn
import scripts.gnn_io as gio
import scripts.gnn_architectures as garch
import pprint
import itertools

def train(model, config=None, loss_fct=None, optimizer=None, train_dl=None, valid_dl=None, device=None, early_stopping=None):
    for epoch in range(config.epochs):
        model.train()
        for idx, data in enumerate(train_dl):
            input_node_features, targets = data.x.to(device), data.y.to(device)
            optimizer.zero_grad()

            # Forward pass
            predicted = model(data)
            train_loss = loss_fct(predicted, targets)
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            wandb.log({"train_loss": train_loss.item(), "epoch": epoch, "step": idx})
            # print(f"epoch: {epoch}, step: {idx}, loss: {train_loss.item()}")
        
        val_loss = garch.validate_model(model, valid_dl, loss_fct, device)
        print(f"epoch: {epoch}, validation loss: {val_loss}")
        wandb.log({"loss": val_loss, "epoch": epoch})
            
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    
    print("Best validation loss: ", val_loss)
    wandb.summary["val_loss"] = val_loss
    wandb.finish()
    return val_loss, epoch
    

def main():
    # Define parameters 
    num_epochs = 1000
    project_name = 'finetuning_model_architecture'
    train_ratio = 0.8
    
    # Reconstruct the Data objects
    data_dict_list = torch.load('../data/dataset_1pm_0-1382.pt')
    datalist = [Data(x=d['x'], edge_index=d['edge_index'], pos=d['pos'], y=d['y']) for d in data_dict_list]
    dataset_normalized = gio.normalize_dataset(datalist)
    
    batch_size_range = [16]
    output_layer_range = ['gat', 'gcn']
    hidden_size_range = [16, 32, 64]
    gat_layers_range = [0, 1]
    gcn_layers_range = [1]
    lr_range = [0.001]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopping = gio.EarlyStopping(patience=5, verbose=True)

    # Create a list to hold all configurations
    configurations = list(itertools.product(batch_size_range, lr_range, output_layer_range, hidden_size_range, gat_layers_range, gcn_layers_range))
    print(configurations)
    wandb.login()
    counter = 0
    # Open the text file for writing
    with open('model_performance.txt', 'w') as f:
        for config in configurations:
            counter +=1
            print(f"Configuration {counter}/{len(configurations)}")
            batch_size, lr, output_layer_parameter, hidden_size_parameter, gat_layer_parameter, gcn_layer_parameter = config

            train_dl = gio.create_dataloader(dataset=dataset_normalized, is_train=True, batch_size=batch_size, train_ratio=train_ratio)
            valid_dl = gio.create_dataloader(dataset=dataset_normalized, is_train=False, batch_size=batch_size, train_ratio=train_ratio)
            
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
            
            gnn_instance = MyGnn(in_channels=3, out_channels=1, hidden_size=hidden_size_parameter, gat_layers=gat_layer_parameter, gcn_layers=gcn_layer_parameter, output_layer=output_layer_parameter)
            model = gnn_instance.to(device)
            
            best_val_loss, best_epoch = train(model, config=config, 
                                            loss_fct=torch.nn.MSELoss(), 
                                            optimizer=torch.optim.Adam(model.parameters(), lr=lr),
                                            train_dl=train_dl, valid_dl=valid_dl,
                                            device=device, early_stopping=early_stopping)
            
            # Write the configuration and the best loss to the text file
            f.write(f"Configuration: batch_size={batch_size}, lr={lr}, output_layer={output_layer_parameter}, hidden_size={hidden_size_parameter}, gat_layers={gat_layer_parameter}, gcn_layers={gcn_layer_parameter}\n")
            f.write(f"Best Validation Loss: {best_val_loss}, Best Epoch: {best_epoch}\n\n")
    
if __name__ == '__main__':
    main()