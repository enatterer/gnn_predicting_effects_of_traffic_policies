import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, Subset
import torch_geometric
from torch_geometric.data import Data, Batch

from torch_geometric.nn import PointNetConv
from tqdm import tqdm
import wandb
import numpy as np

class MyGnn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int, gat_layers: int, 
                 gcn_layers: int, output_layer: str = 'gcn'):
        """
        in_channels: number of input features
        out_channels: number of output features
        hidden_size: size of hidden layer
        gat_layers: number of GAT layers
        heads: number of attention heads
        gcn_layers: number of GCN layers
        output_layer: 'gat' or 'gcn'
        """
        super().__init__()
        
        # Hyperparameters 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.gat_layers = gat_layers
        self.gcn_layers = gcn_layers
        self.output_layer = output_layer
        self.graph_layers = []
        layers = []
        
        # Architecture
        local_MLP_1 = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        global_MLP_1 = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), int(hidden_size*2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size*2), hidden_size)
        )
        
        self.pointLayer = PointNetConv(local_nn = local_MLP_1, global_nn = global_MLP_1)
                
        # Add GAT layers and ReLU activations to the list
        for _ in range(gat_layers):
            layers.append((torch_geometric.nn.GATConv(hidden_size, hidden_size), 'x, edge_index -> x'))
            layers.append(nn.ReLU(inplace=True))
            
        for _ in range(gcn_layers):
            layers.append((torch_geometric.nn.GCNConv(hidden_size, hidden_size), 'x, edge_index -> x'))
            layers.append(nn.ReLU(inplace=True))

        # Create the Sequential module with the layers
        if layers:
            self.graph_layers = torch_geometric.nn.Sequential('x, edge_index', layers)
                    
        if output_layer == 'gcn':
            self.output_layer = torch_geometric.nn.GCNConv(hidden_size, out_channels)
        elif output_layer == 'gat':
            self.output_layer = torch_geometric.nn.GATConv(hidden_size, out_channels)
        else:
            raise ValueError("Invalid output layer")
            
        print("Model initialized")
        print(self)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = self.pointLayer(x, data.pos, edge_index)
        
        if self.graph_layers:
            x = self.graph_layers(x, edge_index)
            
        x = self.output_layer(x, edge_index)
        return x

def validate_model_pos_features(model, valid_dl, loss_func, device):
    model.eval()
    val_loss = 0
    num_batches = 0
    with torch.inference_mode():
        for idx, data in enumerate(valid_dl):
            input_node_features, targets = data.x.to(device), data.y.to(device)
            predicted = model(data.to(device))
            val_loss += loss_func(predicted, targets).item()
            num_batches += 1
    return val_loss / num_batches if num_batches > 0 else 0


def train(model, config=None, loss_fct=None, optimizer=None, train_dl=None, valid_dl=None, device=None, early_stopping=None):
    for epoch in range(config.epochs):
        model.train()
        actual_vals = []
        predictions = []
        for idx, data in tqdm(enumerate(train_dl)):
            input_node_features, targets = data.x.to(device), data.y.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            predicted = model(data.to(device))
            
            actual_vals.extend(targets.detach().numpy())
            predictions.extend(predicted.detach().numpy())
            
            # Backward pass
            train_loss = loss_fct(predicted, targets)
            train_loss.backward()
            optimizer.step()
            
            wandb.log({"train_loss": train_loss.item(), "epoch": epoch, "step": idx})
            # print(f"epoch: {epoch}, step: {idx}, loss: {train_loss.item()}")
        
        actual_vals = np.array(actual_vals)
        predictions = np.array(predictions)
        
        # Calculate R^2
        sst = ((actual_vals - actual_vals.mean()) ** 2).sum()
        ssr = ((actual_vals - predictions) ** 2).sum()
        r2 = 1 - ssr / sst

        val_loss = validate_model_pos_features(model, valid_dl, loss_fct, device)
        print(f"epoch: {epoch}, validation loss: {val_loss}, R^2: {r2}")
        wandb.log({"loss": val_loss, "epoch": epoch, "r2": r2})
            
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    
    print("Best validation loss: ", val_loss)
    wandb.summary["val_loss"] = val_loss
    wandb.finish()
    return val_loss, epoch

def evaluate(model, test_dl, device, loss_func=torch.nn.MSELoss()):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    output_list = []
    with torch.no_grad():  # Disable gradient computation
        for data in test_dl:
            inputs, targets = data.x.to(device), data.y.to(device)
            outputs = model(data.to(device))
            output_list.append(outputs)
            loss = loss_func(outputs, targets)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_dl)
    return avg_test_loss, output_list

def load_model(model_path):
    # Load the saved model checkpoint
    checkpoint = torch.load(model_path)
    
    # Extract the state dictionary and configuration
    state_dict = checkpoint['state_dict']
    config = checkpoint['config']
    
    # Initialize the model with the configuration
    model = MyGnn(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        hidden_size=config['hidden_size'],
        gat_layers=config['gat_layers'],
        gcn_layers=config['gcn_layers'],
        # output_layer=config['output_layer'],
        output_layer='gat'
    )
    model.load_state_dict(state_dict)
    return model, config