import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, Subset
import torch_geometric
from torch_geometric.data import Data, Batch

from torch_geometric.nn import PointNetConv

class MyGnn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int, gat_layers: int, 
                 gcn_layers: int, output_layer: str = 'gcn', graph_layers_before: bool = False):
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
        self.ot_channels = out_channels
        self.hidden_size = hidden_size
        self.gat_layers = gat_layers
        self.gcn_layers = gcn_layers
        self.output_layer = output_layer
        self.graph_layers_before = graph_layers_before
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
        
        if self.graph_layers_before:
            if self.graph_layers:
                x = self.graph_layers(x, edge_index)
                
        x = self.pointLayer(x, data.pos, edge_index)
        
        if not self.graph_layers_before:
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