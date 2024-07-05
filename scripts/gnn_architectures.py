import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, Subset
import torch_geometric
from torch_geometric.data import Data, Batch

from torch_geometric.nn import PointNetConv

class GnnBasic(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = torch_geometric.nn.GCNConv(1, 16)
        # self.conv2 = torch_geometric.nn.GATConv(16, 16)
        self.conv3 = torch_geometric.nn.GCNConv(16, 1)
        # self.conv3 = torch_geometric.nn.GCNConv(16, 1)
        # self.gat1 = torch_geometric.nn.GATConv(16, 16)
        # self.conv4 = torch_geometric.nn.GCNConv(16, 1)
                
        # self.convWithPos = torch_geometric.nn.conv.PointNetConv(1, 16, 3)
        
    def forward(self, data):
        x, edge_index = data.normalized_x, data.edge_index
        x = x[:, [0]]
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.gat1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv4(x, edge_index)
        return x

class GnnMultipleInputFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = torch_geometric.nn.GCNConv(2, 16)
        # self.conv2 = torch_geometric.nn.GATConv(16, 16)
        self.conv3 = torch_geometric.nn.GCNConv(16, 1)
        self.weight_first_dim = 2.0
        # self.conv3 = torch_geometric.nn.GCNConv(16, 1)
        # self.gat1 = torch_geometric.nn.GATConv(16, 16)
        # self.conv4 = torch_geometric.nn.GCNConv(16, 1)
                
        # self.convWithPos = torch_geometric.nn.conv.PointNetConv(1, 16, 3)
        
    def forward(self, data):
        x, edge_index = data.normalized_x, data.edge_index
        x = x[:, [0, 2]]
        # x = x[:, [0, 3]]
        x[:, 0] *= self.weight_first_dim
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.gat1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv4(x, edge_index)
        return x
    
    
class GnnWithPos(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        local_MLP_1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        global_MLP_1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )
        
        self.conv1 = PointNetConv(local_nn = local_MLP_1, global_nn = global_MLP_1)
        
        # local_MLP_2 = nn.Sequential(
        #     nn.Linear(128, 32),
        # )
        
        # global_MLP_2 = nn.Sequential(
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, out_channels)
        # )
        # self.conv2 = PointNetConv(local_nn = local_MLP_2, global_nn = global_MLP_2)

    def forward(self, x, pos, edge_index):
        x = self.conv1(x=x, pos=pos, edge_index=edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x=x, pos=pos, edge_index=edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        return x
    
    
def validate_model_basic(model, valid_dl, loss_func, device):
    model.eval()
    val_loss = 0
    num_batches = 0
    with torch.inference_mode():
        for idx, data in enumerate(valid_dl):
            input_node_features, targets = data.normalized_x.to(device), data.normalized_y.to(device)
            predicted = model(data)
            val_loss += loss_func(predicted, targets).item()
            num_batches += 1
    return val_loss / num_batches if num_batches > 0 else 0


def validate_model_pos_features(model, valid_dl, loss_func, device):
    model.eval()
    val_loss = 0
    num_batches = 0
    with torch.inference_mode():
        for idx, data in enumerate(valid_dl):
            input_node_features, targets = data.normalized_x.to(device), data.normalized_y.to(device)
            predicted = model(data.normalized_x, data.normalized_pos, data.edge_index)
            val_loss += loss_func(predicted, targets).item()
            num_batches += 1
    return val_loss / num_batches if num_batches > 0 else 0