import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, Subset
import torch_geometric
from torch_geometric.data import Data, Batch
from sklearn.metrics import r2_score

from torch_geometric.nn import PointNetConv
from tqdm import tqdm
import wandb
import numpy as np
import os
from torch.cuda.amp import GradScaler, autocast

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
    
    
class MyGnnHardCoded(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int, output_layer: str = 'gcn'):
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
                
        layers.append((torch_geometric.nn.GATConv(hidden_size, int(hidden_size*2)), 'x, edge_index -> x'))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append((torch_geometric.nn.GATConv(int(hidden_size*2), int(hidden_size/2)), 'x, edge_index -> x'))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append((torch_geometric.nn.GATConv(int(hidden_size/2), int(hidden_size*2)), 'x, edge_index -> x'))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append((torch_geometric.nn.GATConv(int(hidden_size*2), hidden_size), 'x, edge_index -> x'))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append((torch_geometric.nn.GATConv(hidden_size, hidden_size), 'x, edge_index -> x'))
        layers.append(nn.ReLU(inplace=True))
        
        
        # for _ in range(gcn_layers):
        #     layers.append((torch_geometric.nn.GCNConv(hidden_size, hidden_size), 'x, edge_index -> x'))
        #     layers.append(nn.ReLU(inplace=True))

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

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    train_loss = checkpoint['train_loss']
    print(f'Loaded checkpoint from epoch {epoch} with val_loss {val_loss} and train_loss {train_loss}')
    return model, optimizer, epoch, val_loss, train_loss

def train(model, config=None, loss_fct=None, optimizer=None, train_dl=None, valid_dl=None, device=None, early_stopping=None, accumulation_steps:int=3, save_checkpoints:bool=True, iteration_save_checkpoint:int=5, use_existing_checkpoint:bool=False, path_existing_checkpoints:str=None, compute_r_squared:bool = False):
    """
    Trains the machine learning model with the specified configurations.

    Parameters:
    model: torch.nn.Module
        The model to be trained, a GNN.
    config: Config, optional
        Configuration object containing training hyperparameters such as the number of epochs.
    loss_fct: callable, optional
        The loss function to be used during training.
    optimizer: torch.optim.Optimizer, optional
        The optimizer to use for model parameter updates.
    train_dl: DataLoader, optional
        DataLoader for the training dataset.
    valid_dl: DataLoader, optional
        DataLoader for the validation dataset.
    device: torch.device, optional
        Device on which to perform training (e.g., 'cpu' or 'cuda').
    early_stopping: EarlyStopping, optional
        EarlyStopping object to handle early stopping based on validation loss.
    accumulation_steps: int, default=3
        Number of steps to accumulate gradients before performing an optimizer step.
    save_checkpoints: bool, default=True
        Flag indicating whether to save checkpoints. 
    iteration_save_checkpoint: int, default=5
        At which iterations should the model be saved. 
    use_existing_checkpoint: bool, default=False
        Flag indicating whether to use an existing checkpoint to resume training.
    path_existing_checkpoints: str, optional
        Path to the directory containing existing checkpoints.
    compute_r_squared: bool, default=False
        Flag indicating whether to compute and log the R^2 metric during training. We found that it uses a lot of CPU, because numpy only works on CPU, therefore, it is recommended to switch it off for training with limited CPU usage.

    Returns:
    val_loss: float
        The best validation loss achieved during training.
    epoch: int
        The epoch at which the best validation loss was achieved.
    """
    
    scaler = GradScaler()
    
    # Check if a checkpoint exists and load it
    if use_existing_checkpoint:
        latest_checkpoint = get_latest_checkpoint(path_existing_checkpoints)
        if latest_checkpoint:
            model, optimizer, start_epoch, _, _ = load_checkpoint(latest_checkpoint, model, optimizer)
    
    for epoch in range(config.epochs):
        model.train()
        if compute_r_squared:
            actual_vals = []
            predictions = []
        optimizer.zero_grad()
        total_train_loss = 0
        
        for idx, data in tqdm(enumerate(train_dl)):
            input_node_features, targets = data.x.to(device), data.y.to(device)
            with autocast():
                # Forward pass
                predicted = model(data.to(device))
                train_loss = loss_fct(predicted, targets)

            if compute_r_squared:
                actual_vals.extend(targets.cpu().detach().numpy())
                predictions.extend(predicted.cpu().detach().numpy())
            
            # Backward pass
            scaler.scale(train_loss).backward() 

            if (idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Do not log train loss at every iteration, as it uses CPU
            if (idx + 1) % 300 == 0:
                wandb.log({"train_loss": train_loss.item(), "epoch": epoch, "step": idx})
        
        if len(train_dl) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        if compute_r_squared:
            actual_vals = np.array(actual_vals)
            predictions = np.array(predictions)
            # Calculate R^2
            r2 = r2_score(actual_vals, predictions)

        val_loss = validate_model_pos_features(model, valid_dl, loss_fct, device)
        
        # Log and print validation loss and R^2 if compute_r_squared is True
        log_data = {"loss": val_loss, "epoch": epoch}
        if compute_r_squared:
            log_data["R^2"] = r2
        print(f"epoch: {epoch}, validation loss: {val_loss}")
        wandb.log(log_data)
        
        # Save model checkpoint every 5 epochs
        if save_checkpoints and (epoch + 1) % iteration_save_checkpoint == 0:
            # total_train_loss /= len(train_dl)
            checkpoint_path = f"../../data/checkpoints/checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, total_train_loss, checkpoint_path)
            
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    
    print("Best validation loss: ", val_loss)
    wandb.summary["val_loss"] = val_loss
    wandb.finish()
    return val_loss, epoch

def save_checkpoint(model, optimizer, epoch, val_loss, train_loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Model checkpoint saved at epoch {epoch}')

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