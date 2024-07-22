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
import math
            
class MyGnn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_layers_size: int, hidden_layer_size_structure: list, gat_and_conv_structure: list):
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
        self.hidden_size = hidden_layers_size
        self.hidden_layer_size_structure = hidden_layer_size_structure
        self.gat_and_conv_structure = gat_and_conv_structure
        self.graph_layers = []
        layers = []
        
        # Architecture of PointNetConv
        local_MLP_1 = nn.Sequential(
            nn.Linear(in_channels, hidden_layers_size),
            nn.ReLU(),
            nn.Linear(hidden_layers_size, hidden_layers_size),
        )
        global_MLP_1 = nn.Sequential(
            nn.Linear(hidden_layers_size, int(hidden_layers_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_layers_size/2), int(hidden_layers_size*2)),
            nn.ReLU(),
            nn.Linear(int(hidden_layers_size*2), hidden_layers_size)
        )
        self.pointLayer = PointNetConv(local_nn = local_MLP_1, global_nn = global_MLP_1)
        
        hidden_layer_structure = define_hidden_layer_structure(hidden_layer_size_structure= self.hidden_layer_size_structure, hidden_layer_size=self.hidden_size, output_layer_size=self.out_channels)
        layers = define_gat_and_conv_layers(hidden_layer_structure=hidden_layer_structure, gat_and_conv_structure=self.gat_and_conv_structure)
        
        # Create the Sequential module with the layers
        if layers:
            self.graph_layers = torch_geometric.nn.Sequential('x, edge_index', layers)
                    
        print("Model initialized")
        print(self)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.pointLayer(x, data.pos, edge_index)
        if self.graph_layers:
            x = self.graph_layers(x, edge_index)
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

def train(model, config=None, loss_fct=None, optimizer=None, train_dl=None, valid_dl=None, device=None, early_stopping=None, accumulation_steps:int=3, use_existing_checkpoint:bool=False, path_existing_checkpoints:str=None, compute_r_squared:bool = False, model_save_path:str=None):
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
    start_epoch = 0

    # Check if a checkpoint exists and load it
    if use_existing_checkpoint:
        latest_checkpoint = get_latest_checkpoint(path_existing_checkpoints)
        if latest_checkpoint:
            model, optimizer, start_epoch, _, _ = load_checkpoint(latest_checkpoint, model, optimizer)
        
    total_steps = config.epochs * len(train_dl)
    scheduler = LinearWarmupCosineDecayScheduler(optimizer.param_groups[0]['lr'], warmup_steps=10, total_steps=total_steps)

    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.epochs):
        model.train()
        if compute_r_squared:
            actual_vals = []
            predictions = []
        optimizer.zero_grad()
        
        for idx, data in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{config.epochs}"):
            step = epoch * len(train_dl) + idx
            lr = scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
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
        
        # Save the model if validation loss improves
        if val_loss < best_val_loss and model_save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved to {model_save_path} with validation loss: {val_loss}')
            
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    
    print("Best validation loss: ", val_loss)
    wandb.summary["val_loss"] = val_loss
    wandb.finish()
    return val_loss, epoch

def define_hidden_layer_structure(hidden_layer_size_structure: list, hidden_layer_size: int, output_layer_size = int):
    """
    Generates a list of hidden layer sizes based on an initial size and a list of instructions.

    Parameters:
    list_of_halfs_and_duplicates (list): List of instructions where 1 means double the size,
                                         0 means the same size, and -1 means half the size.
    hidden_layer_size (int): The initial size of the hidden layer.

    Returns:
    list: A list of integers representing the sizes of the hidden layers.
    """
    if not all(isinstance(i, int) and i in [-1, 0, 1] for i in hidden_layer_size_structure):
        raise ValueError("list_of_halfs_and_duplicates must contain only -1, 0, or 1.")
    if hidden_layer_size <= 0:
        raise ValueError("hidden_layer_size must be a positive integer.")
    
    result_list = [hidden_layer_size]
    for i in hidden_layer_size_structure:
        if i == 1:
            result_list.append(int(result_list[-1] * 2))
        elif i == 0:
            result_list.append(result_list[-1])
        elif i == -1:
            result_list.append(int(result_list[-1] / 2))
    result_list.append(output_layer_size)
    return result_list

def define_gat_and_conv_layers(hidden_layer_structure: list, gat_and_conv_structure: list):
    """
    Generates a list of GNN layers and ReLU activations based on the provided hidden layer structure.

    Parameters:
    hidden_layer_structure (list[int]): A list of integers representing the sizes of the hidden layers.
    gat_and_conv_structure (list[int]): A list specifying the type of GNN layer to use. 
        Use 1 for 'GATConv' and -1 for 'GCNConv'.
        Note that the size of hidden_layer_structure must be the same size as gat_and_conv_structure.

    Returns:
    list: A list of tuples and ReLU activations, where each tuple contains a GNN layer and a string describing the data flow.
    
    Raises:
    ValueError: If an invalid layer type is specified or if the input lengths are incompatible.
    """
    
    if len(hidden_layer_structure) != len(gat_and_conv_structure):
        raise ValueError("The size of hidden_layer_structure and gat_and_conv_structure must be the same!")

    # Mapping layer types to their corresponding classes
    layer_types = {
        1: torch_geometric.nn.GATConv,
        -1: torch_geometric.nn.GCNConv
    }

    layers = []
    for idx in range(len(hidden_layer_structure) - 1):
        layer_type = gat_and_conv_structure[idx]
        if layer_type in layer_types:
            layer_class = layer_types[layer_type]
            layers.append((layer_class(hidden_layer_structure[idx], hidden_layer_structure[idx + 1]), 'x, edge_index -> x'))
        else:
            raise ValueError("Invalid layer_type. Choose 1 for 'GATConv' or -1 for 'GCNConv'.")
        layers.append(torch.nn.ReLU(inplace=True))
    return layers

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

class LinearWarmupCosineDecayScheduler:
    def __init__(self, initial_lr, warmup_steps, total_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.initial_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.initial_lr * cosine_decay 
        
        

