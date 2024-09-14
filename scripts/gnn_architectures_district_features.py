import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import PointNetConv, Sequential as GeoSequential
import os
import math

class MyGnn(torch.nn.Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                point_net_conv_layer_structure_local_mlp: list, 
                point_net_conv_layer_structure_global_mlp: list, 
                gat_conv_layer_structure: list, 
                graph_mlp_layer_structure: list,
                dropout: float = 0.0, 
                use_dropout: bool = False):
        """
        Initialize the GNN model with specified configurations.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - point_net_conv_layer_structure_local_mlp (list): Layer structure for local MLP in PointNetConv.
        - point_net_conv_layer_structure_global_mlp (list): Layer structure for global MLP in PointNetConv.
        - gat_conv_layer_structure (list): Layer structure for GATConv layers.
        - dropout (float, optional): Dropout rate. Default is 0.0.
        - use_dropout (bool, optional): Whether to use dropout. Default is False.
        """
        super(MyGnn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pnc_local = point_net_conv_layer_structure_local_mlp
        self.pnc_global = point_net_conv_layer_structure_global_mlp
        self.gat_conv_layer_structure = gat_conv_layer_structure
        self.graph_mlp_structure = graph_mlp_layer_structure
        self.use_dropout = use_dropout

        if use_dropout:
            self.dropout_layer = nn.Dropout(dropout)
        
        local_nn_1, global_nn_1 = self.create_point_net_layer(gat_conv_starts_with_layer=self.gat_conv_layer_structure[0], is_first_layer=True, is_last_layer=False)
        local_nn_2, global_nn_2 = self.create_point_net_layer(gat_conv_starts_with_layer=self.gat_conv_layer_structure[0], is_first_layer=False, is_last_layer=False)
        local_nn_3, global_nn_3 = self.create_point_net_layer(gat_conv_starts_with_layer=self.gat_conv_layer_structure[0], is_first_layer=False, is_last_layer=True)
        
        self.point_net_conv_1 = PointNetConv(local_nn=local_nn_1, global_nn=global_nn_1)
        self.point_net_conv_2 = PointNetConv(local_nn=local_nn_2, global_nn=global_nn_2)
        self.point_net_conv_3 = PointNetConv(local_nn=local_nn_3, global_nn=global_nn_3)
    
        layers = self.define_layers()
        self.gat_graph_layers = GeoSequential('x, edge_index', layers)
        
        graph_mlp_layers = []
        graph_mlp_layers.append(nn.Linear(2, self.graph_mlp_structure[0]))
        for idx in range(len(self.graph_mlp_structure) - 1):
            graph_mlp_layers.append(nn.Linear(self.graph_mlp_structure[idx], self.graph_mlp_structure[idx + 1]))
            graph_mlp_layers.append(nn.ReLU())
            if use_dropout:
                graph_mlp_layers.append(self.dropout_layer)
        graph_mlp_layers.append(nn.Linear(self.graph_mlp_structure[-1], out_channels))
        self.graph_mlp = nn.Sequential(*graph_mlp_layers)
        self.initialize_weights()
        print("Model initialized")
    
    
    def forward(self, data):
        """
        Forward pass for the GNN model.

        Parameters:
        - data (Data): Input data containing node features and edge indices.

        Returns:
        - torch.Tensor: Output features after passing through the model.
        """
        
        x = data.x
        edge_index = data.edge_index
        mode_stats = data.mode_stats
        
        pos1 = data.pos[:, 0, :]  # First set of positions
        pos2 = data.pos[:, 1, :]  # Second set of positions
        pos3 = data.pos[:, 2, :]  # Third set of positions
        
        x = self.point_net_conv_1(x, pos1, edge_index)
        x = self.point_net_conv_2(x, pos2, edge_index)
        x = self.point_net_conv_3(x, pos3, edge_index)
        
        x = self.gat_graph_layers(x, edge_index)
        
        graph_output = self.graph_mlp(mode_stats)
        
        return x, graph_output
    
    def process_global_graph_attributes(self):
        """
        Process graph-level attributes.

        Parameters:
        - mode_stats (torch.Tensor): Graph-level attributes.

        Returns:
        - torch.Tensor: Processed graph-level attributes.
        """
        layers = []
        for i in range(len(self.graph_mlp_layer_structure) - 1):
            layers.append(nn.Linear(self.graph_mlp_layer_structure[i], self.graph_mlp_layer_structure[i + 1]))
            if i < len(self.graph_mlp_layer_structure) - 2:  # Add ReLU between layers, but not after the last layer
                layers.append(nn.ReLU())
            if self.use_dropout:
                layers.append(self.dropout_layer)        
        return layers
    
    # TODO Elena: introduce smooth transition from one layer to the next. 
    def create_point_net_layer(self, gat_conv_starts_with_layer:int, is_first_layer:bool = False, is_last_layer:bool = False):
        """
        Create PointNetConv layers with specified configurations.

        Parameters:
        - gat_conv_starts_with_layer (int): Starting layer size for GATConv.

        Returns:
        - Tuple[nn.Sequential, nn.Sequential]: Local and global MLP layers.
        """
        # Create local MLP layers
        
        local_MLP_layers = []
        offset_for_first_layer = 2
        if is_first_layer:  
            local_MLP_layers.append(nn.Linear(self.in_channels + offset_for_first_layer, self.pnc_local[0]))
        else:
            local_MLP_layers.append(nn.Linear(self.pnc_global[-1] + offset_for_first_layer, self.pnc_local[0]))
        local_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            local_MLP_layers.append(self.dropout_layer)
        for idx in range(len(self.pnc_local)-1):
            local_MLP_layers.append(nn.Linear(self.pnc_local[idx], self.pnc_local[idx + 1]))
            local_MLP_layers.append(nn.ReLU())
            if self.use_dropout:
                local_MLP_layers.append(self.dropout_layer)
        local_MLP = nn.Sequential(*local_MLP_layers)
        
        global_MLP_layers = []
        global_MLP_layers.append(nn.Linear(self.pnc_local[-1], self.pnc_global[0]))
        for idx in range(len(self.pnc_global) - 1):
            global_MLP_layers.append(nn.Linear(self.pnc_global[idx], self.pnc_global[idx + 1]))
            global_MLP_layers.append(nn.ReLU())
            if self.use_dropout:
                global_MLP_layers.append(self.dropout_layer)
        if is_last_layer:
            global_MLP_layers.append(nn.Linear(self.pnc_global[ - 1], gat_conv_starts_with_layer))
            global_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            global_MLP_layers.append(self.dropout_layer)
        global_MLP = nn.Sequential(*global_MLP_layers)
        return local_MLP, global_MLP
    
    def define_layers(self):
        """
        Define layers for GATConv based on configuration.

        Returns:
        - List: Layers for GATConv.
        """
        layers = []
        for idx in range(len(self.gat_conv_layer_structure) - 1):
            layers.append((torch_geometric.nn.GATConv(self.gat_conv_layer_structure[idx], self.gat_conv_layer_structure[idx + 1]), 'x, edge_index -> x'))
            layers.append(nn.ReLU(inplace=True))
            if self.use_dropout:
                layers.append(self.dropout_layer)
        layers.append((torch_geometric.nn.GATConv(self.gat_conv_layer_structure[-1], self.out_channels), 'x, edge_index -> x'))
        return layers
    
    def initialize_weights(self):
        """
        Initialize model weights using Xavier and Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, PointNetConv):
                self._initialize_pointnetconv(m)
            elif isinstance(m, torch_geometric.nn.GATConv):
                self._initialize_gatconv(m)

    def _initialize_pointnetconv(self, m: PointNetConv):
        """
        Initialize weights for PointNetConv layers.

        Parameters:
        - m (PointNetConv): PointNetConv layer to initialize.
        """
        for name, param in m.local_nn.named_parameters():
            if param.dim() > 1:  # weight parameters
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:  # bias parameters
                init.zeros_(param)
        for name, param in m.global_nn.named_parameters():
            if param.dim() > 1:  # weight parameters
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:  # bias parameters
                init.zeros_(param)

    def _initialize_gatconv(self, m: torch_geometric.nn.GATConv):
        """
        Initialize weights for GATConv layers.

        Parameters:
        - m (GATConv): GATConv layer to initialize.
        """
        if hasattr(m, 'lin') and m.lin is not None:
            init.xavier_normal_(m.lin.weight)
            if m.lin.bias is not None:
                init.zeros_(m.lin.bias)
        if hasattr(m, 'att_src') and m.att_src is not None:
            init.xavier_normal_(m.att_src)
        if hasattr(m, 'att_dst') and m.att_dst is not None:
            init.xavier_normal_(m.att_dst)


def train(model: nn.Module, 
          config: object = None, 
          loss_fct: nn.Module = None, 
          optimizer: optim.Optimizer = None, 
          train_dl: DataLoader = None, 
          valid_dl: DataLoader = None, 
          device: torch.device = None, 
          early_stopping: object = None, 
          accumulation_steps: int = 3, 
          model_save_path: str = None, 
          use_gradient_clipping: bool = True,
          lr_scheduler_warmup_steps: int = 20000, 
          lr_scheduler_cosine_decay_rate: float = 0.2) -> tuple:
    """
    Train the GNN model.

    Parameters:
    - model (nn.Module): The model to train.
    - config (object, optional): Configuration object containing training parameters.
    - loss_fct (nn.Module, optional): Loss function for training.
    - optimizer (optim.Optimizer, optional): Optimizer for model training.
    - train_dl (DataLoader, optional): DataLoader for training data.
    - valid_dl (DataLoader, optional): DataLoader for validation data.
    - device (torch.device, optional): Device to use for training.
    - early_stopping (object, optional): Early stopping mechanism.
    - accumulation_steps (int, optional): Number of steps for gradient accumulation. Default is 3.
    - model_save_path (str, optional): Path to save the best model.
    - use_gradient_clipping (bool, optional): Whether to use gradient clipping. Default is True.
    - lr_scheduler_warmup_steps (int, optional): Number of warmup steps for learning rate scheduler. Default is 20000.
    - lr_scheduler_cosine_decay_rate (float, optional): Cosine decay rate for learning rate scheduler. Default is 0.2.

    Returns:
    - tuple: Validation loss and the best epoch.
    """
    scaler = GradScaler()
    total_steps = config.epochs * len(train_dl)
    scheduler = LinearWarmupCosineDecayScheduler(optimizer.param_groups[0]['lr'], warmup_steps=lr_scheduler_warmup_steps, total_steps=total_steps, cosine_decay_rate=lr_scheduler_cosine_decay_rate)
    best_val_loss = float('inf')
    
    # Create a directory for checkpoints if it doesn't exist
    checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        for idx, data in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{config.epochs}"):
            step = epoch * len(train_dl) + idx
            lr = scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            data = data.to(device)
            targets, graph_attributes = data.y, data.mode_stats
           
            with autocast():
                # Forward pass
                predicted, output_graph_attributes = model(data)
                
                # Compute losses
                node_edge_loss = loss_fct(predicted, targets)
                # print(f"node_edge_loss: {node_edge_loss}")
                graph_loss = loss_fct(output_graph_attributes, graph_attributes)
                # print(f"graph_loss: {graph_loss}")
                
                # Normalize losses to have equal weight
                node_edge_loss_weight = node_edge_loss / (node_edge_loss + graph_loss)
                graph_loss_weight = graph_loss / (node_edge_loss + graph_loss)
                train_loss = graph_loss_weight * node_edge_loss + node_edge_loss_weight * graph_loss
                
            # Backward pass
            scaler.scale(train_loss).backward() 
            
            # Gradient clipping
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Do not log train loss at every iteration, as it uses CPU
            if (idx + 1) % 10 == 0:
                wandb.log({"train_loss": train_loss.item(), "epoch": epoch})
        
        if len(train_dl) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        val_loss, r_squared = validate_model_during_training(model=model, dataset=valid_dl, loss_func=loss_fct, device=device)
        wandb.log({"val_loss": val_loss, "epoch": epoch, "lr": lr, "r^2": r_squared})
        print(f"epoch: {epoch}, validation loss: {val_loss}, lr: {lr}, r^2: {r_squared}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss   
            if model_save_path:         
                torch.save(model.state_dict(), model_save_path)
                print(f'Best model saved to {model_save_path} with validation loss: {val_loss}')
        
        # Save checkpoint
        if epoch % 20 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_loss': val_loss,
                # 'r_squared': r_squared
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    
    print("Best validation loss: ", best_val_loss)
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.finish()
    return val_loss, epoch

def validate_model_during_training(model: nn.Module, 
                                   dataset: DataLoader, 
                                   loss_func: nn.Module, 
                                   device: torch.device) -> tuple:
    """
    Validate the model during training and compute validation loss and R^2 score.

    Parameters:
    - model (nn.Module): The model to validate.
    - dataset (DataLoader): DataLoader for validation data.
    - loss_func (nn.Module): Loss function to compute validation loss.
    - device (torch.device): Device to use for validation.

    Returns:
    - tuple: Total validation loss and R^2 score.
    """
    model.eval()
    val_loss = 0
    num_batches = 0
    actual_vals = []
    predictions = []
    with torch.inference_mode():
        for idx, data in enumerate(dataset):
            data = data.to(device)
            input_node_features, targets, graph_attributes = data.x, data.y, data.mode_stats
            predicted, output_graph_attributes = model(data)

            actual_vals.append(targets)
            predictions.append(predicted)
            
            node_edge_loss = loss_func(predicted, targets).item()
            graph_loss = loss_func(output_graph_attributes, graph_attributes).item()

            val_loss += node_edge_loss + graph_loss
            num_batches += 1
            
    total_validation_loss = val_loss / num_batches if num_batches > 0 else 0
    actual_vals=torch.cat(actual_vals)
    predictions = torch.cat(predictions)
    r_squared = compute_r2_torch(preds=predictions, targets=actual_vals)
    return total_validation_loss, r_squared

def compute_r2_torch(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute R^2 score using PyTorch.

    Parameters:
    - preds (torch.Tensor): Predicted values.
    - targets (torch.Tensor): Actual target values.

    Returns:
    - torch.Tensor: Computed R^2 score.
    """
    mean_targets = torch.mean(targets)
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Retrieve the latest checkpoint file from the specified directory.

    Parameters:
    - checkpoint_dir (str): Directory where checkpoint files are stored.

    Returns:
    - str: Path to the latest checkpoint file if it exists, otherwise None.
    """
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer = None) -> tuple:
    """
    Load a checkpoint and restore the model and optimizer states.

    Parameters:
    - checkpoint_path (str): Path to the checkpoint file.
    - model (nn.Module): The model to load the state dict into.
    - optimizer (optim.Optimizer, optional): The optimizer to load the state dict into.

    Returns:
    - tuple: Restored model, optimizer, epoch, validation loss, and training loss.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    train_loss = checkpoint['train_loss']
    print(f'Loaded checkpoint from epoch {epoch} with val_loss {val_loss} and train_loss {train_loss}')
    return model, optimizer, epoch, val_loss, train_loss

def save_checkpoint(model: nn.Module, 
                    optimizer: optim.Optimizer, 
                    epoch: int, 
                    val_loss: float, 
                    train_loss: float, 
                    checkpoint_path: str) -> None:
    """
    Save a checkpoint of the model and optimizer states.

    Parameters:
    - model (nn.Module): The model to save.
    - optimizer (optim.Optimizer): The optimizer to save.
    - epoch (int): The current epoch.
    - val_loss (float): Validation loss at the current epoch.
    - train_loss (float): Training loss at the current epoch.
    - checkpoint_path (str): Path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Model checkpoint saved at epoch {epoch}')

def load_model(model_path: str) -> tuple:
    """
    Load a saved model checkpoint and initialize the model with the configuration.

    Parameters:
    - model_path (str): Path to the model checkpoint file.

    Returns:
    - tuple: Loaded model and configuration.
    """
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
    def __init__(self, 
                 initial_lr: float, 
                 warmup_steps: int, 
                 total_steps: int, 
                 cosine_decay_rate: float = 0.5):
        """
        Linear warmup and cosine decay scheduler.

        Parameters:
        - initial_lr (float): Initial learning rate.
        - warmup_steps (int): Number of warmup steps.
        - total_steps (int): Total number of steps.
        - cosine_decay_rate (float, optional): Cosine decay rate. Default is 0.5.
        """
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cosine_decay_rate = cosine_decay_rate
        self.decay_steps = total_steps - warmup_steps

    def get_lr(self, step: int) -> float:
        """
        Get the learning rate at a specific step.

        Parameters:
        - step (int): The current step.

        Returns:
        - float: Calculated learning rate.
        """
        if step < self.warmup_steps:
            return self.initial_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / self.decay_steps
            cosine_decay = self.cosine_decay_rate * (1 + math.cos(math.pi * progress))
            return self.initial_lr * cosine_decay 