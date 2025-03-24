import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
import torch_geometric.nn as geo_nn

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import PointNetConv, Sequential as GeoSequential
import os
import math
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch_geometric.nn import TransformerConv
from models.base_gnn import BaseGNN

"""
This architecture is a Graph Neural Network (GNN) model that combines PointNet Convolutions, Graph Attention Networks, and Transformer layers.
It is designed to predict the effects of traffic policies using graph-based data.
The model includes configurations for dropout, Monte Carlo dropout, and mode statistics prediction. Mode statistics prediction is not finetuned. 
This architecture was used for the paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182100
The experiments in the paper were conducted using 10,000 simulations of a 1% downsampled population of Paris.
"""

class PointNetTransfGAT(BaseGNN):
    def __init__(self, 
                in_channels: int = 0, 
                out_channels: int = 0, 
                point_net_conv_layer_structure_local_mlp: list = [], 
                point_net_conv_layer_structure_global_mlp: list = [], 
                gat_conv_layer_structure: list = [], 
                dropout: float = 0.0, 
                use_dropout: bool = False,
                use_monte_carlo_dropout: bool = False,
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                verbose: bool = True
                ):
        
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
        - use_monte_carlo_dropout (bool, optional): Whether or not to use Monte Carlo Dropout. It does make the inference slower. Note that it only makes sense to have use_monte_carlo_dropout = True if use_dropout = True.
        - predict_mode_stats (bool, optional): Whether to predict mode stats. Default is False.
        """
        # Call parent class constructor
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            use_monte_carlo_dropout=use_monte_carlo_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype
        )
        
        # Architecture-specific parameters
        self.pnc_local = point_net_conv_layer_structure_local_mlp
        self.pnc_global = point_net_conv_layer_structure_global_mlp
        self.gat_conv = gat_conv_layer_structure
        
        # Initialize dropout if needed
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize network layers
        self._init_network_layers()
        
        # Initialize weights
        self.initialize_weights()
        
        if verbose:
            print("Model initialized")
            print(self)
            
            
    def _init_network_layers(self):
        """Initialize all network layers."""
        # PointNet layers 
        # Use start + end pos
        self.point_net_conv_1 = self.create_point_net_layer(
            gat_conv_starts_with_layer=self.gat_conv[0], 
            is_first_layer=True, 
            is_last_layer=False
        )
        self.point_net_conv_2 = self.create_point_net_layer(
            gat_conv_starts_with_layer=self.gat_conv[0], 
            is_first_layer=False, 
            is_last_layer=True
        )
        
        # GAT layers
        layers_global = self.define_gat_layers()
        self.gat_graph_layers = GeoSequential('x, edge_index', layers_global)
        
        # Output layers
        self.read_out_node_predictions = nn.Linear(64, 1)
        
        # Mode stats predictor (if enabled)
        if self.predict_mode_stats:
            self.mode_stat_predictor = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                TransformerEncoder(TransformerEncoderLayer(d_model=64, nhead=4), num_layers=2),
                nn.Linear(64, 2)
            )
            self.additional_predictor = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                TransformerEncoder(TransformerEncoderLayer(d_model=64, nhead=4), num_layers=2),
                nn.Linear(64, 1)
            )


    def forward(self, data):
        """
        Forward pass for the GNN model.

        Parameters:
        - data (Data): Input data containing node features and edge indices.

        Returns:
        - torch.Tensor: Output features after passing through the model.
        """
        x = data.x.to(self.dtype)
        edge_index = data.edge_index

        # Use start + end pos
        pos1 = data.pos[:, 0, :]  # Start position
        pos2 = data.pos[:, 1, :]  # End position
        x = self.point_net_conv_1(x, pos1, edge_index)
        x = self.point_net_conv_2(x, pos2, edge_index)
        
        x = self.gat_graph_layers(x, edge_index)
        
        if self.predict_mode_stats:
            node_predictions= self.additional_predictor(x)
        
        node_predictions = self.read_out_node_predictions(x)
        
        if self.predict_mode_stats:
            mode_stats = data.mode_stats
            batch = data.batch
            pooled_node_predictions = geo_nn.global_mean_pool(x, batch)
            shape_node_preds = pooled_node_predictions.shape[0]
            shape_mode_stats = int(mode_stats.shape[0]/shape_node_preds)
            
            tensor_for_pooling = torch.repeat_interleave(torch.arange(shape_node_preds), shape_mode_stats).to(x.device)
            mode_stats_pooled = geo_nn.global_mean_pool(mode_stats, tensor_for_pooling)
            
            mode_stats_pred = self.mode_stat_predictor(mode_stats_pooled)
            mode_stats_pred = mode_stats_pred.repeat_interleave(shape_mode_stats, dim=0)
            return node_predictions, mode_stats_pred
        return node_predictions
    
    def define_gat_layers(self):
        """
        Define layers for GATConv based on configuration.

        Returns:
        - List: Layers for GATConv.
        """
        layers = []
        for idx in range(len(self.gat_conv) - 1):      
            # Transformer layer
            layers.append((TransformerConv(self.gat_conv[idx], int(self.gat_conv[idx + 1]/4), heads=4), 'x, edge_index -> x'))
            layers.append(nn.ReLU(inplace=True))
            if self.use_dropout:
                layers.append(self.dropout_layer)
        layers.append((torch_geometric.nn.GATConv(self.gat_conv[-1], 64), 'x, edge_index -> x'))
        return layers
    
    def create_point_net_layer(self, gat_conv_starts_with_layer:int, is_first_layer:bool=False, is_last_layer:bool=False):
        """
        Create PointNetConv layers with specified configurations.

        Parameters:
        - gat_conv_starts_with_layer (int): Starting layer size for GATConv.

        Returns:
        - Tuple[nn.Sequential, nn.Sequential]: Local and global MLP layers.
        """
        offset_due_to_pos = 2
        local_MLP_layers = []
        if is_first_layer:
            local_MLP_layers.append(nn.Linear(self.in_channels + offset_due_to_pos, self.pnc_local[0]))
        else:
            local_MLP_layers.append(nn.Linear(self.pnc_global[-1] + offset_due_to_pos, self.pnc_local[0]))
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
                
        if is_last_layer:
            global_MLP_layers.append(nn.Linear(self.pnc_global[- 1], gat_conv_starts_with_layer))
        else:
            global_MLP_layers.append(nn.Linear(self.pnc_global[-1], self.pnc_global[-1]))
        
        global_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            global_MLP_layers.append(self.dropout_layer)
        global_MLP = nn.Sequential(*global_MLP_layers)
        return PointNetConv(local_nn=local_MLP, global_nn=global_MLP)
    
    # WEIGHT INITIALIZION
    
    def initialize_weights(self):
        """
        Initialize model weights using Xavier and Kaiming initialization.
        """
        super().initialize_weights()  # Call parent's initialization
        for m in self.modules():
            if isinstance(m, PointNetConv):
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

    def train_model(self, 
            config: object = None, 
            loss_fct: nn.Module = None, 
            optimizer: optim.Optimizer = None, 
            train_dl: DataLoader = None, 
            valid_dl: DataLoader = None, 
            device: torch.device = None, 
            early_stopping: object = None, 
            model_save_path: str = None,
            scalers_train: dict = None,
            scalers_validation: dict = None) -> tuple:
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
        - model_save_path (str, optional): Path to save the best model.
        - scalers_train (dict, optional): x and pos scalers for training data.
        - scalers_validation (dict, optional): x and pos scalers for validation data.

        Returns:
        - tuple: Validation loss and the best epoch.
        """
        if config is None:
            raise ValueError("Config cannot be None")
        
        scaler = GradScaler()
        total_steps = config.num_epochs * len(train_dl)
        scheduler = LinearWarmupCosineDecayScheduler(initial_lr=config.lr, total_steps=total_steps)
        best_val_loss = float('inf')
        checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(config.num_epochs):
            super().train()
            optimizer.zero_grad()
            for idx, data in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                step = epoch * len(train_dl) + idx
                lr = scheduler.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    
                data = data.to(device)
                targets_node_predictions = data.y
                x_unscaled = scalers_train["x_scaler"].inverse_transform(data.x.detach().clone().cpu().numpy())

                if config.predict_mode_stats:
                    targets_mode_stats = data.mode_stats
            
                with autocast():
                    # Forward pass
                    if config.predict_mode_stats:
                        predicted, mode_stats_pred = self(data)
                        train_loss_node_predictions = loss_fct(predicted, targets_node_predictions, x_unscaled)
                        train_loss_mode_stats = loss_fct(mode_stats_pred, targets_mode_stats) # add weight here also later!
                        train_loss = train_loss_node_predictions + train_loss_mode_stats
                    else:
                        predicted = self(data)
                        train_loss = loss_fct(predicted, targets_node_predictions, x_unscaled)
        
                # Backward pass
                scaler.scale(train_loss).backward() 
                
                # Gradient clipping
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                # Do not log train loss at every iteration, as it uses CPU
                if (idx + 1) % 10 == 0:
                    if config.predict_mode_stats:
                        wandb.log({"train_loss": train_loss.item(), "epoch": epoch, "train_loss_node_predictions": train_loss_node_predictions.item(), "train_loss_mode_stats": train_loss_mode_stats.item()})
                    else:   
                        wandb.log({"train_loss": train_loss.item(), "epoch": epoch})
            
            if len(train_dl) % config.gradient_accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Validation step
            if config.predict_mode_stats:
                val_loss, r_squared, spearman_corr, pearson_corr, val_loss_node_predictions, val_loss_mode_stats = validate_model_during_training(
                    config=config,
                    model=self,
                    dataset=valid_dl,
                    loss_func=loss_fct,
                    device=device,
                    scalers_validation=scalers_validation
                )
                wandb.log({
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "lr": lr,
                    "r^2": r_squared,
                    "spearman": spearman_corr,
                    "pearson": pearson_corr,
                    "val_loss_node_predictions": val_loss_node_predictions,
                    "val_loss_mode_stats": val_loss_mode_stats
                })
            else:
                val_loss, r_squared, spearman_corr, pearson_corr = validate_model_during_training(
                    config=config,
                    model=self,
                    dataset=valid_dl,
                    loss_func=loss_fct,
                    device=device,
                    scalers_validation=scalers_validation
                )
                wandb.log({
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "lr": lr,
                    "r^2": r_squared,
                    "spearman": spearman_corr,
                    "pearson": pearson_corr
                })

            # Monte Carlo Dropout Logging
            if config.use_monte_carlo_dropout:
                data_example = next(iter(valid_dl))  # Use one batch from the validation loader
                mean_prediction, uncertainty = mc_dropout_predict(self, data_example, num_samples=50, device=device)
                if epoch % 10 == 0:
                    wandb.log({
                        "mc_dropout_uncertainty_std": np.std(uncertainty)
                    })

            print(f"epoch: {epoch}, validation loss: {val_loss}, lr: {lr}, r^2: {r_squared}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss   
                if model_save_path:         
                    torch.save(self.state_dict(), model_save_path)
                    print(f'Best model saved to {model_save_path} with validation loss: {val_loss}')
            
            # Save checkpoint
            if epoch % 20 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_loss': val_loss,
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

def validate_model_during_training(config: object, 
                                   model: nn.Module, 
                                   dataset: DataLoader, 
                                   loss_func: nn.Module, 
                                   device: torch.device,
                                   scalers_validation: dict) -> tuple:
    """
    Validate the model during training, with support for Monte Carlo Dropout and mode stats predictions.

    Parameters:
    - config (object): Configuration object with flags and parameters.
    - model (nn.Module): The GNN model.
    - dataset (DataLoader): Validation dataset loader.
    - loss_func (nn.Module): Loss function for validation.
    - device (torch.device): Device to perform validation on.
    - scalers_validation (dict): x and pos scalers for validation data.

    Returns:
    - tuple: Validation metrics including loss, R^2, Spearman, and Pearson correlations.
    """
    model.eval()
    val_loss = 0
    num_batches = 0
    actual_node_targets = []
    node_predictions = []
    mode_stats_targets = []
    mode_stats_predictions = []

    # Choose the appropriate inference mode
    with torch.inference_mode() if not config.use_monte_carlo_dropout else torch.no_grad():
        for idx, data in enumerate(dataset):
            data = data.to(device)
            targets_node_predictions = data.y
            x_unscaled = scalers_validation["x_scaler"].inverse_transform(data.x.detach().clone().cpu().numpy())
            targets_mode_stats = data.mode_stats if config.predict_mode_stats else None

            # Monte Carlo Dropout Inference
            if config.use_monte_carlo_dropout:
                mean_prediction, uncertainty = mc_dropout_predict(
                    model, data, num_samples=50, device=device
                )
                node_predicted = torch.tensor(mean_prediction).to(device)
                mode_stats_pred = None  # MC Dropout currently only affects node predictions
            else:
                # Standard Forward Pass
                if config.predict_mode_stats:
                    node_predicted, mode_stats_pred = model(data)
                else:
                    node_predicted = model(data)

            # Compute validation losses
            if config.predict_mode_stats:
                val_loss_node_predictions = loss_func(node_predicted, targets_node_predictions, x_unscaled).item()
                val_loss_mode_stats = loss_func(mode_stats_pred, targets_mode_stats).item() # add weight here also later!
                val_loss += val_loss_node_predictions + val_loss_mode_stats
                mode_stats_targets.append(targets_mode_stats)
                mode_stats_predictions.append(mode_stats_pred)
            else:
                val_loss += loss_func(node_predicted, targets_node_predictions, x_unscaled).item()

            # Collect predictions and targets
            actual_node_targets.append(targets_node_predictions)
            node_predictions.append(node_predicted)
            num_batches += 1

    # Compute overall metrics
    total_validation_loss = val_loss / num_batches if num_batches > 0 else 0
    actual_node_targets = torch.cat(actual_node_targets)
    node_predictions = torch.cat(node_predictions)
    r_squared = compute_r2_torch(preds=node_predictions, targets=actual_node_targets)
    spearman_corr, pearson_corr = compute_spearman_pearson(node_predictions, actual_node_targets)

    # Handle mode stats results if enabled
    if config.predict_mode_stats:
        mode_stats_targets = torch.cat(mode_stats_targets)
        mode_stats_predictions = torch.cat(mode_stats_predictions)
        return (
            total_validation_loss,
            r_squared,
            spearman_corr,
            pearson_corr,
            val_loss_node_predictions,
            val_loss_mode_stats,
        )
    else:
        return total_validation_loss, r_squared, spearman_corr, pearson_corr
    
    
def compute_spearman_pearson(preds: torch.Tensor, targets: torch.Tensor) -> tuple:
    """
    Compute Spearman and Pearson correlation coefficients.

    Parameters:
    - preds (torch.Tensor): Predicted values.
    - targets (torch.Tensor): Actual target values.

    Returns:
    - tuple: Spearman and Pearson correlation coefficients.
    """
    preds = preds.cpu().detach().numpy().flatten()  
    targets = targets.cpu().detach().numpy().flatten()
    spearman_corr, _ = spearmanr(preds, targets)
    pearson_corr, _ = pearsonr(preds, targets)
    return spearman_corr, pearson_corr

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


def mc_dropout_predict(model, data, num_samples: int = 50, device: torch.device = None):
    """
    Perform Monte Carlo Dropout inference to estimate uncertainty.

    Parameters:
    - model (nn.Module): The GNN model with dropout layers.
    - data (torch_geometric.data.Data): Input graph data.
    - num_samples (int): Number of stochastic forward passes.
    - device (torch.device): Device to run the model.

    Returns:
    - tuple: Mean predictions and uncertainty (variance) for each node or edge.
    """
    model = model.to(device)
    predictions = []

    model.train()  # Activate dropout layers during inference
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(data.to(device))
            if isinstance(pred, tuple):  # If multiple outputs (e.g., mode_stats)
                pred = pred[0]
            predictions.append(pred.cpu().numpy())  # Collect predictions

    # Stack predictions and calculate statistics
    predictions = np.stack(predictions, axis=0)  # Shape: (num_samples, num_predictions)
    mean_prediction = predictions.mean(axis=0)  # Mean prediction
    uncertainty = predictions.std(axis=0)       # Uncertainty (standard deviation)

    return mean_prediction, uncertainty

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
                 total_steps: int):
        """
        Linear warmup and cosine decay scheduler.

        Parameters:
        - initial_lr (float): Initial learning rate.
        - total_steps (int): Total number of steps.
        """
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        
        self.min_lr = 0.01*initial_lr
        self.warmup_steps = int(0.05*total_steps)
        self.decay_steps = total_steps - self.warmup_steps
        self.cosine_decay_rate = 0.5

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
            return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay 