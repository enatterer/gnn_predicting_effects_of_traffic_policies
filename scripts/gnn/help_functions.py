import os
import sys
import math

from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.process_simulations_for_gnn import EdgeFeatures


################################################# ↓ Custom GNN Losses ↓ #################################################

class GNN_Loss:
    """
    Custom loss function for GNN that supports weighted loss computation.
    The road with highest vol_base_case gets a weight of 1, and the rest are scaled accordingly (sample-wise).
    """
    
    def __init__(self, loss_fct, device, weighted=False, num_nodes=None):

        if loss_fct == 'mse':
            self.loss_fct = torch.nn.MSELoss(reduction='none' if weighted else 'mean').to(dtype=torch.float32).to(device)
        elif loss_fct == 'l1':
            self.loss_fct = torch.nn.L1Loss(reduction='none' if weighted else 'mean').to(dtype=torch.float32).to(device)
        else:
            raise ValueError(f"Loss function {loss_fct} not supported.")
        
        self.num_nodes = num_nodes
        self.device = device
        self.weighted = weighted

    def __call__(self, y_pred: Tensor, y_true: Tensor, data: object = None, batch: Tensor = None) -> Tensor:
        if self.weighted:
            loss = self.loss_fct(y_pred, y_true)
            
            # Use edge_weights from data object if available, otherwise fall back to VOL_BASE_CASE
            if hasattr(data, 'edge_weights') and data.edge_weights is not None:
                weights = data.edge_weights
                print(f"Using edge_weights for loss weighting: shape={weights.shape}")
            else:
                # Fallback to using VOL_BASE_CASE from features
                print('WARNING: No edge weights found')
                weights = data.x[:, EdgeFeatures.VOL_BASE_CASE]
                print(f"Using VOL_BASE_CASE for loss weighting: shape={weights.shape}")
            
            # Use batch information to handle variable graph sizes
            if batch is not None:
                print('Batch is not None - Multiple Graphs')
                unique_batch_ids = torch.unique(batch)
                normalized_weights = torch.zeros_like(weights)
                
                for batch_id in unique_batch_ids:
                    mask = (batch == batch_id)
                    batch_weights = weights[mask]
                    # Since weights are already normalized per graph, we just use them as-is
                    # But we need to ensure each graph's weights are properly scaled relative to each other
                    max_weight = torch.max(batch_weights)
                    if max_weight > 0:
                        # Use the pre-normalized weights directly (they're already [0,1] per graph)
                        normalized_weights[mask] = batch_weights
                    else:
                        normalized_weights[mask] = 1.0  # Equal weights if all zeros
            
            # Single graph case - weights are already normalized per graph
            else:
                print('Batch is None - single graph')
                normalized_weights = weights
            
            return torch.mean(loss * normalized_weights.unsqueeze(1))
        else:
            return self.loss_fct(y_pred, y_true)

        
class CityBalancedGNNLoss(nn.Module):
    """
    City-balanced MSE for link-level predictions in batched PyG graphs.

    Steps:
    1. Compute per-node squared error (collapse time/features).
    2. Aggregate to per-graph mean (each graph equally weighted).
    3. Optionally aggregate per-city mean (each city equally weighted).
    """

    def __init__(self, loss_fct='mse', device=None, weighted=False, num_nodes=None, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.loss_fct = loss_fct  # Keep for compatibility
        self.device = device  # Keep for compatibility
        self.weighted = weighted  # Keep for compatibility
        self.num_nodes = num_nodes  # Keep for compatibility
        
        print(f"Initialized CityBalancedGNNLoss with {loss_fct} loss")

    def forward(
        self,
        y_pred: torch.Tensor,           # [N, H] or [N, H, C]
        y_true: torch.Tensor,           # [N, H] or [N, H, C]
        batch: torch.Tensor = None,     # [N] node -> graph_id
        city_id_per_graph: torch.Tensor = None,  # [G] graph -> city_id
        ) -> torch.Tensor:

        # 0) Per-node error (support both MSE and L1)
        if self.loss_fct == 'l1':
            se = torch.abs(y_pred - y_true)
        else:  # Default to MSE
            se = (y_pred - y_true) ** 2
            
        while se.dim() > 1:
            se = se.mean(dim=-1)  # collapse H (/C)
            
        if batch is None:
            # Single-graph case
            return se.mean()

        # 1) Per-graph mean SE
        num_graphs = int(batch.max().item()) + 1
        device = se.device

        per_graph_sum = torch.zeros(num_graphs, device=device, dtype=se.dtype)
        per_graph_cnt = torch.zeros(num_graphs, device=device, dtype=se.dtype)

        per_graph_sum.index_add_(0, batch, se)
        per_graph_cnt.index_add_(0, batch, torch.ones_like(se))

        per_graph_mse = per_graph_sum / (per_graph_cnt + self.eps)  # [G]

        if city_id_per_graph is None:
            # Graph-balanced
            return per_graph_mse.mean()

        # 2) Per-city mean of graph MSEs
        num_cities = int(city_id_per_graph.max().item()) + 1

        sum_city = torch.zeros(num_cities, device=device, dtype=per_graph_mse.dtype)
        cnt_city = torch.zeros(num_cities, device=device, dtype=per_graph_mse.dtype)

        sum_city.index_add_(0, city_id_per_graph, per_graph_mse)
        cnt_city.index_add_(0, city_id_per_graph, torch.ones_like(per_graph_mse))

        per_city_mse = sum_city / (cnt_city + self.eps)  # [C]

        # 3) City-balanced
        return per_city_mse.mean()

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, data: object = None, batch: torch.Tensor = None) -> torch.Tensor:
        """Compatibility method to match GNN_Loss interface"""
        # Extract city information from data if available
        city_id_per_graph = None
        if data is not None and hasattr(data, 'city'):
            # Convert city names to city IDs (only when called via old interface)
            cities_in_batch = data.city if isinstance(data.city, list) else [data.city] * (batch.max().item() + 1 if batch is not None else 1)
            unique_cities = list(set(cities_in_batch))
            city_name_to_id = {city: i for i, city in enumerate(unique_cities)}
            city_ids = [city_name_to_id[city] for city in cities_in_batch]
            city_id_per_graph = torch.tensor(city_ids, device=y_pred.device, dtype=torch.long)
        
        return self.forward(y_pred, y_true, batch, city_id_per_graph)
    

################################################# ↓ Base GNN Train Helpers ↓ #################################################

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

def compute_spearman_pearson(preds, targets, is_np = False) -> tuple:
    """
    Compute Spearman and Pearson correlation coefficients.

    Parameters:
    - preds (torch.Tensor): Predicted values.
    - targets (torch.Tensor): Actual target values.

    Returns:
    - tuple: Spearman and Pearson correlation coefficients.
    """
    if not is_np:
        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

    preds = preds.flatten()
    targets = targets.flatten()
    spearman_corr, _ = spearmanr(preds, targets)
    pearson_corr, _ = pearsonr(preds, targets)
    return spearman_corr, pearson_corr

def select_target_tensor(data, target_type: str):
    """
    Select the appropriate target tensor based on target_type.
    
    Args:
        data: PyTorch Geometric data object
        target_type: String specifying which target to use
        
    Returns:
        Selected target tensor
    """
    
    # First check if data.y exists and is not None
    if hasattr(data, 'y') and data.y is not None:
        return data.y
    
    # If data.y is None, try specific target attributes
    elif target_type == "abs_vol_car" and hasattr(data, 'y_abs_vol_car') and data.y_abs_vol_car is not None:
        return data.y_abs_vol_car
    elif target_type == "abs_vol_car_percentage" and hasattr(data, 'y_abs_vol_car_percentage') and data.y_abs_vol_car_percentage is not None:
        return data.y_abs_vol_car_percentage
    elif target_type == "vol_car_signed_log" and hasattr(data, 'y_vol_car_signed_log') and data.y_vol_car_signed_log is not None:
        return data.y_vol_car_signed_log
    elif target_type == "vol_car_percentage_signed_log" and hasattr(data, 'y_vol_car_percentage_signed_log') and data.y_vol_car_percentage_signed_log is not None:
        return data.y_vol_car_percentage_signed_log
    elif target_type == "vol_car_mean_std" and hasattr(data, 'y_vol_car_mean_std') and data.y_vol_car_mean_std is not None:
        return data.y_vol_car_mean_std
    elif target_type == "vol_car_percentage_mean_std" and hasattr(data, 'y_vol_car_percentage_mean_std') and data.y_vol_car_percentage_mean_std is not None:
        return data.y_vol_car_percentage_mean_std
    elif target_type == "vol_car_min_max" and hasattr(data, 'y_vol_car_min_max') and data.y_vol_car_min_max is not None:
        return data.y_vol_car_min_max
    elif target_type == "vol_car_percentage_min_max" and hasattr(data, 'y_vol_car_percentage_min_max') and data.y_vol_car_percentage_min_max is not None:
        return data.y_vol_car_percentage_min_max
    else:
        # If no target is available, raise an error with helpful message
        available_targets = []
        for attr in dir(data):
            if attr.startswith('y_') and getattr(data, attr) is not None:
                available_targets.append(attr)
        
        error_msg = f"No valid target found for target_type='{target_type}'. "
        if available_targets:
            error_msg += f"Available targets: {available_targets}"
        else:
            error_msg += "No target attributes found in data object."
        
        raise ValueError(error_msg)

def validate_model_during_training(config: object, 
                                   model: nn.Module, 
                                   dataset: DataLoader, 
                                   loss_func: nn.Module, 
                                   device: torch.device) -> tuple:
    """
    Validate the model during training, with support for mode stats predictions and target standardization.

    Parameters:
    - config (object): Configuration object with flags and parameters.
    - model (nn.Module): The GNN model.
    - dataset (DataLoader): Validation dataset loader.
    - loss_func (nn.Module): Loss function for validation.
    - device (torch.device): Device to perform validation on.

    Returns:
    - tuple: Validation metrics including loss and correlations.
    """
    
    print("Starting validation...")
    model.eval()
    val_loss = 0
    num_batches = 0
    actual_node_targets = []
    node_predictions = []

    print('Len dataset:', len(dataset))

    with torch.inference_mode():
        for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="Validation", unit="batch"):
            
            data = data.to(device)
            targets_node_predictions = select_target_tensor(data, config.target_type)
            
            # STANDARDIZE TARGET FOR LOSS COMPUTATION
            if hasattr(model, 'target_normalization') and model.target_normalization is not None:
                targets_standardized = model.standardize_target(targets_node_predictions, data=data)
            else:
                targets_standardized = targets_node_predictions

            # Forward pass    
            node_predicted = model(data)
            
            # Compute loss on standardized targets
            val_loss += loss_func(node_predicted, targets_standardized, data, data.batch).item()
            
            # INVERSE STANDARDIZE PREDICTIONS FOR METRICS
            if hasattr(model, 'target_normalization') and model.target_normalization is not None:
                node_predicted_original = model.inverse_standardize_target(node_predicted, data=data)
            else:
                node_predicted_original = node_predicted

            # COLLECT ORIGINAL SCALE TARGETS AND PREDICTIONS FOR METRICS
            actual_node_targets.append(targets_node_predictions)  # Original scale
            node_predictions.append(node_predicted_original)      # Original scale
            num_batches += 1

            # Clean up
            del data, targets_node_predictions, node_predicted
            if hasattr(model, 'target_normalization') and model.target_normalization is not None:
                del targets_standardized, node_predicted_original
            torch.cuda.empty_cache()

    print("Validation completed!")
    print(f"Total validation loss (sum): {val_loss:.4f}")
    
    # Compute metrics on original scale
    total_validation_loss = val_loss / num_batches if num_batches > 0 else 0
    
    # Concatenate tensors for correlation metrics
    actual_node_targets = torch.cat(actual_node_targets)
    node_predictions = torch.cat(node_predictions)
    
    # Compute metrics on original scale
    r_squared = compute_r2_torch(preds=node_predictions, targets=actual_node_targets)
    spearman_corr, pearson_corr = compute_spearman_pearson(node_predictions, actual_node_targets)
    
    print(f"Original-scale validation metrics: Loss={total_validation_loss:.4f}, R²={r_squared:.4f}, Spearman={spearman_corr:.4f}")
    
    # Clear large tensors to save memory
    del actual_node_targets, node_predictions
    torch.cuda.empty_cache()

    return total_validation_loss, r_squared, spearman_corr, pearson_corr

class LinearWarmupCosineDecayScheduler:
    def __init__(self, 
                 initial_lr: float = 0.0001, 
                 total_steps: int = 1000,
                 peak_lr: float = 0.0003,
                 warmup_fraction: float = 0.15,
                 min_lr_fraction: float = 0.01,
                 cosine_decay_rate: float = 0.5):
        """
        Linear warmup and cosine decay scheduler.

        Parameters:
        - initial_lr (float): Starting learning rate.
        - peak_lr (float): Peak learning rate reached after warmup.
        - total_steps (int): Total number of steps.
        - warmup_fraction (float): Fraction of total steps for warmup (default: 0.15 = 15%).
        - min_lr_fraction (float): Fraction of peak_lr to which it converges during cosine decay.
        - cosine_decay_rate (float): The rate at which the learning rate decays after warmup.
        """
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.cosine_decay_rate = cosine_decay_rate
        self.min_lr_fraction = min_lr_fraction

        # Minimum LR is defined as a fraction of the peak learning rate
        self.min_lr = self.min_lr_fraction * self.peak_lr
        self.warmup_steps = max(1, int(warmup_fraction * total_steps))  # Ensure at least 1 step to avoid division by zero
        self.decay_steps = max(1, total_steps - self.warmup_steps)  # Ensure at least 1 step
        
    def get_lr(self, step: int) -> float:
        """
        Get the learning rate at a specific step.

        Parameters:
        - step (int): The current step.

        Returns:
        - float: Calculated learning rate.
        """
        if step < self.warmup_steps:
            # Linear interpolation from initial_lr to peak_lr during warmup
            # At step 0: returns initial_lr
            # At step warmup_steps-1: returns value slightly less than peak_lr
            return self.initial_lr + (self.peak_lr - self.initial_lr) * (step / self.warmup_steps)
        else:
            # Cosine decay phase
            # At step warmup_steps: progress = 0, cosine_decay = 1.0, returns peak_lr (smooth transition)
            progress = (step - self.warmup_steps) / self.decay_steps
            cosine_decay = self.cosine_decay_rate * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay