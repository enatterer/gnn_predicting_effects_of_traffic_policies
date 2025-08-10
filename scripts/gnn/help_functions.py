import os
import sys
import math

import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.process_simulations_for_gnn import EdgeFeatures

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
                #print(f"Using edge_weights for loss weighting: shape={weights.shape}")
            else:
                # Fallback to using VOL_BASE_CASE from features
                print('WARNING: No edge weights found')
                weights = data.x[:, EdgeFeatures.VOL_BASE_CASE]
                print(f"Using VOL_BASE_CASE for loss weighting: shape={weights.shape}")
            
            if batch is not None:
                #print('batch is not None - handling multiple graphs in batch')
                # Use batch information to handle variable graph sizes
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
            else:
                print('batch is None - single graph')
                # Single graph case - weights are already normalized per graph
                normalized_weights = weights
            
            return torch.mean(loss * normalized_weights.unsqueeze(1))
        else:
            return self.loss_fct(y_pred, y_true)
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


def select_target_tensor(data, target_type: str ):
    """
    Select the appropriate target tensor based on target_type.
    
    Args:
        data: PyTorch Geometric data object
        target_type: String specifying which target to use
        
    Returns:
        Selected target tensor
    """
    #print(f"[DEBUG] select_target_tensor: target_type={target_type}")
    #print(f"[DEBUG] select_target_tensor: hasattr(data, 'y')={hasattr(data, 'y')}, data.y={data.y if hasattr(data, 'y') else 'No y attr'}")
    #print(f"[DEBUG] select_target_tensor: hasattr(data, 'y_vol_car')={hasattr(data, 'y_vol_car')}, data.y_vol_car={data.y_vol_car if hasattr(data, 'y_vol_car') else 'No y_vol_car attr'}")
    #print(f"[DEBUG] select_target_tensor: hasattr(data, 'y_vol_car_percentage')={hasattr(data, 'y_vol_car_percentage')}, data.y_vol_car_percentage={data.y_vol_car_percentage if hasattr(data, 'y_vol_car_percentage') else 'No y_vol_car_percentage attr'}")
    
    # First check if data.y exists and is not None
    if hasattr(data, 'y') and data.y is not None:
        #print(f"[DEBUG] select_target_tensor: Returning data.y with type={type(data.y)}")
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
    
def compute_baseline_of_mean_target(dataset, loss_fct, device, scalers):
    """
    Computes the baseline Mean Squared Error (MSE) for normalized y values in the dataset.

    Parameters:
    - dataset: A dataset containing normalized y values.
    - scalers: The scalers used to normalize the x and pos values.

    Returns:
    - mse_value: The baseline MSE value.
    """
    # Concatenate the normalized y values from the dataset
    y_values_normalized = np.concatenate([data.y for data in dataset])

    # Compute the mean of the normalized y values
    mean_y_normalized = np.mean(y_values_normalized)

    # Original x values - only inverse transform if scalers are available
    if scalers is not None and "x_scaler" in scalers:
        continuous_feat = [0, 1, 2, 3, 4]  # Correct indices for 5-feature tensor: VOL_BASE_CASE, CAPACITY_BASE_CASE, CAPACITY_REDUCTION, FREESPEED, LENGTH
        x = np.concatenate([scalers["x_scaler"].inverse_transform(data.x[:, continuous_feat]) for data in dataset])
    else:
        # Use features directly if no scaler available (features pre-normalized)
        x = np.concatenate([data.x.numpy() for data in dataset])

    # Convert numpy arrays to torch tensors
    y_values_normalized_tensor = torch.tensor(y_values_normalized, dtype=torch.float32).to(device)
    mean_y_normalized_tensor = torch.tensor(mean_y_normalized, dtype=torch.float32).to(device)
    
    # Create the target tensor with the same shape as y_values_normalized_tensor
    target_tensor = mean_y_normalized_tensor.expand_as(y_values_normalized_tensor)

    # Compute the MSE
    loss = loss_fct(y_values_normalized_tensor, target_tensor, x)
    return loss.item()

def compute_baseline_of_no_policies(dataset, loss_fct, device, scalers):
    """
    Computes the baseline Mean Squared Error (MSE) for normalized y values in the dataset.

    Parameters:
    - dataset: A dataset containing y values: The actual difference of the volume of cars.
    - scalers: The scalers used to normalize the x and pos values.

    Returns:
    - mse_value: The baseline MSE value.
    """
    # Concatenate the normalized y values from the dataset
    actual_difference_vol_car = np.concatenate([data.y for data in dataset])

    target_tensor = np.zeros(actual_difference_vol_car.shape) # presume no difference in vol car due to policy

    # Original x values - only inverse transform if scalers are available
    if scalers is not None and "x_scaler" in scalers:
        continuous_feat = [0, 1, 2, 3, 4]  # Correct indices for 5-feature tensor: VOL_BASE_CASE, CAPACITY_BASE_CASE, CAPACITY_REDUCTION, FREESPEED, LENGTH
        x = np.concatenate([scalers["x_scaler"].inverse_transform(data.x[:, continuous_feat]) for data in dataset])
    else:
        # Use features directly if no scaler available (features pre-normalized)
        x = np.concatenate([data.x.numpy() for data in dataset])
    
    target_tensor = torch.tensor(target_tensor, dtype=torch.float32).to(device)
    actual_difference_vol_car = torch.tensor(actual_difference_vol_car, dtype=torch.float32).to(device)

    # Compute the loss
    loss = loss_fct(actual_difference_vol_car, target_tensor, x)
    return loss.item()

def inverse_signed_log_transform(y_log):
    """
    Inverse transform from signed log space back to original scale.
    
    Args:
        y_log (torch.Tensor): Values in signed log space
        
    Returns:
        torch.Tensor: Values in original scale
    """
    # Inverse of signed_log_normalization: sign(x) * (exp(|x|) - 1)
    sign = torch.sign(y_log)
    abs_y = torch.abs(y_log)
    return sign * (torch.exp(abs_y) - 1)
    
def compute_percentage_metrics(pred_cars_diff, true_cars_diff, base_volumes, batch_indices=None, epsilon=1.0):
    """
    Compute percentage-based metrics of a batch after converting from log space.
    
    Args:
        pred_cars_diff (torch.Tensor): Predicted volume changes in cars
        true_cars_diff (torch.Tensor): True volume changes in cars  
        base_volumes (torch.Tensor): Base volumes in cars
        batch_indices (torch.Tensor): Batch indices for each node (for graph-level aggregation)
        epsilon (float): Small value to avoid division by zero
        
    Returns:
        dict: Dictionary with percentage metrics including sum_abs_pct, sum_sq_pct, and n for aggregation
    """
    # Stable denominator
    den = torch.clamp(base_volumes.abs(), min=epsilon)
    if den.dim() == 1:
        den = den.unsqueeze(-1)

    # Percent errors per node
    diff = true_cars_diff - pred_cars_diff
    ae_pct = (diff.abs() / den) * 100.0
    se_pct = ((diff / den) ** 2) * (100.0 ** 2)  # square of percent error

    # Node-weighted (global) metrics for this batch
    n = ae_pct.numel()
    mae_pct = ae_pct.mean().item()
    rmse_pct = se_pct.mean().sqrt().item()

    out = {
        "mae": mae_pct,
        "rmse": rmse_pct,
        "sum_abs_pct": ae_pct.sum().item(),
        "sum_sq_pct": se_pct.sum().item(),
        "n": n,
    }

    # Optional per-graph (node-weighted) metrics
    if batch_indices is not None:
        unique_batches = torch.unique(batch_indices)
        graph_mae_vals = []
        graph_rmse_vals = []
        graph_sizes = []

        for bid in unique_batches:
            m = (batch_indices == bid)
            if m.any():
                graph_sizes.append(int(m.sum().item()))
                graph_mae_vals.append(ae_pct[m].mean())
                graph_rmse_vals.append(se_pct[m].mean().sqrt())

        if graph_mae_vals:
            graph_mae_t = torch.stack(graph_mae_vals)
            graph_rmse_t = torch.stack(graph_rmse_vals)
            # equal-graph weighting (if you want node-weighted, weight by graph_sizes)
            out["graph_mae_mean"] = graph_mae_t.mean().item()
            out["graph_rmse_mean"] = graph_rmse_t.mean().item()

    return out

def validate_model_during_training(config: object, 
                                   model: nn.Module, 
                                   dataset: DataLoader, 
                                   loss_func: nn.Module, 
                                   device: torch.device,
                                   scalers_train: dict) -> tuple:
    """
    Validate the model during training, with support for mode stats predictions and log-space evaluation.

    Parameters:
    - config (object): Configuration object with flags and parameters.
    - model (nn.Module): The GNN model.
    - dataset (DataLoader): Validation dataset loader.
    - loss_func (nn.Module): Loss function for validation.
    - device (torch.device): Device to perform validation on.
    - scalers_validation (dict): x and pos scalers for validation data.

    Returns:
    - tuple: Validation metrics including log-space loss, percentage metrics, and correlations.
    """
    print("Starting validation...")
    model.eval()
    val_loss = 0 #for log space loss
    num_batches = 0
    actual_node_targets = [] #for log space loss
    node_predictions = [] #for log space loss
    mode_stats_targets = [] #for mode stats loss
    mode_stats_predictions = [] #for mode stats loss
    
    # For percentage metrics
    all_pred_cars_diff = []
    all_true_cars_diff = []
    all_base_volumes = []
    all_batch_indices = []

    # TODO: Maybe add as a parameter later?
    # Separate loss for mode stats
    mode_stats_loss = nn.MSELoss().to(dtype=torch.float32).to(device)
    print('len dataset', len(dataset))
    
    # Choose the appropriate inference mode
    with torch.inference_mode():
        for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="Validation", unit="batch"):
            data = data.to(device)
            
            # Import the target selection function
            targets_node_predictions = select_target_tensor(data, config.target_type)
            # Check if scalers are available (they may be None if features are pre-normalized)
            if scalers_train is not None and "x_scaler" in scalers_train:
                # Only inverse transform the continuous features that were originally normalized
                continuous_feat = [0, 1, 2, 3, 4]  # Correct indices for 5-feature tensor: VOL_BASE_CASE, CAPACITY_BASE_CASE, CAPACITY_REDUCTION, FREESPEED, LENGTH
                continuous_features = data.x[:, continuous_feat].detach().clone().cpu().numpy()
                x_unscaled = scalers_train["x_scaler"].inverse_transform(continuous_features)
                x_unscaled = torch.tensor(x_unscaled, dtype=torch.float32, device=device)
            else:
                # Features are already normalized during preprocessing, use them directly
                x_unscaled = data.x
            targets_mode_stats = data.mode_stats if config.predict_mode_stats else None

            # Standard Forward Pass
            if config.predict_mode_stats:
                node_predicted, mode_stats_pred = model(data)
            else:
                node_predicted = model(data)

            # # Example MC Dropout Prediction, if to be used later. Use with torch.no_grad().
            # mean_prediction, uncertainty = mc_dropout_predict(model, data, num_samples=50, device=device)
            # node_predicted = torch.tensor(mean_prediction).to(device)
            # mode_stats_pred = None  # MC Dropout currently only affects node predictions

            # Compute validation losses
            if config.predict_mode_stats:
                val_loss_node_predictions = loss_func(node_predicted, targets_node_predictions, data, data.batch).item()
                val_loss_mode_stats = mode_stats_loss(mode_stats_pred, targets_mode_stats).item()
                val_loss += val_loss_node_predictions + val_loss_mode_stats
                mode_stats_targets.append(targets_mode_stats)
                mode_stats_predictions.append(mode_stats_pred)
            else:
                val_loss += loss_func(node_predicted, targets_node_predictions, data, data.batch).item()
                print('val_loss', val_loss)
                
            # Collect predictions and targets for percentage metrics
            if hasattr(data, 'unscaled_vol_base') and data.unscaled_vol_base is not None:
                if config.target_normalization:
                # Convert from log space to cars
                    pred_cars_diff = inverse_signed_log_transform(node_predicted)
                    true_cars_diff = inverse_signed_log_transform(targets_node_predictions)
                    base_volumes = data.unscaled_vol_base
                else:
                    pred_cars_diff = node_predicted
                    true_cars_diff = targets_node_predictions
                    base_volumes = data.unscaled_vol_base
                    
                batch_indices = data.batch  # Get batch indices for graph-level aggregation
                
                #print(f"DEBUG: Batch {idx} - pred_cars shape: {pred_cars_diff.shape}, batch_indices shape: {batch_indices.shape}")
                #print(f"DEBUG: Batch {idx} - unique batch IDs: {torch.unique(batch_indices)}")
                
                all_pred_cars_diff.append(pred_cars_diff)
                all_true_cars_diff.append(true_cars_diff)
                all_base_volumes.append(base_volumes)
                all_batch_indices.append(batch_indices)

            # Collect predictions and targets (in log space for correlation metrics)
            actual_node_targets.append(targets_node_predictions)
            node_predictions.append(node_predicted)
            num_batches += 1
            
            # Clear batch memory
            del data, targets_node_predictions, node_predicted
            if config.predict_mode_stats:
                del targets_mode_stats, mode_stats_pred
            torch.cuda.empty_cache()
    
    print("Validation completed!")
    
    # Compute log-space metrics (with memory cleanup)
    total_validation_loss = val_loss / num_batches if num_batches > 0 else 0
    
    # Concatenate tensors for correlation metrics
    actual_node_targets = torch.cat(actual_node_targets)
    node_predictions = torch.cat(node_predictions)
    
    # Compute metrics
    r_squared = compute_r2_torch(preds=node_predictions, targets=actual_node_targets)
    spearman_corr, pearson_corr = compute_spearman_pearson(node_predictions, actual_node_targets)
    
    print(f"Target-space validation metrics: Loss={total_validation_loss:.4f}, R²={r_squared:.4f}, Spearman={spearman_corr:.4f}")
    
    # Clear large tensors to save memory
    del actual_node_targets, node_predictions
    torch.cuda.empty_cache()
    
    # Compute percentage metrics if base volumes are available
    percentage_metrics = None
    if all_pred_cars_diff:
        # Process in chunks to avoid memory explosion
        print(f"Computing percentage metrics for {len(all_pred_cars_diff)} batches...")
        
        # Initialize accumulators
        total_abs_pct_sum = 0.0
        total_sq_pct_sum = 0.0
        total_n = 0
        
        # Process each batch separately to avoid concatenating large tensors
        for i, (pred_cars_diff, true_cars_diff, base_volumes, batch_indices) in enumerate(zip(all_pred_cars_diff, all_true_cars_diff, all_base_volumes, all_batch_indices)):
            batch_metrics = compute_percentage_metrics(pred_cars_diff, true_cars_diff, base_volumes, batch_indices)
            
            total_abs_pct_sum += batch_metrics["sum_abs_pct"]
            total_sq_pct_sum += batch_metrics["sum_sq_pct"]
            total_n += batch_metrics["n"]
            
            # Clear batch tensors to save memory
            del pred_cars_diff, true_cars_diff, base_volumes, batch_indices
            torch.cuda.empty_cache()
        
        # Global (node-weighted) metrics across the whole val set
        if total_n > 0:
            percentage_metrics = {
                "mae": total_abs_pct_sum / total_n,
                "rmse": (total_sq_pct_sum / total_n) ** 0.5,
            }
            print(f"Percentage metrics: MAE={percentage_metrics['mae']:.2f}%, RMSE={percentage_metrics['rmse']:.2f}%")
        
        # Clear the lists to save memory
        del all_pred_cars_diff, all_true_cars_diff , all_base_volumes, all_batch_indices
        torch.cuda.empty_cache()
    
    # Handle mode stats results if enabled
    if config.predict_mode_stats:
        mode_stats_targets = torch.cat(mode_stats_targets)
        mode_stats_predictions = torch.cat(mode_stats_predictions)
        return (
            total_validation_loss,
            r_squared,
            spearman_corr,
            pearson_corr,
            percentage_metrics,
            val_loss_node_predictions,
            val_loss_mode_stats,
        )
    else:
        return total_validation_loss, r_squared, spearman_corr, pearson_corr, percentage_metrics

def validate_model_during_training_eign(
        config: object,
        model: nn.Module,
        dataset: DataLoader,
        loss_func: nn.Module,
        device: torch.device,
        scalers_validation: dict,
        use_signed: bool = False) -> tuple:
    
    model.eval()
    val_loss = 0
    num_batches = 0
    actual_node_targets = []
    node_predictions = []

    # For percentage metrics
    all_pred_cars_diff_signed = []
    all_pred_cars_diff_unsigned = []
    all_true_cars_diff_signed = []
    all_true_cars_diff_unsigned = []
    all_base_volumes = []
    all_batch_indices = []

    # Choose the appropriate inference mode
    with torch.inference_mode():
        for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="EIGN Validation", unit="batch"):
            data = data.to(device)
            
            targets_node_predictions_signed = data.y_signed if hasattr(data, 'y_signed') else None
            targets_node_predictions_unsigned = data.y if hasattr(data, 'y') else None
            
            # Check if scalers are available (they may be None if features are pre-normalized)
            if scalers_validation is not None and "x_scaler" in scalers_validation:
                # Only inverse transform the continuous features that were originally normalized
                continuous_feat = [0, 1, 2, 3, 4]  # Correct indices for 5-feature tensor: VOL_BASE_CASE, CAPACITY_BASE_CASE, CAPACITY_REDUCTION, FREESPEED, LENGTH
                continuous_features = data.x[:, continuous_feat].detach().clone().cpu().numpy()
                x_unscaled = scalers_validation["x_scaler"].inverse_transform(continuous_features)
                x_unscaled = torch.tensor(x_unscaled, dtype=torch.float32, device=device)
            else:
                # Features are already normalized during preprocessing, use them directly
                x_unscaled = data
            
            if scalers_validation is not None and "x_signed_scaler" in scalers_validation:
                x_signed_unscaled = scalers_validation["x_signed_scaler"].inverse_transform(
                    data.x_signed.detach().clone().cpu().numpy())
            else:
                # Use x_signed directly if no scaler available
                x_signed_unscaled = data

            # Standard Forward Pass
            if config.predict_mode_stats:
                raise NotImplementedError(
                    "EIGN model does not support mode stats prediction."
                )
            
            else:
                eign_output = model(
                    x_unsigned=(
                        data.x if hasattr(data, "x") and data.x is not None else None
                    ),
                    x_signed=(
                        data.x_signed
                        if hasattr(data, "x_signed") and data.x_signed is not None
                        else None
                    ),
                    edge_index=data.edge_index,
                    is_directed=data.edge_is_directed,
                )

                predicted_signed, predicted_unsigned = (
                    eign_output.signed,
                    eign_output.unsigned,
                )

            # Compute validation losses
            if config.predict_mode_stats:
                raise NotImplementedError(
                    "EIGN model does not support mode stats prediction."
                )
            else:
                if use_signed and x_signed_unscaled is not None:
                    batch_loss = loss_func(
                        predicted_signed,
                        targets_node_predictions_signed,
                        x_signed_unscaled,
                        data.batch
                    ).item()
                else:
                    batch_loss = loss_func(
                        predicted_unsigned,
                        targets_node_predictions_unsigned,
                        x_unscaled,
                        data.batch
                    ).item()

                val_loss += batch_loss
                print('val_loss', val_loss)
            
            if hasattr(data, 'unscaled_vol_base') and data.unscaled_vol_base is not None:
                if config.target_normalization:
                    pred_cars_diff_signed = inverse_signed_log_transform(predicted_signed)
                    pred_cars_diff_unsigned = inverse_signed_log_transform(predicted_unsigned)
                    true_cars_diff_signed = inverse_signed_log_transform(targets_node_predictions_signed)
                    true_cars_diff_unsigned = inverse_signed_log_transform(targets_node_predictions_unsigned)
                else:
                    pred_cars_diff_signed = predicted_signed
                    pred_cars_diff_unsigned = predicted_unsigned
                    true_cars_diff_signed = targets_node_predictions_signed
                    true_cars_diff_unsigned = targets_node_predictions_unsigned
                base_volumes = data.unscaled_vol_base
                batch_indices = data.batch
                
                all_pred_cars_diff_signed.append(pred_cars_diff_signed)
                all_pred_cars_diff_unsigned.append(pred_cars_diff_unsigned)
                all_true_cars_diff_signed.append(true_cars_diff_signed)
                all_true_cars_diff_unsigned.append(true_cars_diff_unsigned)
                all_base_volumes.append(base_volumes)
                all_batch_indices.append(batch_indices)
             
            # Collect predictions and targets
            if use_signed:
                actual_node_targets.append(targets_node_predictions_signed)
                node_predictions.append(predicted_signed)
            else:
                actual_node_targets.append(targets_node_predictions_unsigned)
                node_predictions.append(predicted_unsigned)
            num_batches += 1

    print("EIGN validation completed!")
    # Compute overall metrics
    total_validation_loss = val_loss / num_batches if num_batches > 0 else 0
    actual_node_targets = torch.cat(actual_node_targets)
    node_predictions = torch.cat(node_predictions)
    r_squared = compute_r2_torch(preds=node_predictions, targets=actual_node_targets)
    spearman_corr, pearson_corr = compute_spearman_pearson(
        node_predictions, actual_node_targets
    )
    print(f"EIGN validation metrics (normalized space): Loss={total_validation_loss:.4f}, R²={r_squared:.4f}, Spearman={spearman_corr:.4f}")

    # Compute percentage metrics if base volumes are available (process in chunks to save memory)
    percentage_metrics = None
    if all_pred_cars_diff_signed:
        # Process in chunks to avoid memory explosion
        print(f"Computing percentage metrics for {len(all_pred_cars_diff_signed)} batches (signed)...")
        
        # Initialize accumulators
        total_abs_pct_sum = 0.0
        total_sq_pct_sum = 0.0
        total_n = 0
        
        # Process each batch separately to avoid concatenating large tensors
        for i, (pred_cars_diff_signed, pred_cars_diff_unsigned, true_cars_diff_signed, true_cars_diff_unsigned, base_volumes, batch_indices) in enumerate(zip(all_pred_cars_diff_signed, all_pred_cars_diff_unsigned, all_true_cars_diff_signed, true_cars_diff_unsigned, all_base_volumes, all_batch_indices)):
            batch_metrics = compute_percentage_metrics(pred_cars_diff_signed, true_cars_diff_signed, base_volumes, batch_indices)
            
            total_abs_pct_sum += batch_metrics["sum_abs_pct"]
            total_sq_pct_sum += batch_metrics["sum_sq_pct"]
            total_n += batch_metrics["n"]
            
            # Clear batch tensors to save memory
            del pred_cars_diff_signed, pred_cars_diff_unsigned, true_cars_diff_signed, true_cars_diff_unsigned, base_volumes, batch_indices
            torch.cuda.empty_cache()
        
        # Global (node-weighted) metrics across the whole val set
        if total_n > 0:
            percentage_metrics = {
                "mae": total_abs_pct_sum / total_n,
                "rmse": (total_sq_pct_sum / total_n) ** 0.5,
            }
            print(f"Percentage metrics: MAE={percentage_metrics['mae']:.2f}%, RMSE={percentage_metrics['rmse']:.2f}%")
    
        # Clear the lists to save memory
        del all_pred_cars_diff_signed, all_pred_cars_diff_unsigned, all_true_cars_diff_signed, all_true_cars_diff_unsigned, all_base_volumes, all_batch_indices
        torch.cuda.empty_cache()
        
    elif all_pred_cars_diff_unsigned:
        # Process in chunks to avoid memory explosion
        print(f"Computing percentage metrics for {len(all_pred_cars_diff_unsigned)} batches (unsigned)...")
        
        # Initialize accumulators
        total_abs_pct_sum = 0.0
        total_sq_pct_sum = 0.0
        total_n = 0
        
        # Process each batch separately to avoid concatenating large tensors
        for i, (pred_cars_diff_unsigned, true_cars_diff_unsigned, base_volumes, batch_indices) in enumerate(zip(all_pred_cars_diff_unsigned, all_true_cars_diff_unsigned, all_base_volumes, all_batch_indices)):
            batch_metrics = compute_percentage_metrics(pred_cars_diff_unsigned, true_cars_diff_unsigned, base_volumes, batch_indices)
            
            total_abs_pct_sum += batch_metrics["sum_abs_pct"]
            total_sq_pct_sum += batch_metrics["sum_sq_pct"]
            total_n += batch_metrics["n"]
            
            # Clear batch tensors to save memory
            del pred_cars_diff_unsigned, true_cars_diff_unsigned, base_volumes, batch_indices
            torch.cuda.empty_cache()
        
        # Global (node-weighted) metrics across the whole val set
        if total_n > 0:
            percentage_metrics = {
                "mae": total_abs_pct_sum / total_n,
                "rmse": (total_sq_pct_sum / total_n) ** 0.5,
            }
            print(f"Percentage metrics: MAE={percentage_metrics['mae']:.2f}%, RMSE={percentage_metrics['rmse']:.2f}%")      
            
        # Clear the lists to save memory
        del all_pred_cars_diff_unsigned, all_true_cars_diff_unsigned, all_base_volumes, all_batch_indices
        torch.cuda.empty_cache()
        
    # Handle mode stats results if enabled
    if config.predict_mode_stats:
        raise NotImplementedError("EIGN model does not support mode stats prediction.")
    else:
        return total_validation_loss, r_squared, spearman_corr, pearson_corr, percentage_metrics

def compute_spearman_pearson(preds, targets, is_np=False) -> tuple:
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

def compute_r2_torch_with_mean_targets(mean_targets, preds, targets):
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - (ss_res / ss_tot)
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