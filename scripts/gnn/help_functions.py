import os
import sys
import math

import numpy as np
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

class GNN_Loss:
    """
    Custom loss function for GNN that supports weighted loss computation.
    The road with highest vol_base_case gets a weight of 1, and the rest are scaled accordingly (sample-wise).
    """
    
    def __init__(self, loss_fct, num_nodes, device, weighted=False):

        if loss_fct == 'mse':
            self.loss_fct = torch.nn.MSELoss(reduction='none' if weighted else 'mean').to(dtype=torch.float32).to(device)
        elif loss_fct == 'l1':
            self.loss_fct = torch.nn.L1Loss(reduction='none' if weighted else 'mean').to(dtype=torch.float32).to(device)
        else:
            raise ValueError(f"Loss function {loss_fct} not supported.")
        
        self.num_nodes = num_nodes
        self.device = device
        self.weighted = weighted

    def __call__(self, y_pred:Tensor, y_true:Tensor, x: np.ndarray = None) -> Tensor: # x is before normalization (unscaled)
        
        if self.weighted:

            loss = self.loss_fct(y_pred, y_true)
            weights = x[:, EdgeFeatures.VOL_BASE_CASE]

            # Normalize by the maximum value in each sample
            for i in range(weights.shape[0] // self.num_nodes):
                max_val = np.max(weights[i * self.num_nodes:(i + 1) * self.num_nodes])
                if max_val != 0:
                    weights[i * self.num_nodes:(i + 1) * self.num_nodes] /= max_val

            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            return torch.mean(loss * weights.unsqueeze(1))

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

    # Original x values
    x = np.concatenate([scalers["x_scaler"].inverse_transform(data.x) for data in dataset])  

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

    # Original x values
    x = np.concatenate([scalers["x_scaler"].inverse_transform(data.x) for data in dataset])
    
    target_tensor = torch.tensor(target_tensor, dtype=torch.float32).to(device)
    actual_difference_vol_car = torch.tensor(actual_difference_vol_car, dtype=torch.float32).to(device)

    # Compute the loss
    loss = loss_fct(actual_difference_vol_car, target_tensor, x)
    return loss.item()

def validate_model_during_training(config: object, 
                                   model: nn.Module, 
                                   dataset: DataLoader, 
                                   loss_func: nn.Module, 
                                   device: torch.device,
                                   scalers_validation: dict) -> tuple:
    """
    Validate the model during training, with support for mode stats predictions.

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

    # TODO: Maybe add as a parameter later?
    # Separate loss for mode stats
    mode_stats_loss = nn.MSELoss().to(dtype=torch.float32).to(device)

    # Choose the appropriate inference mode
    with torch.inference_mode():
        for idx, data in enumerate(dataset):
            data = data.to(device)
            targets_node_predictions = data.y
            x_unscaled = scalers_validation["x_scaler"].inverse_transform(data.x.detach().clone().cpu().numpy())
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
                val_loss_node_predictions = loss_func(node_predicted, targets_node_predictions, x_unscaled).item()
                val_loss_mode_stats = mode_stats_loss(mode_stats_pred, targets_mode_stats).item()
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

    # Choose the appropriate inference mode
    with torch.inference_mode():
        for idx, data in enumerate(dataset):
            data = data.to(device)
            targets_node_predictions_signed = data.y_signed
            targets_node_predictions_unsigned = data.y
            
            x_unscaled = scalers_validation["x_scaler"].inverse_transform(
                data.x.detach().clone().cpu().numpy())
            
            x_signed_unscaled = scalers_validation["x_signed_scaler"].inverse_transform(
                data.x_signed.detach().clone().cpu().numpy())

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
                batch_loss = (
                    loss_func(
                        predicted_signed,
                        targets_node_predictions_signed,
                        x_signed_unscaled,
                    ).item()
                    if use_signed
                    else loss_func(
                        predicted_unsigned,
                        targets_node_predictions_unsigned,
                        x_unscaled,
                    ).item()
                )

                val_loss += batch_loss

            # Collect predictions and targets
            if use_signed:
                actual_node_targets.append(targets_node_predictions_signed)
                node_predictions.append(predicted_signed)
            else:
                actual_node_targets.append(targets_node_predictions_unsigned)
                node_predictions.append(predicted_unsigned)
            num_batches += 1

    # Compute overall metrics
    total_validation_loss = val_loss / num_batches if num_batches > 0 else 0
    actual_node_targets = torch.cat(actual_node_targets)
    node_predictions = torch.cat(node_predictions)
    r_squared = compute_r2_torch(preds=node_predictions, targets=actual_node_targets)
    spearman_corr, pearson_corr = compute_spearman_pearson(
        node_predictions, actual_node_targets
    )

    # Handle mode stats results if enabled
    if config.predict_mode_stats:
        raise NotImplementedError("EIGN model does not support mode stats prediction.")
    else:
        return total_validation_loss, r_squared, spearman_corr, pearson_corr

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