import os
import sys
from abc import ABC, abstractmethod

import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.help_functions import validate_model_during_training, select_target_tensor, LinearWarmupCosineDecayScheduler
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures

class BaseGNN(nn.Module, ABC):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.3,
                 use_dropout: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = False,
                 use_target_standardization: bool = False,
                 target_normalization: str = None):
        """
        Base class for all GNN implementations.
        
        Args:
            use_target_standardization: [DEPRECATED] Use target_normalization instead
            target_normalization: Target normalization method. Options:
                - None: No normalization (absolute values)
                - "relative_to_max_traffic_vol_base_case": Normalize by max vol_base_case per graph
                - "relative_standard_scaler": Standardize with mean/std from training data
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.dtype = dtype
        self.log_to_wandb = log_to_wandb
        
        # Backward compatibility: map use_target_standardization to target_normalization
        if target_normalization is None and use_target_standardization:
            target_normalization = "relative_standard_scaler"
        
        self.target_normalization = target_normalization
        self.use_target_standardization = (target_normalization == "relative_standard_scaler")  # For backward compatibility
        
        # Statistics for standard scaler normalization
        self.target_mean = None
        self.target_std = None
        
        # For relative normalization, we need to find vol_base_case index dynamically
        # Store it when we first encounter data
        self._vol_base_case_idx = None

    @abstractmethod
    def define_layers(self):
        """
        Define layers of the model. Must be implemented by all child classes.
        """
        pass
            
    @abstractmethod
    def forward(self, data):
        """
        Forward pass of the model.
        Must be implemented by all child classes.
        """
        pass

    def initialize_weights(self):
        """
        Initialize model weights. Can be overridden by child classes.
        Call super().initialize_weights() to apply this base logic.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def compute_target_statistics(self,
                                  train_dl: DataLoader,
                                  save_path: str,
                                  config: object,
                                  device: torch.device):
        """
        Compute mean and std of target variable from training data.
        Only used for 'relative_standard_scaler' normalization.
        """
        if self.target_normalization != "relative_standard_scaler":
            print(f"Warning: compute_target_statistics called but target_normalization is '{self.target_normalization}'. Skipping.")
            return

        print("Computing target statistics for standardization...")
        all_targets = []
        
        for data in tqdm(train_dl):
            data = data.to(device)
            targets = select_target_tensor(data, config.target_type)
            all_targets.append(targets.cpu())
        
        # Concatenate all targets
        y_train = torch.cat(all_targets, dim=0)
        
        # Compute mean and std
        self.target_mean = y_train.mean().to(device)
        self.target_std = y_train.std().to(device)
        
        # Avoid division by zero
        if self.target_std.item() < 1e-8:
            print(f"Warning: target std is very small ({self.target_std.item():.2e}). Setting to 1.0")
            self.target_std = torch.tensor(1.0, device=device)
        
        print(f"Target statistics computed: mean={self.target_mean.item():.6f}, std={self.target_std.item():.6f}")
        
        # Save statistics to file
        if save_path:
            try:
                stats_dict = {
                    'target_mean': self.target_mean.cpu().item(),
                    'target_std': self.target_std.cpu().item(),
                    'target_normalization': self.target_normalization
                }
                torch.save(stats_dict, save_path)
                print(f"Target statistics saved to {save_path}")
            except Exception as e:
                print(f"Warning: Could not save target statistics to {save_path}: {e}")
    
    def standardize_target(self, targets: torch.Tensor, data=None) -> torch.Tensor:
        """
        Standardize/normalize target values based on target_normalization mode.
        
        Args:
            targets: Target tensor to normalize
            data: PyG data object (required for 'relative_to_max_traffic_vol_base_case')
        
        Returns:
            Normalized targets
        """
        if self.target_normalization is None:
            return targets
        
        elif self.target_normalization == "relative_standard_scaler":
            if self.target_mean is None or self.target_std is None:
                raise ValueError("Target statistics not computed. Call compute_target_statistics first.")
            return (targets - self.target_mean) / self.target_std
        
        elif self.target_normalization == "relative_to_max_traffic_vol_base_case":
            if data is None:
                raise ValueError("data parameter required for 'relative_to_max_traffic_vol_base_case' normalization")
            if not hasattr(data, 'unscaled_vol_base'):
                raise ValueError("data.unscaled_vol_base not found. Cannot perform relative normalization.")
            
            # Get original vol_base_case values (non-normalized)
            vol_base_case = data.unscaled_vol_base  # Shape: [num_nodes]
            
            # Compute normalization factor per graph
            # Use max(max_vol_base_case, max_abs_target_difference) to ensure normalized values stay in [-1, 1]
            if hasattr(data, 'batch') and data.batch is not None:
                # Handle batched data
                num_graphs = data.batch.max().item() + 1
                max_vol_per_graph = torch.zeros(num_graphs, device=targets.device, dtype=targets.dtype)
                max_abs_target_per_graph = torch.zeros(num_graphs, device=targets.device, dtype=targets.dtype)
                
                for graph_idx in range(num_graphs):
                    graph_mask = (data.batch == graph_idx)
                    if graph_mask.any():
                        # Get max vol_base_case for this graph
                        max_vol_per_graph[graph_idx] = vol_base_case[graph_mask].max()
                        # Get max absolute target difference for this graph
                        graph_targets = targets[graph_mask]
                        max_abs_target_per_graph[graph_idx] = graph_targets.abs().max()
                
                # Use the maximum of max_vol_base_case and max_abs_target_difference per graph
                # This ensures normalized values stay roughly in [-1, 1]
                normalization_factor = torch.maximum(max_vol_per_graph, max_abs_target_per_graph)
                
                # Expand normalization_factor to match targets shape
                normalization_factor_expanded = normalization_factor[data.batch]
                # Ensure shape matches targets (targets may be [num_nodes, 1])
                if len(targets.shape) > 1 and targets.shape[1] == 1:
                    normalization_factor_expanded = normalization_factor_expanded.unsqueeze(1)
            else:
                # Single graph (no batching)
                max_vol = vol_base_case.max()
                max_abs_target = targets.abs().max()
                normalization_factor_expanded = torch.maximum(
                    max_vol.expand_as(targets),
                    max_abs_target.expand_as(targets)
                )
            
            # Avoid division by zero
            epsilon = 1e-8
            normalization_factor_expanded = torch.clamp(normalization_factor_expanded, min=epsilon)
            
            # DEBUG: Print statistics (only first time to avoid spam)
            if not hasattr(self, '_debug_printed_relative_norm'):
                print(f"\n[DEBUG] Relative normalization statistics:")
                print(f"  Targets shape: {targets.shape}")
                print(f"  Targets range: [{targets.min().item():.2f}, {targets.max().item():.2f}]")
                print(f"  Targets mean: {targets.mean().item():.2f}, std: {targets.std().item():.2f}")
                print(f"  vol_base_case shape: {vol_base_case.shape}")
                print(f"  vol_base_case range: [{vol_base_case.min().item():.2f}, {vol_base_case.max().item():.2f}]")
                if hasattr(data, 'batch') and data.batch is not None:
                    print(f"  Number of graphs in batch: {num_graphs}")
                    print(f"  max_vol_per_graph: {max_vol_per_graph.tolist()}")
                    print(f"  max_abs_target_per_graph: {max_abs_target_per_graph.tolist()}")
                    print(f"  normalization_factor (max of both): {normalization_factor.tolist()}")
                print(f"  normalization_factor_expanded range: [{normalization_factor_expanded.min().item():.2f}, {normalization_factor_expanded.max().item():.2f}]")
                normalized = targets / normalization_factor_expanded
                print(f"  Normalized targets range: [{normalized.min().item():.4f}, {normalized.max().item():.4f}]")
                print(f"  Normalized targets mean: {normalized.mean().item():.4f}, std: {normalized.std().item():.4f}")
                self._debug_printed_relative_norm = True
            
            # Store normalization factor in data object for inverse transform
            # This ensures we use the exact same factor during inverse
            if not hasattr(data, '_target_norm_factor'):
                data._target_norm_factor = normalization_factor_expanded.clone()
            else:
                # If already exists (e.g., in batched data), ensure it matches
                # In batched data, this should already be set correctly
                pass
            
            # Normalize: target / normalization_factor
            return targets / normalization_factor_expanded
        
        else:
            raise ValueError(f"Unknown target_normalization: {self.target_normalization}")
    
    def inverse_standardize_target(self, standardized_targets: torch.Tensor, data=None) -> torch.Tensor:
        """
        Convert standardized/normalized targets back to original scale.
        
        Args:
            standardized_targets: Normalized target tensor
            data: PyG data object (required for 'relative_to_max_traffic_vol_base_case')
        
        Returns:
            Targets in original scale
        """
        if self.target_normalization is None:
            return standardized_targets
        
        elif self.target_normalization == "relative_standard_scaler":
            if self.target_mean is None or self.target_std is None:
                raise ValueError("Target statistics not computed. Cannot inverse transform.")
            return standardized_targets * self.target_std + self.target_mean
        
        elif self.target_normalization == "relative_to_max_traffic_vol_base_case":
            if data is None:
                raise ValueError("data parameter required for 'relative_to_max_traffic_vol_base_case' inverse normalization")
            if not hasattr(data, 'unscaled_vol_base'):
                raise ValueError("data.unscaled_vol_base not found. Cannot perform inverse relative normalization.")
            
            # Use the normalization factor stored during forward transform
            if hasattr(data, '_target_norm_factor'):
                normalization_factor_expanded = data._target_norm_factor
                # Ensure device and dtype match
                normalization_factor_expanded = normalization_factor_expanded.to(
                    device=standardized_targets.device,
                    dtype=standardized_targets.dtype
                )
                # Ensure shape matches
                if normalization_factor_expanded.shape != standardized_targets.shape:
                    if len(standardized_targets.shape) > 1 and standardized_targets.shape[1] == 1:
                        if len(normalization_factor_expanded.shape) == 1:
                            normalization_factor_expanded = normalization_factor_expanded.unsqueeze(1)
            else:
                # Fallback: compute from max_vol_base_case (shouldn't happen if forward pass worked correctly)
                print("WARNING: _target_norm_factor not found in data. Using max_vol_base_case as fallback.")
                vol_base_case = data.unscaled_vol_base
                if hasattr(data, 'batch') and data.batch is not None:
                    num_graphs = data.batch.max().item() + 1
                    max_vol_per_graph = torch.zeros(num_graphs, device=standardized_targets.device, dtype=standardized_targets.dtype)
                    for graph_idx in range(num_graphs):
                        graph_mask = (data.batch == graph_idx)
                        if graph_mask.any():
                            max_vol_per_graph[graph_idx] = vol_base_case[graph_mask].max()
                    normalization_factor_expanded = max_vol_per_graph[data.batch]
                    if len(standardized_targets.shape) > 1 and standardized_targets.shape[1] == 1:
                        normalization_factor_expanded = normalization_factor_expanded.unsqueeze(1)
                else:
                    normalization_factor_expanded = vol_base_case.max().expand_as(standardized_targets)
            
            # DEBUG: Print statistics (only first time to avoid spam)
            if not hasattr(self, '_debug_printed_inverse_relative_norm'):
                print(f"\n[DEBUG] Inverse relative normalization statistics:")
                print(f"  Standardized targets shape: {standardized_targets.shape}")
                print(f"  Standardized targets range: [{standardized_targets.min().item():.4f}, {standardized_targets.max().item():.4f}]")
                print(f"  normalization_factor_expanded range: [{normalization_factor_expanded.min().item():.2f}, {normalization_factor_expanded.max().item():.2f}]")
                inverse = standardized_targets * normalization_factor_expanded
                print(f"  Inverse normalized range: [{inverse.min().item():.2f}, {inverse.max().item():.2f}]")
                self._debug_printed_inverse_relative_norm = True
            
            # Inverse normalize: standardized * normalization_factor
            return standardized_targets * normalization_factor_expanded
        
        else:
            raise ValueError(f"Unknown target_normalization: {self.target_normalization}")

    def train_model(self, 
            config: object = None, 
            loss_fct: nn.Module = None, 
            optimizer: optim.Optimizer = None, 
            train_dl: DataLoader = None, 
            valid_dl: DataLoader = None, 
            device: torch.device = None, 
            early_stopping: object = None, 
            model_save_path: str = None) -> tuple:
        """
        Basic training pipeline for GNN models, can be overridden by child classes.

        Parameters:
        - config (object, optional): Configuration object containing training parameters.
        - loss_fct (nn.Module, optional): Loss function for training.
        - optimizer (optim.Optimizer, optional): Optimizer for model training.
        - train_dl (DataLoader, optional): DataLoader for training data.
        - valid_dl (DataLoader, optional): DataLoader for validation data.
        - device (torch.device, optional): Device to use for training.
        - early_stopping (object, optional): Early stopping mechanism.
        - model_save_path (str, optional): Path to save the best model.

        Returns:
        - tuple: Validation loss and the best epoch.
        """
        
        if config is None:
            raise ValueError("Config cannot be None")
        
        # COMPUTE TARGET STATISTICS BEFORE TRAINING (only for standard scaler)
        if self.target_normalization == "relative_standard_scaler":
            self.compute_target_statistics(train_dl, model_save_path.replace('.pt', '_target_stats.pt'), config, device)
        
        scaler = GradScaler()
        total_steps = config.num_epochs * len(train_dl)
        # Debug: Check config values before creating scheduler
        peak_lr_val = getattr(config, 'peak_lr', None)
        initial_lr_val = getattr(config, 'initial_lr', None)
        peak_lr_str = f"{peak_lr_val:.6f}" if peak_lr_val is not None else "NOT SET"
        initial_lr_str = f"{initial_lr_val:.6f}" if initial_lr_val is not None else "NOT SET"
        print(f"DEBUG: Before scheduler creation - config.peak_lr={peak_lr_str}, config.initial_lr={initial_lr_str}")
        
        scheduler = LinearWarmupCosineDecayScheduler(
            initial_lr=config.initial_lr,
            total_steps=total_steps,
            peak_lr=config.peak_lr,
            warmup_fraction=config.warmup_fraction,
            min_lr_fraction=config.min_lr_fraction,
            cosine_decay_rate=config.cosine_decay_rate
        )
        
        # Debug: Check scheduler values after creation
        print(f"DEBUG: After scheduler creation - scheduler.initial_lr={scheduler.initial_lr:.6f}, scheduler.peak_lr={scheduler.peak_lr:.6f}")
        
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.initial_lr
            print(f"DEBUG: Set optimizer LR to scheduler.initial_lr={scheduler.initial_lr:.6f}, scheduler.get_lr(0)={scheduler.get_lr(0):.6f}")
            print(f"DEBUG: Optimizer param_groups[0]['lr'] after setting: {optimizer.param_groups[0]['lr']:.6f}")
        
        best_val_loss = float('inf')
        checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Define WandB Logging Metrics
        from training.help_functions import setup_wandb_metrics
        setup_wandb_metrics()

        if config.continue_training:
            # Load checkpoint
            checkpoint = torch.load(config.base_checkpoint_path)
            
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # After loading checkpoint, reset LR to peak_lr (starting LR) for new training schedule
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler.initial_lr
                print(f"DEBUG: After checkpoint load, reset optimizer LR to scheduler.initial_lr={scheduler.initial_lr:.6f}")
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            best_val_loss = checkpoint['best_val_loss']
            start_epoch = checkpoint['epoch'] + 1
            
            # Load target statistics when continuing training (for standard scaler)
            if self.target_normalization == "relative_standard_scaler":
                if 'target_mean' in checkpoint and 'target_std' in checkpoint:
                    self.target_mean = checkpoint['target_mean'].to(device)
                    self.target_std = checkpoint['target_std'].to(device)
                    print("Target statistics loaded from checkpoint")
                else:
                    print("Target statistics not found in checkpoint, recomputing...")
                    self.compute_target_statistics(train_dl, model_save_path.replace('.pt', '_target_stats.pt'), config, device)
            
            # Load target_normalization if present (for backward compatibility)
            if 'target_normalization' in checkpoint:
                self.target_normalization = checkpoint['target_normalization']
                self.use_target_standardization = (self.target_normalization == "relative_standard_scaler")
                print(f"Target normalization loaded from checkpoint: {self.target_normalization}")
            
            print(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss}")

        for epoch in range(start_epoch if config.continue_training else 0, config.num_epochs):
            super().train()
            optimizer.zero_grad()

            # Total loss
            epoch_train_loss = 0
            epoch_train_loss_node_predictions = 0
            epoch_train_loss_mode_stats = 0
            epoch_train_domain_loss = 0
            
            # Capture learning rate at the START of the epoch (first batch)
            epoch_start_step = epoch * len(train_dl)
            epoch_start_lr = scheduler.get_lr(epoch_start_step)

            for idx, data in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                step = epoch * len(train_dl) + idx
                lr = scheduler.get_lr(step)
                if step == 0:
                    print(f"DEBUG: Step 0 - scheduler.get_lr(0)={lr:.6f}, optimizer.param_groups[0]['lr'] before update={optimizer.param_groups[0]['lr']:.6f}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                if step == 0:
                    print(f"DEBUG: Step 0 - optimizer.param_groups[0]['lr'] after update={optimizer.param_groups[0]['lr']:.6f}")
                    
                data = data.to(device)
                
                # Select target based on configuration
                targets_node_predictions = select_target_tensor(data, config.target_type)
                
                # STANDARDIZE TARGET (pass data for relative normalization)
                targets_node_predictions = self.standardize_target(targets_node_predictions, data=data)
            
                with autocast():
                    # Forward pass
                    predicted = self(data)
                    train_loss = loss_fct(predicted, targets_node_predictions, data, data.batch)

                # Total loss
                epoch_train_loss += train_loss.item()
        
                # Backward pass
                scaler.scale(train_loss).backward() 
                
                # Gradient clipping
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                # Batch level logging
                wandb.log({
                    "batch_train_loss": train_loss.item(),
                    "batch_step": step,
                })
            
            if len(train_dl) % config.gradient_accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Validation step
            val_loss, r_squared, spearman_corr, pearson_corr = validate_model_during_training(
                config=config,
                model=self,
                dataset=valid_dl,
                loss_func=loss_fct,
                device=device)

            # Epoch level logging
            wandb.log({
                "val_loss": val_loss,
                "train_loss": epoch_train_loss / len(train_dl),
                "lr": epoch_start_lr,
                "r^2": r_squared,
                "spearman": spearman_corr,
                "pearson": pearson_corr,
                "epoch": epoch
            })

            print(f"epoch: {epoch}, validation loss: {val_loss}, lr: {epoch_start_lr} (start), lr_end: {lr:.6f}, r^2: {r_squared}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss   
                if model_save_path:         
                    torch.save(self.state_dict(), model_save_path)
                    print(f'Best model saved to {model_save_path} with validation loss: {val_loss}')
            
            # Save checkpoint
            if epoch % 20 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_loss': val_loss,
                }
                
                # Include target normalization info and statistics in checkpoints
                checkpoint_dict['target_normalization'] = self.target_normalization
                if self.target_normalization == "relative_standard_scaler" and self.target_mean is not None:
                    checkpoint_dict['target_mean'] = self.target_mean.cpu()
                    checkpoint_dict['target_std'] = self.target_std.cpu()
                
                torch.save(checkpoint_dict, checkpoint_path)
                print(f'Checkpoint saved to {checkpoint_path}')
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
        
        print("Best validation loss: ", best_val_loss)
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()
        
        return val_loss, epoch