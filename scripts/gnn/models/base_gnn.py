import os
import sys
from abc import ABC, abstractmethod

from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.help_functions import validate_model_during_training, LinearWarmupCosineDecayScheduler, select_target_tensor

class BaseGNN(nn.Module, ABC):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.3,
                 use_dropout: bool = False,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = False,
                 use_target_standardization: bool = False,
                 use_city_balanced_loss: bool = False): 
        """
        Base class for all GNN implementations.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.predict_mode_stats = predict_mode_stats
        self.dtype = dtype
        self.log_to_wandb = log_to_wandb
        
        # Target standardization parameters
        self.use_target_standardization = use_target_standardization
        self.target_mean = None
        self.target_std = None
        self.use_city_balanced_loss = use_city_balanced_loss

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
            scalers_validation: dict = None,
            target_normalization: bool = False) -> tuple:
        
        if config is None:
            raise ValueError("Config cannot be None")
        
        # COMPUTE TARGET STATISTICS BEFORE TRAINING
        if self.use_target_standardization:
            self.compute_target_statistics(train_dl, config, device)
        
        scaler = GradScaler()
        total_steps = config.num_epochs * len(train_dl)
        scheduler = LinearWarmupCosineDecayScheduler(initial_lr=config.lr, total_steps=total_steps)
        best_val_loss = float('inf')
        checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # TODO: Maybe add as a parameter later?
        # Separate loss for mode stats
        mode_stats_loss = nn.MSELoss().to(dtype=torch.float32).to(device)

        # Define WandB Logging Metrics
        from training.help_functions import setup_wandb_metrics
        setup_wandb_metrics(predict_mode_stats=config.predict_mode_stats)

        if config.continue_training:
            # Load checkpoint
            checkpoint = torch.load(config.base_checkpoint_path)
            
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            best_val_loss = checkpoint['best_val_loss']
            start_epoch = checkpoint['epoch'] + 1
            
            # Load target statistics when continuing training
            if self.use_target_standardization:
                if 'target_mean' in checkpoint and 'target_std' in checkpoint:
                    self.target_mean = checkpoint['target_mean'].to(device)
                    self.target_std = checkpoint['target_std'].to(device)
                    print("Target statistics loaded from checkpoint")
                else:
                    print("Target statistics not found in checkpoint, recomputing...")
                    self.compute_target_statistics(train_dl, config, device)
            
            print(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss}")
        else:
            # ✅ ADD THIS: Initialize for new training
            start_epoch = 0

        for epoch in range(start_epoch if config.continue_training else 0, config.num_epochs):
            super().train()
            optimizer.zero_grad()

            # Total loss
            epoch_train_loss = 0
            epoch_train_loss_node_predictions = 0
            epoch_train_loss_mode_stats = 0

            print(f"Starting training loop with {len(train_dl)} batches")
            for idx, data in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                step = epoch * len(train_dl) + idx
                lr = scheduler.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    
                data = data.to(device)
                
                # Select target based on configuration
                targets_node_predictions = select_target_tensor(data, config.target_type)
                
                # STANDARDIZE TARGET
                targets_node_predictions = self.standardize_target(targets_node_predictions)

                if config.predict_mode_stats:
                    targets_mode_stats = data.mode_stats
            
                with autocast():
                    # Forward pass
                    if config.predict_mode_stats:
                        predicted, mode_stats_pred = self(data)
                        # Debug prints for shape mismatch
                        print(f"DEBUG TRAIN MODE: predicted.shape = {predicted.shape}")
                        print(f"DEBUG TRAIN MODE: targets_node_predictions.shape = {targets_node_predictions.shape}")
                        print(f"DEBUG TRAIN MODE: x_unscaled.shape = {data.x.shape}")
                        print(f"DEBUG TRAIN MODE: data.batch.shape = {data.batch.shape}")
                        train_loss_node_predictions = loss_fct(predicted, targets_node_predictions, data, data.batch)
                        train_loss_mode_stats = mode_stats_loss(mode_stats_pred, targets_mode_stats)
                        train_loss = train_loss_node_predictions + train_loss_mode_stats
                    else:
                        predicted = self(data)
                        # Debug prints for shape mismatch
                        #print(f"DEBUG TRAIN: predicted.shape = {predicted.shape}")
                        #print(f"DEBUG TRAIN: targets_node_predictions.shape = {targets_node_predictions.shape}")
                        #print(f"DEBUG TRAIN: data.x.shape = {data.x.shape}")
                        #print(f"DEBUG TRAIN: data.batch.shape = {data.batch.shape}")
                        #print(f"DEBUG TRAIN: data.batch.dtype = {data.batch.dtype}")
                        #print(f"DEBUG TRAIN: About to call loss_fct with data.x type={type(data.x)}, data.x={data.x if isinstance(data.x, int) else 'Tensor'}")
                        train_loss = loss_fct(predicted, targets_node_predictions, data, data.batch)

                # Total loss
                epoch_train_loss += train_loss.item()
                if config.predict_mode_stats:
                    epoch_train_loss_node_predictions += train_loss_node_predictions.item()
                    epoch_train_loss_mode_stats += train_loss_mode_stats.item()
        
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
                if config.predict_mode_stats:
                    wandb.log({"batch_train_loss": train_loss.item(),
                               "batch_train_loss-node_predictions": train_loss_node_predictions.item(),
                               "batch_train_loss-mode_stats": train_loss_mode_stats.item(),
                               "batch_step":step})
                else:   
                    wandb.log({"batch_train_loss": train_loss.item(),
                               "batch_step":step})
            
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
                    scalers=scalers_train
                )
                # Epoch level logging
                log_dict = {
                    "val_loss": val_loss,
                    "train_loss": epoch_train_loss / len(train_dl),
                    "lr": lr,
                    "r^2": r_squared,
                    "spearman": spearman_corr,
                    "pearson": pearson_corr,
                    "train_loss-node_predictions": epoch_train_loss_node_predictions / len(train_dl),
                    "train_loss-mode_stats": epoch_train_loss_mode_stats / len(train_dl),
                    "val_loss-node_predictions": val_loss_node_predictions,
                    "val_loss-mode_stats": val_loss_mode_stats,
                    "epoch": epoch
                }
                
                wandb.log(log_dict)
            else:
                val_loss, r_squared, spearman_corr, pearson_corr = validate_model_during_training(
                    config=config,
                    model=self,
                    dataset=valid_dl,
                    loss_func=loss_fct,
                    device=device,
                    scalers=scalers_train
                )
                # Epoch level logging
                log_dict = {
                    "val_loss": val_loss,
                    "train_loss": epoch_train_loss / len(train_dl),
                    "lr": lr,
                    "r^2": r_squared,
                    "spearman": spearman_corr,
                    "pearson": pearson_corr,
                    "epoch": epoch
                }
                
                wandb.log(log_dict)

            print(f"epoch: {epoch}, validation loss: {val_loss}, lr: {lr}, r^2: {r_squared}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss   
                if model_save_path:         
                    torch.save(self.state_dict(), model_save_path)
                    print(f'Best model saved to {model_save_path} with validation loss: {val_loss}')
                    
                    # ✅ ADD THIS: Save target statistics with best model
                    if self.use_target_standardization and self.target_mean is not None:
                        stats_path = model_save_path.replace('.pt', '_target_stats.pt')
                        self.save_target_statistics(stats_path)
            
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
                
                # ✅ ADD THIS: Include target statistics in checkpoints
                if self.use_target_standardization and self.target_mean is not None:
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
        
        # ✅ ADD THIS: Save target statistics after training
        if self.use_target_standardization and model_save_path:
            stats_path = model_save_path.replace('.pt', '_target_stats.pt')
            self.save_target_statistics(stats_path)
        
        return val_loss, epoch
    
    def train_model_inductive(self, 
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
        Basic training pipeline for GNN models, can be overridden by child classes.

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
        
        # COMPUTE TARGET STATISTICS BEFORE TRAINING
        if self.use_target_standardization:
            self.compute_target_statistics(train_dl, config, device)
        
        scaler = GradScaler()
        total_steps = config.num_epochs * len(train_dl)
        scheduler = LinearWarmupCosineDecayScheduler(initial_lr=config.lr, total_steps=total_steps)
        best_val_loss = float('inf')
        checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # TODO: Maybe add as a parameter later?
        # Separate loss for mode stats
        mode_stats_loss = nn.MSELoss().to(dtype=torch.float32).to(device)

        # Define WandB Logging Metrics
        from training.help_functions import setup_wandb_metrics
        setup_wandb_metrics(predict_mode_stats=config.predict_mode_stats)

        if config.continue_training:

            # Load checkpoint
            checkpoint = torch.load(config.base_checkpoint_path)
            
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            best_val_loss = checkpoint['best_val_loss']
            start_epoch = checkpoint['epoch'] + 1
            
            # ✅ ADD THIS: Load target statistics when continuing training
            if self.use_target_standardization:
                if 'target_mean' in checkpoint and 'target_std' in checkpoint:
                    self.target_mean = checkpoint['target_mean'].to(device)
                    self.target_std = checkpoint['target_std'].to(device)
                    print("Target statistics loaded from checkpoint")
                else:
                    print("Target statistics not found in checkpoint, recomputing...")
                    self.compute_target_statistics(train_dl, config, device)
            
            print(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss}")
        else:
            # ✅ ADD THIS: Initialize for new training
            start_epoch = 0

        for epoch in range(start_epoch if config.continue_training else 0, config.num_epochs):
            super().train()
            optimizer.zero_grad()

            # Total loss
            epoch_train_loss = 0
            epoch_train_loss_node_predictions = 0
            epoch_train_loss_mode_stats = 0
            epoch_train_domain_loss = 0

            for idx, data in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                step = epoch * len(train_dl) + idx
                lr = scheduler.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    
                data = data.to(device)
                targets_node_predictions = select_target_tensor(data, config.target_type)
                
                # STANDARDIZE TARGET
                targets_node_predictions = self.standardize_target(targets_node_predictions)

                if config.predict_mode_stats:
                    targets_mode_stats = data.mode_stats

                with autocast():
                    if config.use_dann:
                        # DANN forward pass
                        predicted, domain_logits = self(data, alpha=config.domain_lambda)
                        city_labels = self.get_city_labels(data.to_data_list() if isinstance(data, Batch) else [data])
                        
                        task_loss = loss_fct(predicted, targets_node_predictions, data, data.batch)
                        domain_loss = nn.functional.cross_entropy(domain_logits, city_labels)
                        train_loss = task_loss + config.domain_lambda * domain_loss
                        epoch_train_domain_loss += domain_loss.item()
                    elif config.predict_mode_stats:
                        predicted, mode_stats_pred = self(data)
                        train_loss_node_predictions = loss_fct(predicted, targets_node_predictions, data, data.batch)
                        train_loss_mode_stats = mode_stats_loss(mode_stats_pred, targets_mode_stats)
                        train_loss = train_loss_node_predictions + train_loss_mode_stats
                    else:
                        predicted = self(data)
                        train_loss = loss_fct(predicted, targets_node_predictions, data, data.batch)

                # Total loss
                epoch_train_loss += train_loss.item()
                if config.predict_mode_stats and not config.use_dann:
                    epoch_train_loss_node_predictions += train_loss_node_predictions.item()
                    epoch_train_loss_mode_stats += train_loss_mode_stats.item()
        
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
                log_dict = {"batch_train_loss": train_loss.item(), "batch_step": step}
                if config.use_dann:
                    log_dict["batch_train_domain_loss"] = domain_loss.item()
                if config.predict_mode_stats and not config.use_dann:
                    log_dict["batch_train_loss-node_predictions"] = train_loss_node_predictions.item()
                    log_dict["batch_train_loss-mode_stats"] = train_loss_mode_stats.item()
                wandb.log(log_dict)
            
            if len(train_dl) % config.gradient_accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Validation step
            if config.use_dann:
                val_loss, r_squared, spearman_corr, pearson_corr, val_domain_loss = validate_model_during_training(
                    config=config,
                    model=self,
                    dataset=valid_dl,
                    loss_func=loss_fct,
                    device=device,
                    scalers=scalers_validation
                )
                wandb.log({
                    "val_loss": val_loss,
                    "train_loss": epoch_train_loss / len(train_dl),
                    "lr": lr,
                    "r^2": r_squared,
                    "spearman": spearman_corr,
                    "pearson": pearson_corr,
                    "val_domain_loss": val_domain_loss,
                    "train_domain_loss": epoch_train_domain_loss / len(train_dl),
                    "epoch": epoch
                })
            elif config.predict_mode_stats:
                val_loss, r_squared, spearman_corr, pearson_corr, val_loss_node_predictions, val_loss_mode_stats = validate_model_during_training(
                    config=config,
                    model=self,
                    dataset=valid_dl,
                    loss_func=loss_fct,
                    device=device,
                    scalers=scalers_validation
                )
                # Epoch level logging
                wandb.log({
                    "val_loss": val_loss,
                    "train_loss": epoch_train_loss / len(train_dl),
                    "lr": lr,
                    "r^2": r_squared,
                    "spearman": spearman_corr,
                    "pearson": pearson_corr,
                    "train_loss-node_predictions": epoch_train_loss_node_predictions / len(train_dl),
                    "train_loss-mode_stats": epoch_train_loss_mode_stats / len(train_dl),
                    "val_loss-node_predictions": val_loss_node_predictions,
                    "val_loss-mode_stats": val_loss_mode_stats,
                    "epoch": epoch
                })
            else:
                val_loss, r_squared, spearman_corr, pearson_corr = validate_model_during_training(
                    config=config,
                    model=self,
                    dataset=valid_dl,
                    loss_func=loss_fct,
                    device=device,
                    scalers=scalers_validation
                )
                # Epoch level logging
                wandb.log({
                    "val_loss": val_loss,
                    "train_loss": epoch_train_loss / len(train_dl),
                    "lr": lr,
                    "r^2": r_squared,
                    "spearman": spearman_corr,
                    "pearson": pearson_corr,
                    "epoch": epoch
                })

            print(f"epoch: {epoch}, validation loss: {val_loss}, lr: {lr}, r^2: {r_squared}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss   
                if model_save_path:         
                    torch.save(self.state_dict(), model_save_path)
                    print(f'Best model saved to {model_save_path} with validation loss: {val_loss}')
                    
                    # ✅ ADD THIS: Save target statistics with best model
                    if self.use_target_standardization and self.target_mean is not None:
                        stats_path = model_save_path.replace('.pt', '_target_stats.pt')
                        self.save_target_statistics(stats_path)
            
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
                
                # ✅ ADD THIS: Include target statistics in checkpoints
                if self.use_target_standardization and self.target_mean is not None:
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
        
        # ✅ ADD THIS: Save target statistics after training
        if self.use_target_standardization and model_save_path:
            stats_path = model_save_path.replace('.pt', '_target_stats.pt')
            self.save_target_statistics(stats_path)
        
        return val_loss, epoch
    
    def compute_target_statistics(self, train_dl: DataLoader, config: object, device: torch.device):
        """
        Compute mean and std of target variable from training data.
        """
        if not self.use_target_standardization:
            return
            
        print("Computing target statistics for standardization...")
        all_targets = []
        
        for data in tqdm(train_dl, desc="Computing target stats"):
            data = data.to(device)
            targets = select_target_tensor(data, config.target_type)
            all_targets.append(targets.cpu())
        
        # Concatenate all targets
        y_train = torch.cat(all_targets, dim=0)
        
        # Compute statistics
        self.target_mean = y_train.mean().to(device)
        self.target_std = y_train.std().clamp_min(1e-6).to(device)  # Avoid division by zero
        
        print(f"Target statistics - Mean: {self.target_mean:.4f}, Std: {self.target_std:.4f}")
    
    def standardize_target(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Standardize target values using computed statistics.
        """
        if not self.use_target_standardization or self.target_mean is None:
            return targets
        return (targets - self.target_mean) / self.target_std
    
    def inverse_standardize_target(self, standardized_targets: torch.Tensor) -> torch.Tensor:
        """
        Convert standardized targets back to original scale.
        """
        if not self.use_target_standardization or self.target_mean is None:
            return standardized_targets
        return standardized_targets * self.target_std + self.target_mean

    def save_target_statistics(self, path: str):
        """Save target statistics for later use."""
        if self.use_target_standardization and self.target_mean is not None:
            stats = {
                'target_mean': self.target_mean.cpu(),
                'target_std': self.target_std.cpu()
            }
            torch.save(stats, path)
            print(f"Target statistics saved to {path}")
    
    def load_target_statistics(self, path: str, device: torch.device):
        """Load target statistics."""
        if self.use_target_standardization:
            stats = torch.load(path, map_location=device)
            self.target_mean = stats['target_mean'].to(device)
            self.target_std = stats['target_std'].to(device)
            print(f"Target statistics loaded from {path}")