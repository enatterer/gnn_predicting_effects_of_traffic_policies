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

class BaseGNN(nn.Module, ABC):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.3,
                 use_dropout: bool = False,
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
        self.dtype = dtype
        self.log_to_wandb = log_to_wandb
        
        # Target standardization parameters
        self.use_target_standardization = use_target_standardization
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

    def compute_target_statistics(self,
                                  train_dl: DataLoader,
                                  save_path: str,
                                  config: object,
                                  device: torch.device):
        """
        Compute mean and std of target variable from training data.
        """

        print("Computing target statistics for standardization...")
        all_targets = []
        
        for data in tqdm(train_dl):
            data = data.to(device)
            targets = select_target_tensor(data, config.target_type)
            all_targets.append(targets.cpu())
        
        # Concatenate all targets
        y_train = torch.cat(all_targets, dim=0)
        
        # Compute statistics
        self.target_mean = y_train.mean().to(device)
        self.target_std = y_train.std().clamp_min(1e-6).to(device)  # Avoid division by zero
        
        print(f"Target Statistics - Mean: {self.target_mean:.4f}, Std: {self.target_std:.4f}")

        torch.save({'target_mean': self.target_mean.cpu(),
                    'target_std': self.target_std.cpu()}, save_path)
        
        print(f"Target statistics saved to {save_path}")
    
    def standardize_target(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Standardize target values using computed statistics.
        """
        if not self.use_target_standardization:
            return targets
        return (targets - self.target_mean) / self.target_std
    
    def inverse_standardize_target(self, standardized_targets: torch.Tensor) -> torch.Tensor:
        """
        Convert standardized targets back to original scale.
        """
        if not self.use_target_standardization:
            return standardized_targets
        return standardized_targets * self.target_std + self.target_mean

    def train_model(self, 
            config: object = None, 
            loss_fct: nn.Module = None, 
            optimizer: optim.Optimizer = None, 
            train_dl: DataLoader = None, 
            valid_dl: DataLoader = None, 
            device: torch.device = None, 
            early_stopping: object = None, 
            model_save_path: str = None,
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
        - scalers_validation (dict, optional): x and pos scalers used for validation (during training).

        Returns:
        - tuple: Validation loss and the best epoch.
        """
        
        if config is None:
            raise ValueError("Config cannot be None")
        
        # COMPUTE TARGET STATISTICS BEFORE TRAINING
        if self.use_target_standardization:
            self.compute_target_statistics(train_dl, model_save_path.replace('.pt', '_target_stats.pt'), config, device)
        
        scaler = GradScaler()
        total_steps = config.num_epochs * len(train_dl)
        scheduler = LinearWarmupCosineDecayScheduler(initial_lr=config.lr, total_steps=total_steps)
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
                    self.compute_target_statistics(train_dl, model_save_path.replace('.pt', '_target_stats.pt'), config, device)
            
            print(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss}")

        for epoch in range(start_epoch if config.continue_training else 0, config.num_epochs):
            super().train()
            optimizer.zero_grad()

            # Total loss
            epoch_train_loss = 0

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
                wandb.log({"batch_train_loss": train_loss.item(),
                           "batch_step":step})
            
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
                device=device,
                scalers=scalers_validation)

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
                
                # Include target statistics in checkpoints
                if self.use_target_standardization:
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