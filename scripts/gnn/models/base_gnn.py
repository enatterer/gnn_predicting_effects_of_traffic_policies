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

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.help_functions import validate_model_during_training, LinearWarmupCosineDecayScheduler

class BaseGNN(nn.Module, ABC):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.3,
                 use_dropout: bool = False,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = False):
        """
        Base class for all GNN implementations.
        Core parameters are defined here, additional parameters can be added in child classes.
        
        Child classes must call super().__init__() in their __init__ method. Followed by self.define_layers() and self.initialize_weights().
        See the PointNetTransfGAT class for an example.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.predict_mode_stats = predict_mode_stats
        self.dtype = dtype
        
        # Log specific model kwargs to wandb (during training runs)
        self.log_to_wandb = log_to_wandb
        
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
            
            print(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss}")

        for epoch in range(start_epoch if config.continue_training else 0, config.num_epochs):
            super().train()
            optimizer.zero_grad()

            # Total loss
            epoch_train_loss = 0
            epoch_train_loss_node_predictions = 0
            epoch_train_loss_mode_stats = 0

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
                        train_loss_mode_stats = mode_stats_loss(mode_stats_pred, targets_mode_stats)
                        train_loss = train_loss_node_predictions + train_loss_mode_stats
                    else:
                        predicted = self(data)
                        train_loss = loss_fct(predicted, targets_node_predictions, x_unscaled)

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
                    scalers_validation=scalers_validation
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
                    "epoch":epoch})
            else:
                val_loss, r_squared, spearman_corr, pearson_corr = validate_model_during_training(
                    config=config,
                    model=self,
                    dataset=valid_dl,
                    loss_func=loss_fct,
                    device=device,
                    scalers_validation=scalers_validation
                )
                # Epoch level logging
                wandb.log({
                    "val_loss": val_loss,
                    "train_loss": epoch_train_loss / len(train_dl),
                    "lr": lr,
                    "r^2": r_squared,
                    "spearman": spearman_corr,
                    "pearson": pearson_corr,
                    "epoch":epoch})

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
                    'scaler_state_dict': scaler.state_dict(),
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