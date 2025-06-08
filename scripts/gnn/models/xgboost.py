import os
import sys

import wandb
import torch
import xgboost as xgb
import numpy as np
from torch_geometric.data import Batch, Data

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN

class XGBoostModel(BaseGNN):
    def __init__(self, 
                in_channels: int = 5,
                out_channels: int = 1,
                num_nodes: int = 31635,
                use_pos: bool = False,
                max_depth: int = 6,
                lr: float = 0.1,
                n_estimators: int = 100,
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                log_to_wandb: bool = False,
                dropout: float = 0.3, # Unused, but kept for compatibility
                use_dropout: bool = False): # Unused, but kept for compatibility
    
        # Call parent class constructor
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            log_to_wandb=log_to_wandb)
        
        # Model specific parameters
        self.use_pos = use_pos
        self.num_nodes = num_nodes
        self.max_depth = max_depth
        self.lr = lr
        self.n_estimators = n_estimators

        # Might give it some sense of Graph structure
        if self.use_pos:
            self.in_channels += 2  # x and y for mid pos

        if self.log_to_wandb:
            wandb.config.update({
                'max_depth': max_depth,
                'lr': lr,
                'n_estimators': n_estimators,
                'use_pos': use_pos,
                'in_channels': self.in_channels,
                'num_nodes': num_nodes
            }, allow_val_change=True)

        # Define XGBoost model
        self.define_layers()

    def define_layers(self):
        
        # Initialize XGBoost model with multi-output support
        self.model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.lr,
            n_estimators=self.n_estimators,
            objective='reg:squarederror',
            tree_method='hist',  # Use histogram-based algorithm for better performance
            multi_strategy='multi_output_tree',  # Enable multi-precision training
            verbosity=3,  # Set verbosity level
        )

    def forward(self, data):
        """
        Forward pass that converts PyTorch data to XGBoost format and makes predictions.
        Note: This is only used during inference. Training is handled separately.
        """
        # Unpack data
        if isinstance(data, Batch):
            datalist = data.to_data_list()
        elif isinstance(data, Data):
            datalist = [data]
        else:
            raise ValueError("Input data must be a Batch or Data object")

        # Reshape x to (batch_size, num_nodes * in_channels)
        x = [data.x.flatten() for data in datalist]
        x = torch.stack(x)

        if self.use_pos:
            flattened_pos = [data.pos[:,2,:].flatten() for data in datalist]
            flattened_pos = torch.stack(flattened_pos)
            x = torch.cat((x, flattened_pos), dim=1)

        # Convert to numpy for XGBoost
        x_np = x.cpu().numpy()
        
        # Make predictions
        predictions = self.model.predict(x_np)
        
        # Convert back to torch tensor
        return torch.tensor(predictions, dtype=self.dtype)

    def train_model(self, 
            config: object = None, # Unused, but kept for compatibility
            loss_fct: object = None, # Unused, but kept for compatibility
            optimizer: object = None, # Unused, but kept for compatibility
            train_dl: object = None, 
            valid_dl: object = None, 
            device: torch.device = None, # Unused, but kept for compatibility
            early_stopping: object = None, # Unused, but kept for compatibility
            model_save_path: str = None,
            scalers_train: dict = None, # Unused, but kept for compatibility
            scalers_validation: dict = None) -> tuple: # Unused, but kept for compatibility
        """
        Custom training method for XGBoost that converts PyTorch data to XGBoost format.
        """

        # Prepare training data
        X_train = []
        y_train = []
        
        for data in train_dl:
            # Process features
            if isinstance(data, Batch):
                datalist = data.to_data_list()
            else:
                datalist = [data]
                
            x = [d.x.flatten() for d in datalist]
            x = torch.stack(x)

            y = [d.y.flatten() for d in datalist]
            y = torch.stack(y)
            
            if self.use_pos:
                flattened_pos = [d.pos[:,2,:].flatten() for d in datalist]
                flattened_pos = torch.stack(flattened_pos)
                x = torch.cat((x, flattened_pos), dim=1)
            
            X_train.append(x.cpu().numpy())
            y_train.append(y.cpu().numpy())

        X_train = np.vstack(X_train)
        y_train = np.vstack(y_train)

        # Prepare validation data
        X_valid = []
        y_valid = []
        
        for data in valid_dl:
            if isinstance(data, Batch):
                datalist = data.to_data_list()
            else:
                datalist = [data]
                
            x = [d.x.flatten() for d in datalist]
            x = torch.stack(x)

            y = [d.y.flatten() for d in datalist]
            y = torch.stack(y)
            
            if self.use_pos:
                flattened_pos = [d.pos[:,2,:].flatten() for d in datalist]
                flattened_pos = torch.stack(flattened_pos)
                x = torch.cat((x, flattened_pos), dim=1)
            
            X_valid.append(x.cpu().numpy())
            y_valid.append(y.cpu().numpy())

        X_valid = np.vstack(X_valid)
        y_valid = np.vstack(y_valid)

        # Train XGBoost model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=True)

        # Save the model
        if model_save_path:
            self.model.save_model(model_save_path)

        # Get best validation score
        best_val_loss = self.model.best_score
        best_epoch = self.model.best_iteration

        return best_val_loss, best_epoch