import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, Subset
import torch_geometric
from torch_geometric.data import Data, Batch
from sklearn.metrics import r2_score

from torch_geometric.nn import PointNetConv
from tqdm import tqdm
import wandb
import numpy as np
import os
from torch.cuda.amp import GradScaler, autocast
import math
import torch.nn.init as init
from torch_geometric.nn import PointNetConv, GATConv, Sequential as GeoSequential


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            # You can choose a different initialization method
            init.xavier_normal_(m.weight)
            init.zeros_(m.bias)
            
class MyGnn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_layers_base_for_point_net_conv: int, hidden_layer_structure: list):
        super().__init__()
        
        # Hyperparameters 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_layers_base_for_point_net_conv
        self.hidden_layer_structure = hidden_layer_structure
        self.graph_layers = []
        layers = []
        
        # Architecture of PointNetConv
        local_MLP_1 = nn.Sequential(
            nn.Linear(in_channels, hidden_layers_base_for_point_net_conv),
            nn.ReLU(),
            nn.Linear(hidden_layers_base_for_point_net_conv, hidden_layers_base_for_point_net_conv),
        )
        global_MLP_1 = nn.Sequential(
            nn.Linear(hidden_layers_base_for_point_net_conv, int(hidden_layers_base_for_point_net_conv/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_layers_base_for_point_net_conv/2), int(hidden_layers_base_for_point_net_conv*2)),
            nn.ReLU(),
            nn.Linear(int(hidden_layers_base_for_point_net_conv*2), hidden_layers_base_for_point_net_conv)
        )
        self.pointLayer = PointNetConv(local_nn = local_MLP_1, global_nn = global_MLP_1)
        layers = define_layers(hidden_layer_structure=hidden_layer_structure, out_channels = out_channels)
        
        # Create the Sequential module with the layers
        if layers:
            self.graph_layers = GeoSequential('x, edge_index', layers)
        else:
            self.graph_layers= None
                
        self.initialize_weights()
                
        print("Model initialized")
        print(self)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(f"Initializing {m}")
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, PointNetConv):
                print(f"Initializing {m}")
                for name, param in m.local_nn.named_parameters():
                    if param.dim() > 1:  # weight parameters
                        print(f"Initializing {name} with kaiming_normal")
                        init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:  # bias parameters
                        print(f"Initializing {name} with zeros")
                        init.zeros_(param)
                for name, param in m.global_nn.named_parameters():
                    if param.dim() > 1:  # weight parameters
                        print(f"Initializing {name} with kaiming_normal")
                        init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:  # bias parameters
                        print(f"Initializing {name} with zeros")
                        init.zeros_(param)
            elif isinstance(m, torch_geometric.nn.GATConv):
                print(f"Initializing {m}")
                if hasattr(m, 'lin') and m.lin is not None:
                    init.xavier_normal_(m.lin.weight)
                    if m.lin.bias is not None:
                        init.zeros_(m.lin.bias)
                else:
                    print(f"Warning: {m} does not have lin or it is None")
                if hasattr(m, 'att_src') and m.att_src is not None:
                    init.xavier_normal_(m.att_src)
                else:
                    print(f"Warning: {m} does not have att_src or it is None")
                if hasattr(m, 'att_dst') and m.att_dst is not None:
                    init.xavier_normal_(m.att_dst)
                else:
                    print(f"Warning: {m} does not have att_dst or it is None")

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.pointLayer(x, data.pos, edge_index)
        if self.graph_layers:
            x = self.graph_layers(x, edge_index)
        return x
    
def define_layers(hidden_layer_structure: list, out_channels: int):
    layers = []
    for idx in range(len(hidden_layer_structure) - 1):
        layers.append((torch_geometric.nn.GATConv(hidden_layer_structure[idx], hidden_layer_structure[idx + 1]), 'x, edge_index -> x'))
        layers.append(torch.nn.ReLU(inplace=True))
    layers.append((torch_geometric.nn.GATConv(hidden_layer_structure[-1], out_channels), 'x, edge_index -> x'))
    return layers

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    train_loss = checkpoint['train_loss']
    print(f'Loaded checkpoint from epoch {epoch} with val_loss {val_loss} and train_loss {train_loss}')
    return model, optimizer, epoch, val_loss, train_loss

def train(model, config=None, loss_fct=None, optimizer=None, train_dl=None, valid_dl=None, device=None, early_stopping=None, accumulation_steps:int=3, compute_r_squared:bool = False, model_save_path:str=None): 
    scaler = GradScaler()
    total_steps = config.epochs * len(train_dl)
    scheduler = LinearWarmupCosineDecayScheduler(optimizer.param_groups[0]['lr'], warmup_steps=10000, total_steps=total_steps)
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        for idx, data in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{config.epochs}"):
            step = epoch * len(train_dl) + idx
            lr = scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            input_node_features, targets = data.x.to(device), data.y.to(device)
            with autocast():
                # Forward pass
                predicted = model(data.to(device))
                train_loss = loss_fct(predicted, targets)
                
            # Backward pass
            scaler.scale(train_loss).backward() 

            if (idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Do not log train loss at every iteration, as it uses CPU
            if (idx + 1) % 300 == 0:
                wandb.log({"train_loss": train_loss.item(), "epoch": epoch, "step": idx})
        
        if len(train_dl) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        val_loss, r_squared = validate_model_during_training(model, valid_dl, loss_fct, device)
        wandb.log({"test_loss": val_loss, "epoch": epoch, "lr_test": lr, "r^2": r_squared})
        print(f"epoch: {epoch}, validation loss: {val_loss}, lr: {lr}, r^2: {r_squared}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss   
            if model_save_path:         
                torch.save(model.state_dict(), model_save_path)
                print(f'Best model saved to {model_save_path} with validation loss: {val_loss}')
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    
    print("Best validation loss: ", val_loss)
    wandb.summary["val_loss"] = val_loss
    wandb.finish()
    return val_loss, epoch

def validate_model_during_training(model, valid_dl, loss_func, device):
    model.eval()
    val_loss = 0
    num_batches = 0
    
    actual_vals = []
    predictions = []
    
    with torch.inference_mode():
        for idx, data in enumerate(valid_dl):
            input_node_features, targets = data.x.to(device), data.y.to(device)
            predicted = model(data.to(device))
            actual_vals.append(targets)
            predictions.append(predicted)
            val_loss += loss_func(predicted, targets).item()
            num_batches += 1
            
    actual_vals=torch.cat(actual_vals)
    predictions = torch.cat(predictions)
    r_squared = compute_r2_torch(preds=predictions, targets=actual_vals)
    return val_loss / num_batches if num_batches > 0 else 0, r_squared


        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # f = r-a  # free inside reserved
        # print('Memory allocation after training, before validation: ')
        # print(t/1073741824)
        # print(r/1073741824)
        # print(a/1073741824)
        # print(f/1073741824)

# def validate_model(model, valid_dl, loss_func, device):
#     model.eval()
#     val_loss = 0
#     num_batches = 0
    
#     actual_vals = []
#     predictions = []
    
#     with torch.inference_mode():
#         for idx, data in enumerate(valid_dl):
#             input_node_features, targets = data.x.to(device), data.y.to(device)
#             predicted = model(data.to(device))
#             actual_vals.append(targets)
#             predictions.append(predicted)
#             val_loss += loss_func(predicted, targets).item()
#             num_batches += 1
            
#     actual_vals=torch.cat(actual_vals)
#     predictions = torch.cat(predictions)
#     r_squared = compute_r2_torch(preds=predictions, targets=actual_vals)
#     return val_loss / num_batches if num_batches > 0 else 0, r_squared, actual_vals, predictions

def compute_r2_torch(preds, targets):
    """Compute R^2 score using PyTorch."""
    mean_targets = torch.mean(targets)
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def save_checkpoint(model, optimizer, epoch, val_loss, train_loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Model checkpoint saved at epoch {epoch}')

def load_model(model_path):
    # Load the saved model checkpoint
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
    def __init__(self, initial_lr, warmup_steps, total_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.initial_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.initial_lr * cosine_decay 
        
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            # You can choose a different initialization method
            init.xavier_normal_(m.weight)
            init.zeros_(m.bias)
