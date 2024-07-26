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
from torch_geometric.nn import PointNetConv, Sequential as GeoSequential


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            # You can choose a different initialization method
            init.xavier_normal_(m.weight)
            init.zeros_(m.bias)
            
class MyGnn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, point_net_conv_layer_structure_local_mlp: list, point_net_conv_layer_structure_global_mlp: list, gat_conv_layer_structure: list, dropout: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.point_net_conv_layer_structure_local_mlp = point_net_conv_layer_structure_local_mlp
        self.point_net_conv_layer_structure_global_mlp = point_net_conv_layer_structure_global_mlp
        self.gat_conv_layer_structure = gat_conv_layer_structure
        self.graph_layers = []
        
        point_net_conv_local_mlp, point_net_conv_global_conv = self.create_point_net_layer(in_channels= in_channels, dropout=dropout, 
                                                                                            local_structure = point_net_conv_layer_structure_local_mlp, 
                                                                                            global_structure = point_net_conv_layer_structure_global_mlp,
                                                                                            gat_conv_starts_with_layer= gat_conv_layer_structure[0])
        self.point_net_layer = PointNetConv(local_nn = point_net_conv_local_mlp, global_nn = point_net_conv_global_conv)

        layers = self.define_layers(hidden_layer_structure=gat_conv_layer_structure, out_channels = out_channels, dropout=dropout)
        self.graph_layers = GeoSequential('x, edge_index', layers)

        self.initialize_weights()
        print("Model initialized")
        print(self)
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.point_net_layer(x, data.pos, edge_index)
        if self.graph_layers:
            x = self.graph_layers(x, edge_index)
        return x
    
    def create_point_net_layer(self, in_channels:int, dropout:int, local_structure:list, global_structure:list, gat_conv_starts_with_layer:int):
        local_MLP_layers = []
        local_MLP_layers.append(nn.Linear(in_channels, local_structure[0]))
        local_MLP_layers.append(nn.ReLU())
        local_MLP_layers.append(nn.Dropout(p=dropout))
        for idx in range(len(local_structure)-1):
            local_MLP_layers.append(nn.Linear(local_structure[idx], local_structure[idx + 1]))
            local_MLP_layers.append(nn.ReLU())
            local_MLP_layers.append(nn.Dropout(p=dropout))
        local_MLP = nn.Sequential(local_MLP_layers)
        
        global_MLP_layers = []
        global_MLP_layers.append(nn.Linear(local_structure[-1], global_structure[0]))
        for idx in range(len(global_structure) - 1):
            global_MLP_layers.append(nn.Linear(global_structure[idx], global_structure[idx + 1]))
            global_MLP_layers.append(nn.ReLU())
            global_MLP_layers.append(nn.Dropout(p=dropout))
        global_MLP_layers.append(nn.Linear(global_structure[ - 1], gat_conv_starts_with_layer))
        global_MLP_layers.append(nn.ReLU())
        global_MLP_layers.append(nn.Dropout(p=dropout))
        global_MLP = nn.Sequential(global_MLP_layers)
        return local_MLP, global_MLP
    
    def define_layers(hidden_layer_structure: list, out_channels: int, dropout:float= 0.3):
        layers = []
        for idx in range(len(hidden_layer_structure) - 1):
            layers.append((torch_geometric.nn.GATConv(hidden_layer_structure[idx], hidden_layer_structure[idx + 1]), 'x, edge_index -> x'))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p= dropout))
        layers.append((torch_geometric.nn.GATConv(hidden_layer_structure[-1], out_channels), 'x, edge_index -> x'))
        return layers
    
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

def train(model, config=None, loss_fct=None, optimizer=None, train_dl=None, 
          valid_dl=None, device=None, early_stopping=None, accumulation_steps:int=3, 
          compute_r_squared:bool = True, model_save_path:str=None): 
    scaler = GradScaler()
    total_steps = config.epochs * len(train_dl)
    scheduler = LinearWarmupCosineDecayScheduler(optimizer.param_groups[0]['lr'], warmup_steps=30000, total_steps=total_steps)
    best_val_loss = float('inf')
    
    # Create a directory for checkpoints if it doesn't exist
    checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        
        if compute_r_squared:
            val_loss, r_squared = validate_model_during_training(model, valid_dl, loss_fct, device, compute_r_squared=True)
            wandb.log({"test_loss": val_loss, "epoch": epoch, "lr": lr, "r^2": r_squared})
            print(f"epoch: {epoch}, validation loss: {val_loss}, lr: {lr}, r^2: {r_squared}")
        else:
            val_loss, r_squared = validate_model_during_training(model=model, dataset=valid_dl, loss_func=loss_fct, device=device, compute_r_squared=False)
            wandb.log({"test_loss": val_loss, "epoch": epoch, "lr": lr})
            print(f"epoch: {epoch}, validation loss: {val_loss}, lr: {lr}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss   
            if model_save_path:         
                torch.save(model.state_dict(), model_save_path)
                print(f'Best model saved to {model_save_path} with validation loss: {val_loss}')
        
            # Save checkpoint
        if epoch % 20 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_lr': scheduler.get_lr(),
                # 'scaler_state_dict': scaler.get(),
                'best_val_loss': best_val_loss,
                'val_loss': val_loss,
                # 'r_squared': r_squared
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

def validate_model_during_training(model, dataset, loss_func, device, compute_r_squared):
    model.eval()
    val_loss = 0
    num_batches = 0
    
    actual_vals = []
    predictions = []
    
    with torch.inference_mode():
        for idx, data in enumerate(dataset):
            input_node_features, targets = data.x.to(device), data.y.to(device)
            predicted = model(data.to(device))
            
            if compute_r_squared:
                actual_vals.append(targets)
                predictions.append(predicted)
            val_loss += loss_func(predicted, targets).item()
            num_batches += 1
    
    total_validation_loss = val_loss / num_batches if num_batches > 0 else 0
    if compute_r_squared:
        actual_vals=torch.cat(actual_vals)
        predictions = torch.cat(predictions)
        r_squared = compute_r2_torch(preds=predictions, targets=actual_vals)
        return total_validation_loss, r_squared
    else: 
        return total_validation_loss

        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # f = r-a  # free inside reserved
        # print('Memory allocation after training, before validation: ')
        # print(t/1073741824)
        # print(r/1073741824)
        # print(a/1073741824)
        # print(f/1073741824)


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
        # self.decay_rate = decay_rate
        self.decay_steps = total_steps - warmup_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.initial_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            # accelerated_decay = cosine_decay * math.exp(-self.decay_rate * progress)
            return self.initial_lr * cosine_decay 
        
# def initialize_weights(self):
#     for m in self.modules():
#         if isinstance(m, nn.Linear):
#             # You can choose a different initialization method
#             init.xavier_normal_(m.weight)
#             init.zeros_(m.bias)
            
            
            
             # Architecture of PointNetConv
        # local_MLP_1 = nn.Sequential(
        #     nn.Linear(in_channels, hidden_layers_base_for_point_net_conv),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layers_base_for_point_net_conv, hidden_layers_base_for_point_net_conv),
        # )
        # global_MLP_1 = nn.Sequential(
        #     nn.Linear(hidden_layers_base_for_point_net_conv, int(hidden_layers_base_for_point_net_conv/2)),
        #     nn.ReLU(),
        #     nn.Linear(int(hidden_layers_base_for_point_net_conv/2), int(hidden_layers_base_for_point_net_conv*2)),
        #     nn.ReLU(),
        #     nn.Linear(int(hidden_layers_base_for_point_net_conv*2), hidden_layers_base_for_point_net_conv)
        # )
