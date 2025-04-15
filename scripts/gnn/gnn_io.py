import os
import json
import random
import joblib  # For saving the scaler

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from help_functions import replace_invalid_values, copy_subset

def create_dataloader(is_train, batch_size, dataset, train_ratio, is_test=False):
    dataset_length = len(dataset)
    print(f"Total dataset length: {dataset_length}")

    if is_test:
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Calculate split index for training and validation
    split_idx = int(dataset_length * train_ratio)
    
    # Calculate the maximum number of samples that fit into complete batches for training and validation
    train_samples = (split_idx // batch_size) * batch_size
    valid_samples = ((dataset_length - split_idx) // batch_size) * batch_size
    if is_train:
        indices = range(0, train_samples)
    else:
        indices = range(split_idx, split_idx + valid_samples)
    sub_dataset = Subset(dataset, indices)
    print(f"{'Training' if is_train else 'Validation'} subset length: {len(sub_dataset)}")
    return DataLoader(dataset=sub_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def split_into_subsets(dataset, train_ratio, val_ratio, test_ratio, shuffle_seed=42):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    dataset_length = len(dataset)
    print(f"Total dataset length: {dataset_length}")

    # Randomly shuffle the dataset
    random.Random(shuffle_seed).shuffle(dataset)
    
    # Calculate split indices
    train_split_idx = int(dataset_length * train_ratio)
    val_split_idx = train_split_idx + int(dataset_length * val_ratio)
    
    # Create indices for each subset
    train_indices = range(0, train_split_idx)
    val_indices = range(train_split_idx, val_split_idx)
    test_indices = range(val_split_idx, dataset_length)
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"Training subset length: {len(train_subset)}")
    print(f"Validation subset length: {len(val_subset)}")
    print(f"Test subset length: {len(test_subset)}")
    
    return train_subset, val_subset, test_subset

def split_into_subsets_with_bootstrapping(dataset, test_ratio=0.1, bootstrap_seed=0, shuffle_seed=42):
    
    dataset_length = len(dataset)
    print(f"Total dataset length: {dataset_length}")

    # Split the dataset into training and testing sets
    train_indices, test_indices = train_test_split(range(dataset_length), test_size=test_ratio, random_state=shuffle_seed)
    
    # Perform bootstrapping on the training set, OOB validation set
    rng = np.random.default_rng(seed=bootstrap_seed)
    train_indices_bootstrap = rng.choice(train_indices, size=len(train_indices), replace=True)
    oob_indices = list(set(train_indices) - set(train_indices_bootstrap))
    
    # Create subsets
    train_subset_bootstrap = Subset(dataset, train_indices_bootstrap)
    val_subset_oob = Subset(dataset, oob_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"Bootstrapping unique samples: {len(set(train_indices_bootstrap))}")
    print(f"Training subset length: {len(train_subset_bootstrap)}")
    print(f"OOB Validation subset length: {len(val_subset_oob)}")
    print(f"Test subset length: {len(test_subset)}")
    
    return train_subset_bootstrap, val_subset_oob, test_subset

def save_dataloader(dataloader, file_path):
    # Extract the dataset from the DataLoader
    dataset = dataloader.dataset
    # Save the dataset to the specified file path
    torch.save(dataset, file_path)

def save_dataloader_params(dataloader, file_path):
    params = {
        'batch_size': dataloader.batch_size,
        # 'shuffle': dataloader.shuffle,
        'collate_fn': dataloader.collate_fn.__name__  # Assuming collate_fn is a known function
    }
    with open(file_path, 'w') as f:
        json.dump(params, f)

def collate_fn(data_list):
    return Batch.from_data_list(data_list)

def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Retrieve the latest checkpoint file from the specified directory.

    Parameters:
    - checkpoint_dir (str): Directory where checkpoint files are stored.

    Returns:
    - str: Path to the latest checkpoint file if it exists, otherwise None.
    """
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer = None) -> tuple:
    """
    Load a checkpoint and restore the model and optimizer states.

    Parameters:
    - checkpoint_path (str): Path to the checkpoint file.
    - model (nn.Module): The model to load the state dict into.
    - optimizer (optim.Optimizer, optional): The optimizer to load the state dict into.

    Returns:
    - tuple: Restored model, optimizer, epoch, validation loss, and training loss.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    train_loss = checkpoint['train_loss']
    print(f'Loaded checkpoint from epoch {epoch} with val_loss {val_loss} and train_loss {train_loss}')
    return model, optimizer, epoch, val_loss, train_loss

def save_checkpoint(model: nn.Module, 
                    optimizer: optim.Optimizer, 
                    epoch: int, 
                    val_loss: float, 
                    train_loss: float, 
                    checkpoint_path: str) -> None:
    """
    Save a checkpoint of the model and optimizer states.

    Parameters:
    - model (nn.Module): The model to save.
    - optimizer (optim.Optimizer): The optimizer to save.
    - epoch (int): The current epoch.
    - val_loss (float): Validation loss at the current epoch.
    - train_loss (float): Training loss at the current epoch.
    - checkpoint_path (str): Path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Model checkpoint saved at epoch {epoch}')

def load_model(model_path: str, gnn_model_class: nn.Module) -> tuple:
    """
    Load a saved model checkpoint and initialize the model with the configuration.

    Parameters:
    - model_path (str): Path to the model checkpoint file.

    Returns:
    - tuple: Loaded model and configuration.
    """
    checkpoint = torch.load(model_path)
    
    # Extract the state dictionary and configuration
    state_dict = checkpoint['state_dict']
    config = checkpoint['config']
    
    # Initialize the model with the configuration
    model = gnn_model_class(**config)
    model.load_state_dict(state_dict)
    return model, config

# Call this function during training without the scalars and with the directory path, and during the testing with the saved scalars and without a directory path to save.
def normalize_dataset_create_scaler(dataset_input, directory_path, normalize_y, normalize_pos):
    dataset = copy_subset(dataset_input)
    dataset, x_scaler = normalize_x_values_create_scalers(dataset, directory_path)
    if normalize_pos:
        dataset, pos_scaler = normalize_positional_features_create_scaler(dataset, directory_path)
    if normalize_y:
        dataset, y_scaler = normalize_y_values_create_scaler(dataset, directory_path)
        if normalize_pos:
            return dataset, x_scaler, pos_scaler, y_scaler 
        else:
            return dataset, x_scaler, y_scaler
    else:
        if normalize_pos:
            return dataset, x_scaler, pos_scaler 
        else:
            return dataset, x_scaler

def normalize_dataset_with_given_scaler(dataset_input, x_scalar_list = None, pos_scalar=None, y_scalar=None, normalize_y=False, normalize_pos=False):
    dataset = copy_subset(dataset_input)
    dataset = normalize_x_values_given_scaler(dataset, x_scalar_list)
    if normalize_pos:
        dataset = normalize_positional_features_given_scaler(dataset, pos_scalar)
    if normalize_y:
        dataset = normalize_y_values_given_scaler(dataset, y_scalar)
    return dataset

def normalize_x_values_given_scaler(dataset, x_scaler_list):
    shape_of_x = dataset[0].x.shape[1]
    for i in range(shape_of_x):
        scaler = x_scaler_list[i]
        for data in dataset:
            data_x_dim = replace_invalid_values(data.x[:, i].reshape(-1, 1))
            normalized_x_dim = torch.tensor(scaler.transform(data_x_dim.numpy()), dtype=torch.float)
            if i == 0:
                data.normalized_x = normalized_x_dim
            else:
                data.normalized_x = torch.cat((data.normalized_x, normalized_x_dim), dim=1)
    for data in dataset:
        data.x = data.normalized_x
        del data.normalized_x
    return dataset

def normalize_positional_features_given_scaler(dataset, pos_scalar=None):
    for data in dataset:
        data.pos = torch.tensor(pos_scalar.transform(data.pos.numpy()), dtype=torch.float)
    return dataset

def normalize_y_values_given_scaler(dataset, y_scalar=None):
    for data in dataset:
        data.y = torch.tensor(y_scalar.transform(data.y.numpy()), dtype=torch.float)
    return dataset

def normalize_x_values_create_scalers(dataset, directory_path):
    shape_of_x = dataset[0].x.shape[1]
    list_of_scalers_to_save = []
    x_values = torch.cat([data.x for data in dataset], dim=0)

    for i in range(shape_of_x):
        all_node_features = replace_invalid_values(x_values[:, i].reshape(-1, 1)).numpy()
        
        scaler = StandardScaler()
        print(f"Scaler created for x values at index {i}: {scaler}")
        scaler.fit(all_node_features)
        list_of_scalers_to_save.append(scaler)

        for data in dataset:
            data_x_dim = replace_invalid_values(data.x[:, i].reshape(-1, 1))
            normalized_x_dim = torch.tensor(scaler.transform(data_x_dim.numpy()), dtype=torch.float)
            if i == 0:
                data.normalized_x = normalized_x_dim
            else:
                data.normalized_x = torch.cat((data.normalized_x, normalized_x_dim), dim=1)

    joblib.dump(list_of_scalers_to_save, (directory_path + 'x_scaler.pkl'))

    for data in dataset:
        data.x = data.normalized_x
        del data.normalized_x
    return dataset, list_of_scalers_to_save

def normalize_positional_features_create_scaler(dataset, directory_path):
    all_pos_features = torch.cat([data.pos for data in dataset], dim=0)
    all_pos_features = replace_invalid_values(all_pos_features).numpy()
    scaler = StandardScaler()
    print(f"Scaler created for pos features: {scaler}")
    scaler.fit(all_pos_features)
    joblib.dump(scaler, os.path.join(directory_path, 'pos_scaler.pkl'))
    for data in dataset:
        data.pos = torch.tensor(scaler.transform(data.pos.numpy()), dtype=torch.float)
    return dataset, scaler

def normalize_y_values_create_scaler(dataset, directory_path):
    all_y_values = torch.cat([data.y for data in dataset], dim=0).reshape(-1, 1)
    all_y_values = replace_invalid_values(all_y_values).numpy()

    scaler = RobustScaler()
    print(f"Scaler created for y values: {scaler}")
    scaler.fit(all_y_values)
    joblib.dump(scaler, os.path.join(directory_path, 'y_scaler.pkl'))

    for data in dataset:
        data.y = torch.tensor(scaler.transform(data.y.reshape(-1, 1).numpy()), dtype=torch.float)
    return dataset, scaler

def cut_dimensions(dataset, indices_of_dimensions_to_keep: list):
    dataset_with_fewer_dimensions = dataset.copy()
    for data in dataset_with_fewer_dimensions:
        if indices_of_dimensions_to_keep == [0]:
            data.x = data.x[:, 0].view(-1, 1)  # Keep only the first column and reshape
        else: 
            data.x = data.x[:, indices_of_dimensions_to_keep]
    return dataset_with_fewer_dimensions