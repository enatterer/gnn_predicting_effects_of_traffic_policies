import os
import sys
import json
import math
import random
from collections import Counter
from functools import lru_cache
import os
from pathlib import Path

import math
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset, Dataset, DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.process_simulations_for_gnn import EdgeFeatures


################################################# ↓ Data IO ↓ #################################################

# On the fly loading with caching
class GraphDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    @lru_cache(maxsize=5000)
    def get(self, idx):
        data = torch.load(self.paths[idx])
        return data
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slice indexing by creating a list of indices
            start, stop, step = idx.indices(len(self))
            return [self.get(i) for i in range(start, stop, step)]
        else:
            return self.get(idx)

# Use exclusive provided sets if available.
# Otherwise split into train, validation, and test sets from entire train_data.
def load_data_and_split_into_subsets(train_data, val_data, test_data,
                                     train_ratio, val_ratio, test_ratio=0, seed=42):
    
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    dataset_length = len(train_data['path']) + (len(val_data['path']) if val_data is not None else 0) + (len(test_data['path']) if test_data is not None else 0)
    print(f"Total dataset length: {dataset_length}")

    if test_data is None and val_data is None:
        paths = train_data['path']
        labels = [f"{city}_{policy_region}" for city, policy_region in zip(train_data['city'], train_data['policy_region'])]

        indices = list(range(len(paths)))

        try:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=test_ratio,
                random_state=seed,
                stratify=labels
            )
        except ValueError:
            print("Warning: Stratified train/test split not possible (class with <2 samples). Falling back to unstratified split.")
            train_indices, test_indices = train_test_split(
                indices,
                test_size=test_ratio,
                random_state=seed,
                stratify=None
            )

        train_labels = [labels[i] for i in train_indices]

        try:
            train_indices, val_indices = train_test_split(
                train_indices,
                test_size=val_ratio/(train_ratio + val_ratio),
                random_state=seed,
                stratify=train_labels
            )
        except ValueError:
            print("Warning: Stratified train/val split not possible (class with <2 samples). Falling back to unstratified split.")
            train_indices, val_indices = train_test_split(
                train_indices,
                test_size=val_ratio/(train_ratio + val_ratio),
                random_state=seed,
                stratify=None
            )

    else:
        
        train_indices = list(range(len(train_data['path'])))
        val_indices = list(range(len(train_data['path']), len(train_data['path']) + len(val_data['path'])))
        test_indices = list(range(len(train_data['path']) + len(val_data['path']), len(train_data['path']) + len(val_data['path']) + len(test_data['path'])))
        
        paths = train_data['path'] + val_data['path'] + test_data['path']
        train_labels = [f"{city}_{policy_region}" for city, policy_region in zip(train_data['city'], train_data['policy_region'])]
        val_labels = [f"{city}_{policy_region}" for city, policy_region in zip(val_data['city'], val_data['policy_region'])]
        test_labels = [f"{city}_{policy_region}" for city, policy_region in zip(test_data['city'], test_data['policy_region'])]
        labels = train_labels + val_labels + test_labels
        
    # Create the dataset
    dataset = GraphDataset(paths, labels)
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    def _log_split_details(name, indices, subset_labels):
        subset_cities = [dataset.labels[i].split('_', 1)[0] for i in indices]
        city_counts = Counter(subset_cities)
        print(f"[DEBUG] {name} subset label counts (total {len(indices)}):")
        for city, count in sorted(city_counts.items()):
            print(f"  {city}: {count}")
        return set(subset_cities)
    
    print(f"Training subset length: {len(train_subset)}")
    print(f"Validation subset length: {len(val_subset)}")
    print(f"Test subset length: {len(test_subset)}")
    train_cities_set = _log_split_details("Train", train_indices, train_labels if 'train_labels' in locals() else labels)
    val_cities_set = _log_split_details("Validation", val_indices, labels)
    test_cities_set = _log_split_details("Test", test_indices, labels)
    
    # Verify no data leakage when val_data and test_data are provided
    # For INDUCTIVE learning: cities must be different (strict separation)
    # For TRANSDUCTIVE learning: same cities are allowed, but graphs must be different
    if val_data is not None or test_data is not None:
        # Check if this is inductive (different cities) or transductive (same cities)
        # Only compare non-empty sets - empty test_data shouldn't affect the check
        has_val_data = val_data is not None and len(val_cities_set) > 0
        has_test_data = test_data is not None and len(test_cities_set) > 0
        
        # It's inductive if any non-empty city sets differ
        is_inductive = False
        if has_val_data and train_cities_set != val_cities_set:
            is_inductive = True
        if has_test_data and train_cities_set != test_cities_set:
            is_inductive = True
        if has_val_data and has_test_data and val_cities_set != test_cities_set:
            is_inductive = True
        
        if is_inductive:
            # INDUCTIVE: Cities must be completely separate
            leakage_val = train_cities_set & val_cities_set
            leakage_test = train_cities_set & test_cities_set
            leakage_val_test = val_cities_set & test_cities_set
            
            if leakage_val:
                raise ValueError(f"DATA LEAKAGE DETECTED in load_data_and_split_into_subsets: Training cities {leakage_val} appear in validation set!")
            if leakage_test:
                raise ValueError(f"DATA LEAKAGE DETECTED in load_data_and_split_into_subsets: Training cities {leakage_test} appear in test set!")
            if leakage_val_test:
                raise ValueError(f"DATA LEAKAGE DETECTED in load_data_and_split_into_subsets: Validation cities {leakage_val_test} appear in test set!")
            
            print(f"[VERIFICATION] ✓ No data leakage (INDUCTIVE): Training cities ({len(train_cities_set)}) are separate from validation ({len(val_cities_set)}) and test ({len(test_cities_set)}) cities")
        else:
            # TRANSDUCTIVE: Same cities are allowed, but verify graphs are different
            # Check that train/val/test indices don't overlap (they shouldn't based on how they're created)
            train_indices_set = set(train_indices)
            val_indices_set = set(val_indices)
            test_indices_set = set(test_indices)
            
            graph_leakage_val = train_indices_set & val_indices_set
            graph_leakage_test = train_indices_set & test_indices_set
            graph_leakage_val_test = val_indices_set & test_indices_set
            
            if graph_leakage_val:
                raise ValueError(f"DATA LEAKAGE DETECTED in load_data_and_split_into_subsets: {len(graph_leakage_val)} training graph(s) appear in validation set!")
            if graph_leakage_test:
                raise ValueError(f"DATA LEAKAGE DETECTED in load_data_and_split_into_subsets: {len(graph_leakage_test)} training graph(s) appear in test set!")
            if graph_leakage_val_test:
                raise ValueError(f"DATA LEAKAGE DETECTED in load_data_and_split_into_subsets: {len(graph_leakage_val_test)} validation graph(s) appear in test set!")
            
            print(f"[VERIFICATION] ✓ No data leakage (TRANSDUCTIVE): Same cities allowed, but {len(train_indices_set)} training, {len(val_indices_set)} validation, and {len(test_indices_set)} test graphs are non-overlapping")
    
    return dataset, train_subset, val_subset, test_subset

def save_dataloader(dataloader, file_path):
    # Extract the dataset from the DataLoader
    dataset = dataloader.dataset
    # Save the dataset to the specified file path
    torch.save(dataset, file_path)

def save_dataloader_params(dataloader, file_path):
    params = {
        'batch_size': dataloader.batch_size,
        # 'shuffle': dataloader.shuffle,
        'collate_fn': dataloader.collate_fn.__name__ if hasattr(dataloader.collate_fn, '__name__') else str(type(dataloader.collate_fn))
    }
    with open(file_path, 'w') as f:
        json.dump(params, f)

def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== Model Information ===")
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("=" * 30)


################################################# ↓ DATA Augmentation + Collation ↓ #################################################

def rotate_pos(pos, theta):
    """Rotate 2D positions by theta radians, with 30% chance of random reflection."""
    
    if pos is None:
        return pos
    
    # Handle different position tensor shapes
    if len(pos.shape) == 3:
        # [N, 3, 2] format - apply SAME rotation to all 3 positions per node
        pos_rotated = pos.clone()
        
        if pos.shape[2] >= 2:
            rot_matrix = torch.tensor([
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta),  math.cos(theta)]
            ], dtype=pos.dtype, device=pos.device)
            
            # Apply the SAME rotation to all 3 positions for each node
            for i in range(pos.shape[1]):  # Loop through the 3 positions
                pos_xy = pos[:, i, :2] @ rot_matrix.T
                pos_rotated[:, i, :2] = pos_xy
                if pos.shape[2] > 2:
                    pos_rotated[:, i, 2:] = pos[:, i, 2:]
            
            # Apply reflection ONCE to all positions (with 30% probability)
            if random.random() < 0.3:
                reflection_type = random.choice(['horizontal', 'vertical', 'diagonal1', 'diagonal2'])
                for i in range(pos.shape[1]):
                    if reflection_type == 'horizontal':
                        pos_rotated[:, i, 1] = -pos_rotated[:, i, 1]
                    elif reflection_type == 'vertical':
                        pos_rotated[:, i, 0] = -pos_rotated[:, i, 0]
                    elif reflection_type == 'diagonal1':  # y = x
                        temp = pos_rotated[:, i, 0].clone()
                        pos_rotated[:, i, 0] = pos_rotated[:, i, 1]
                        pos_rotated[:, i, 1] = temp
                    elif reflection_type == 'diagonal2':  # y = -x
                        temp = pos_rotated[:, i, 0].clone()
                        pos_rotated[:, i, 0] = -pos_rotated[:, i, 1]
                        pos_rotated[:, i, 1] = -temp
        
        return pos_rotated
    
    elif len(pos.shape) == 2 and pos.shape[1] >= 2:
        
        # [N, 2] format - original logic
        rot_matrix = torch.tensor([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)]
        ], dtype=pos.dtype, device=pos.device)
        pos_xy = pos[:, :2] @ rot_matrix.T

        # With 30% probability, apply a random reflection
        if random.random() < 0.3:
            reflection_type = random.choice(['horizontal', 'vertical', 'diagonal1', 'diagonal2'])
            if reflection_type == 'horizontal':
                pos_xy[:, 1] = -pos_xy[:, 1]
            elif reflection_type == 'vertical':
                pos_xy[:, 0] = -pos_xy[:, 0]
            elif reflection_type == 'diagonal1':  # y = x
                pos_xy = pos_xy.flip(1)
            elif reflection_type == 'diagonal2':  # y = -x
                pos_xy = -pos_xy.flip(1)

        if pos.shape[1] > 2:
            pos_aug = torch.cat([pos_xy, pos[:, 2:]], dim=1)
        else:
            pos_aug = pos_xy
        
        return pos_aug
    
    else:
        # Unsupported shape, return as-is
        return pos
    
def apply_simple_node_masking(x: torch.Tensor, node_mask_prob: float = 0.05) -> torch.Tensor:
    """
    Simple node masking: randomly select nodes and mask ALL their features to 0.
    Simulates complete sensor failure at intersection level.
    
    Args:
        x: Node features [num_nodes, num_features]
        node_mask_prob: Probability of masking each node completely
    
    Returns:
        x_masked: Node features with some nodes completely masked to 0
    """
    if node_mask_prob <= 0:
        return x
    
    num_nodes = x.shape[0]
    
    # Randomly select nodes to mask completely
    nodes_to_mask = torch.rand(num_nodes, device=x.device) < node_mask_prob
    
    if nodes_to_mask.any():
        x_masked = x.clone()
        # Set ALL features of selected nodes to 0
        x_masked[nodes_to_mask] = 0.0
        return x_masked
    else:
        return x

def apply_gaussian_noise_to_features(x: torch.Tensor, filtered_feature_mapping: dict = None) -> torch.Tensor:
    """
    Apply Gaussian noise to normalized features using filtered feature mapping.
    If no mapping provided, skip noise application.
    """
    if filtered_feature_mapping is None:
        print("Warning: No filtered_feature_mapping provided. Skipping feature noise.")
        return x
    
    noise = torch.zeros_like(x)
    
    # Volume features
    if EdgeFeatures.VOL_BASE_CASE.value in filtered_feature_mapping:
        idx = filtered_feature_mapping[EdgeFeatures.VOL_BASE_CASE.value]
        if isinstance(idx, int) and idx < x.shape[1]:
            noise[:, idx] = torch.randn(x[:, idx].size(), device=x.device) * 0.1  # ±10%
    
    # Capacity features
    if EdgeFeatures.CAPACITY_BASE_CASE.value in filtered_feature_mapping:
        idx = filtered_feature_mapping[EdgeFeatures.CAPACITY_BASE_CASE.value]
        if isinstance(idx, int) and idx < x.shape[1]:
            noise[:, idx] = torch.randn(x[:, idx].size(), device=x.device) * 0.1  # ±10%
    
    # Length features
    if EdgeFeatures.LENGTH.value in filtered_feature_mapping:
        idx = filtered_feature_mapping[EdgeFeatures.LENGTH.value]
        if isinstance(idx, int) and idx < x.shape[1]:
            noise[:, idx] = torch.randn(x[:, idx].size(), device=x.device) * 0.05  # ±5%
    
    # Speed features
    if EdgeFeatures.FREESPEED.value in filtered_feature_mapping:
        idx = filtered_feature_mapping[EdgeFeatures.FREESPEED.value]
        if isinstance(idx, int) and idx < x.shape[1]:
            noise[:, idx] = torch.randn(x[:, idx].size(), device=x.device) * 0.05  # ±5%
    
    # Conditional noise for capacity reduction
    cap_red_idx = filtered_feature_mapping.get(EdgeFeatures.CAPACITY_REDUCTION.value)
    cap_base_idx = filtered_feature_mapping.get(EdgeFeatures.CAPACITY_BASE_CASE.value)
    
    if (cap_red_idx is not None and isinstance(cap_red_idx, int) and cap_red_idx < x.shape[1] and 
        cap_base_idx is not None and isinstance(cap_base_idx, int) and cap_base_idx < x.shape[1]):
        
        mask = x[:, cap_red_idx] == 1
        if mask.any():
            noise[mask, cap_base_idx] += torch.randn(mask.sum(), device=x.device) * 0.05  # Extra ±5% if policy applied
    
    # Apply noise and clamp to reasonable bounds
    x_noisy = x + noise
    
    return x_noisy


def collate_fn(data_list, node_feature_filter, filtered_feature_mapping=None, is_training=True,
               augment_pos_rotation=False, augment_feature_noise=False, augment_node_masking_prob=0.0):
    """
    Collate function with rotation, Gaussian noise, and simple node masking augmentation.
    Edge dropout and Position extraction is now handled in the model.
    """
    
    # STEP 1: Filter node features FIRST (before augmentation)
    for data in data_list:
        if node_feature_filter is not None:
            data.x = data.x[:, node_feature_filter]
    
    # STEP 2: Apply augmentation to filtered features
    # On-the-fly rotation augmentation
    if is_training and augment_pos_rotation:
        for data in data_list:
            if hasattr(data, 'pos') and data.pos is not None:
                # GENERATE ONE ROTATION ANGLE PER GRAPH
                theta = random.uniform(0, 2 * math.pi)
                
                # Apply rotation to the appropriate dimension based on pos shape
                if len(data.pos.shape) == 3 and data.pos.shape[1] == 3:
                    # For [N, 3, 2] format, apply SAME rotation angle to all positions
                    data.pos = rotate_pos(data.pos, theta)
                elif len(data.pos.shape) == 2:
                    # For [N, 2] format, apply rotation directly
                    data.pos = rotate_pos(data.pos, theta)
    
    # On-the-fly feature noise augmentation
    if is_training and augment_feature_noise:
        for data in data_list:
            if hasattr(data, 'x') and data.x is not None:
                data.x = apply_gaussian_noise_to_features(data.x, filtered_feature_mapping)
    
    # On-the-fly simple node masking
    if is_training and augment_node_masking_prob > 0:
        for data in data_list:
            if hasattr(data, 'x') and data.x is not None:
                data.x = apply_simple_node_masking(data.x, node_mask_prob=augment_node_masking_prob)

    # STEP 3: Create batch (features already filtered and augmented)
    batch = Batch.from_data_list(data_list)
    
    return batch


# TODO: Test and clean this, seems very bloated! 
################################################# ↓ Nested Neighbor Loader ↓ #################################################

def nested_dataloader(base_train_loader: DataLoader,
                     neighbor_sizes: list[int] = [15, 10, 5],
                     subgraphs_per_graph: int = 3,
                     seed_size: int = 1,
                     final_batch_size: int = 24,
                     sampling_strategy: str = 'neighbor_sampling',
                     min_subgraph_nodes: int = 10,
                     max_subgraph_nodes: int = 100,
                     node_feature_filter: list = None,
                     filtered_feature_mapping: dict = None,
                     augment_pos_rotation: bool = False,
                     augment_feature_noise: bool = False,
                     augment_node_masking_prob: float = 0.0,  
                     is_training: bool = True) -> DataLoader:
    """
    Create enhanced nested dataloader with on-the-fly subgraph generation and augmentation.
    """
    
    nested_dataset = NestedNeighborDataset(
        graph_loader=base_train_loader,
        neighbor_sizes=neighbor_sizes,
        subgraphs_per_graph=subgraphs_per_graph,
        seed_size=seed_size,    
        sampling_strategy=sampling_strategy,
        min_subgraph_nodes=min_subgraph_nodes,
        max_subgraph_nodes=max_subgraph_nodes,
        shuffle_mapping=False, ## Set to True for two layer randomization in combination with outer dataloader shuffle
        node_feature_filter=node_feature_filter,
        filtered_feature_mapping=filtered_feature_mapping,
        augment_pos_rotation=augment_pos_rotation,
        augment_feature_noise=augment_feature_noise,
        augment_node_masking_prob=augment_node_masking_prob,
        is_training=is_training
    )
    
    nested_loader = DataLoader(
        nested_dataset,
        batch_size=final_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=Batch.from_data_list,  # Simple, no augmentation here
        drop_last=False,
    )
    
    print(f"Created nested dataloader with {len(nested_dataset)} subgraphs")
    print(f"Final batches will contain {final_batch_size} subgraphs each")
    
    return nested_loader

class NestedNeighborDataset(Dataset):
    '''
    On-the-fly subgraph generation with Dataset interface.
    Generates fresh subgraphs every epoch using neighbor sampling,
    providing better variety and following modern GNN practices.
    '''
    def __init__(self,
                 graph_loader: DataLoader,
                 neighbor_sizes: list,
                 subgraphs_per_graph: int,
                 seed_size: int,  
                 sampling_strategy: str,
                 min_subgraph_nodes: int,
                 max_subgraph_nodes: int,
                 shuffle_mapping: bool = False,
                 node_feature_filter: list = None,
                 filtered_feature_mapping: dict = None,
                 augment_pos_rotation: bool = False,
                 augment_feature_noise: bool = False,
                 augment_node_masking_prob: float = 0.0,
                 is_training: bool = True):
        
        # Store existing parameters
        self.neighbor_sizes = neighbor_sizes
        self.subgraphs_per_graph = subgraphs_per_graph
        self.seed_size = seed_size  
        self.sampling_strategy = sampling_strategy
        self.min_subgraph_nodes = min_subgraph_nodes
        self.max_subgraph_nodes = max_subgraph_nodes
        self.shuffle_mapping = shuffle_mapping
        self.node_feature_filter = node_feature_filter
        self.filtered_feature_mapping = filtered_feature_mapping
        self.augment_pos_rotation = augment_pos_rotation
        self.augment_feature_noise = augment_feature_noise
        self.augment_node_masking_prob = augment_node_masking_prob  
        self.is_training = is_training
        
        print(f"Initializing On-The-Fly Nested Dataset with {subgraphs_per_graph} subgraphs per graph")
        print(f"Using single sampling strategy: {self.sampling_strategy}")
        print(f"Shuffle mapping: {shuffle_mapping}")
        print(f"Feature filter: {node_feature_filter}")
        print(f"Feature mapping: {filtered_feature_mapping}")
        print(f"Node masking probability: {augment_node_masking_prob}")
        
        # Store original dataset for on-demand loading
        self.original_dataset = graph_loader.dataset
        
        # Pre-compute the mapping from subgraph index to original graph
        self._compute_subgraph_mapping()
        
        print(f"Total subgraphs to generate on-the-fly: {len(self.subgraph_mapping)}")
    
    def _compute_subgraph_mapping(self):
        """Pre-compute which original graph each subgraph index corresponds to."""
        self.subgraph_mapping = []
        
        # Direct iteration over dataset (much simpler!)
        for graph_idx in range(len(self.original_dataset)):
            for subgraph_idx in range(self.subgraphs_per_graph):
                self.subgraph_mapping.append({
                    'original_graph_idx': graph_idx,
                    'subgraph_idx': subgraph_idx
                })
        
        # Shuffle mapping if requested (for better randomization)
        if self.shuffle_mapping:
            import random
            random.shuffle(self.subgraph_mapping)
            print("Shuffled subgraph mapping for better randomization")
        
        print(f"Processed {len(self.original_dataset)} original graphs")
    
    def _sample_subgraphs_from_single_graph(self, graph: Data, subgraph_idx: int) -> Data:
        """Sample a single subgraph from a graph."""
        
        # Skip if graph is too small
        if graph.num_nodes < self.min_subgraph_nodes:
            # For small graphs, just replicate the original graph
            subgraph = graph.clone()
            subgraph.sampling_strategy = self.sampling_strategy
            subgraph.subgraph_idx = subgraph_idx
            return subgraph
        
        try:
            if self.sampling_strategy == "neighbor_sampling":
                subgraph = self._neighbor_sampling_subgraph(graph)
            elif self.sampling_strategy == "random_walk":
                subgraph = self._random_walk_subgraph(graph)
            else:
                raise ValueError(f"Invalid sampling strategy: {self.sampling_strategy}")
            
            # Add metadata to subgraph
            subgraph.sampling_strategy = self.sampling_strategy
            subgraph.subgraph_idx = subgraph_idx
            
            # Validate subgraph size
            if subgraph.num_nodes > self.max_subgraph_nodes:
                subgraph = self._truncate_subgraph(subgraph, self.max_subgraph_nodes)
            
            return subgraph
            
        except Exception as e:
            print(f"Error sampling subgraph {subgraph_idx}: {e}")
            # Fallback to original graph
            subgraph = graph.clone()
            subgraph.sampling_strategy = self.sampling_strategy
            subgraph.subgraph_idx = subgraph_idx
            return subgraph
    
    def _neighbor_sampling_subgraph(self, graph: Data) -> Data:
        """
        Use NeighborLoader for subgraph sampling.
        
        Each subgraph contains:
        - 1 seed node (seed_size=1)   
        - Neighbors sampled according to neighbor_sizes [15, 10, 5]
        - Total nodes: ~1 + 15 + 15*10 + 15*10*5 = ~800 nodes max
        """
        all_nodes = torch.arange(graph.num_nodes)
        
        # Create neighbor loader for this specific graph
        neigh_loader = NeighborLoader(
            data=graph,
            num_neighbors=self.neighbor_sizes,
            input_nodes=all_nodes,
            batch_size=self.seed_size,  # 1 seed per subgraph
            shuffle=True,
            # Remove return_e_id parameter as it's not available in this version
        )
        
        # Get first subgraph from the loader
        for subgraph in neigh_loader:
            if hasattr(subgraph, 'pos') and hasattr(subgraph, 'n_id'):
                subgraph.pos = graph.pos[subgraph.n_id]
            return subgraph
    
    def _random_walk_subgraph(self, graph: Data, walk_length: int = 13, num_walks: int = 4) -> Data:
        """Sample subgraph using random walks."""
        if graph.num_nodes == 0:
            return graph.clone()
        
        # Start from random nodes
        start_nodes = torch.randperm(graph.num_nodes)[:min(num_walks, graph.num_nodes)]
        visited_nodes = set()
        
        for start_node in start_nodes:
            current_node = start_node.item()
            visited_nodes.add(current_node)
            
            for _ in range(walk_length):
                # Get neighbors
                edge_mask = (graph.edge_index[0] == current_node)
                neighbors = graph.edge_index[1][edge_mask]
                
                if len(neighbors) == 0:
                    break
                
                # Random walk step
                next_node = neighbors[torch.randint(len(neighbors), (1,))].item()
                visited_nodes.add(next_node)
                current_node = next_node
                
                # Stop if we have enough nodes
                if len(visited_nodes) >= self.max_subgraph_nodes:
                    break
            
            if len(visited_nodes) >= self.max_subgraph_nodes:
                break
        
        # Create subgraph from visited nodes
        if len(visited_nodes) < self.min_subgraph_nodes:
            return self._fallback_subgraph(graph)
        
        subset = torch.tensor(list(visited_nodes), dtype=torch.long)
        return self._create_subgraph_from_nodes(graph, subset)
    
    def _create_subgraph_from_nodes(self, graph: Data, subset: torch.Tensor) -> Data:
        """Create subgraph from node subset."""
        if len(subset) == 0:
            return self._fallback_subgraph(graph)
        
        # Get subgraph edges
        edge_index, edge_attr = subgraph(
            subset, graph.edge_index, 
            edge_attr=graph.edge_attr if hasattr(graph, 'edge_attr') else None,
            relabel_nodes=True,
            num_nodes=graph.num_nodes
        )
        
        # Create new data object with properly subsetted features
        subgraph_data = Data(
            x=graph.x[subset] if graph.x is not None else None,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        
        # Handle positional features - subset them properly
        if hasattr(graph, 'pos') and graph.pos is not None:
            subgraph_data.pos = graph.pos[subset]  # Only keep positions of selected nodes
        
        # LapPE: copy from original graph for subset nodes
        if hasattr(graph, 'lap_pe') and graph.lap_pe is not None:
            subgraph_data.lap_pe = graph.lap_pe[subset]
        
        # City attribute for DANN
        if hasattr(graph, 'city'):
            subgraph_data.city = graph.city
    
        # Handle target values - subset them properly 
        if hasattr(graph, 'y') and graph.y is not None:
            subgraph_data.y = graph.y[subset]  # Only keep targets of selected nodes
        
        # Copy other attributes selectively (avoid copying node-level attributes)
        node_level_attrs = {'x', 'edge_index', 'edge_attr', 'y', 'pos', 'lap_pe', 'city', 'num_nodes', 'num_edges'}
        for key, value in graph.items():
            if key not in node_level_attrs:
                # Only copy graph-level attributes, not node-level ones
                if not torch.is_tensor(value) or value.size(0) != graph.num_nodes:
                    setattr(subgraph_data, key, value)
        
        subgraph_data.orig_subset = subset  # Store original node indices for edge_weights mapping
        return subgraph_data
    
    def _truncate_subgraph(self, subgraph: Data, max_nodes: int) -> Data:
        """Truncate subgraph to maximum number of nodes."""
        if subgraph.num_nodes <= max_nodes:
            return subgraph
        
        # Randomly select nodes to keep
        keep_nodes = torch.randperm(subgraph.num_nodes)[:max_nodes]
        return self._create_subgraph_from_nodes(subgraph, keep_nodes)
    
    def _extract_subgraph_edge_weights(self, subgraph: Data, original_graph: Data) -> torch.Tensor:
        """Extract node weights (road segment weights) for subgraph nodes."""
        # Check if original graph has edge_weights
        if not hasattr(original_graph, 'edge_weights') or original_graph.edge_weights is None:
            raise ValueError(f"Original graph missing 'edge_weights' attribute. Available attributes: {list(original_graph.keys())}")

        # Neighbor sampling: use n_id
        if hasattr(subgraph, 'n_id') and subgraph.n_id is not None:
            return original_graph.edge_weights[subgraph.n_id]
        # Random walk: use mapping from subgraph nodes to original nodes
        elif hasattr(subgraph, 'orig_subset') and subgraph.orig_subset is not None:
            return original_graph.edge_weights[subgraph.orig_subset]
        # Fallback: if subgraph nodes are a subset of original nodes in order
        elif hasattr(subgraph, 'x') and subgraph.x.shape[0] <= original_graph.edge_weights.shape[0]:
            # Try to use the subset indices if stored
            if hasattr(subgraph, 'subset_indices'):
                return original_graph.edge_weights[subgraph.subset_indices]
            else:
                # If not, just take the first N weights (not always correct!)
                return original_graph.edge_weights[:subgraph.x.shape[0]]
        else:
            raise ValueError(f"Cannot map subgraph nodes to original graph for edge_weights. Subgraph attributes: {list(subgraph.keys())}")
        
    def _fallback_subgraph(self, graph: Data) -> Data:
        """Fallback subgraph for small graphs."""
        return graph.clone()
    
    def __len__(self):
        return len(self.subgraph_mapping)
    
    def __getitem__(self, idx):
        """Generate subgraph on-the-fly using neighbor sampling and data augmentation."""
        mapping = self.subgraph_mapping[idx]
        
        # Load the original graph on-demand (direct indexing)
        original_graph = self.original_dataset[mapping['original_graph_idx']]
        
        # Generate the subgraph dynamically (fresh every epoch)
        subgraph = self._sample_subgraphs_from_single_graph(
            graph=original_graph,
            subgraph_idx=mapping['subgraph_idx']
        )
        
        # Apply augmentation to the subgraph
        subgraph = self._apply_augmentation(subgraph)
        
        # Extract edge weights for subgraph
        subgraph.edge_weights = self._extract_subgraph_edge_weights(subgraph, original_graph)
        
        # Apply feature filtering to the subgraph
        if hasattr(self, 'node_feature_filter') and self.node_feature_filter is not None:
            # Only filter if number of features is greater than max index in filter
            if subgraph.x.shape[1] > len(self.node_feature_filter):
                subgraph.x = subgraph.x[:, self.node_feature_filter]
        
        return subgraph
    
    def _apply_augmentation(self, subgraph: Data) -> Data:
        """Apply augmentation to a single subgraph."""
    
        # On-the-fly rotation augmentation
        if self.is_training and self.augment_pos_rotation:
            if hasattr(subgraph, 'pos') and subgraph.pos is not None:
                theta = random.uniform(0, 2 * math.pi)
                if len(subgraph.pos.shape) == 3 and subgraph.pos.shape[1] == 3:
                    # For [N, 3, 2] format, apply SAME rotation angle to all positions
                    subgraph.pos = rotate_pos(subgraph.pos, theta)
                elif len(subgraph.pos.shape) == 2:
                    # For [N, 2] format, apply rotation directly
                    subgraph.pos = rotate_pos(subgraph.pos, theta)
        
        # On-the-fly feature noise augmentation
        if self.is_training and self.augment_feature_noise:
            if hasattr(subgraph, 'x') and subgraph.x is not None:
                subgraph.x = apply_gaussian_noise_to_features(subgraph.x, self.filtered_feature_mapping)

        # On-the-fly node masking augmentation
        if self.is_training and self.augment_node_masking_prob > 0:
            if hasattr(subgraph, 'x') and subgraph.x is not None:
                subgraph.x = apply_simple_node_masking(subgraph.x, node_mask_prob=self.augment_node_masking_prob)
        
        return subgraph