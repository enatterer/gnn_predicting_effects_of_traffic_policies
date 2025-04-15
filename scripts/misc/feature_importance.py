"""
This script calculates feature importance for the GNN model using the
GNNExplainer algorithm. It loads test data and the best model from a training run,
wraps the model to make it compatible with the explainer, and computes feature
importance scores for the input node features. The results are averaged over all test
batches and plotted as a bar chart.
"""

import os
import sys
import json

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

import gnn.gnn_io as gio
import gnn.models.point_net_transf_gat as garch
import training.help_functions as hf
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load test data
run_path = os.path.join(project_root, 'data', 'runs_21_10_2024', 'wannabe_best_6')

with open(os.path.join(run_path, 'data_created_during_training', 'test_loader_params.json')) as f:
    test_loader_params = json.load(f)

batch_size = test_loader_params['batch_size']
test_set_normalized = torch.load(os.path.join(run_path, 'data_created_during_training', 'test_dl.pt'))
test_loader = DataLoader(dataset=test_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=gio.collate_fn, worker_init_fn=hf.seed_worker)

# Load best model
model = garch.PointNetTransfGAT(in_channels=13,
                    out_channels=1,
                    point_net_conv_layer_structure_local_mlp=[256],
                    point_net_conv_layer_structure_global_mlp=[512],
                    gat_conv_layer_structure=[128,256,512,256],
                    use_dropout=False,
                    dropout=0.3,
                    predict_mode_stats=False,
                    dtype=torch.float32)

model_path = os.path.join(run_path, 'trained_model', 'model.pth')
model.load_state_dict(torch.load(model_path))

# Create a model wrapper for the explainer, makes the forward function compatible
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x, edge_index, data):
        data.x = x
        data.edge_index = edge_index
        return self.model(data)

wrapped_model = ModelWrapper(model)

# Explainer for feature importance
explainer = Explainer(model=wrapped_model,
                      algorithm=GNNExplainer(),
                      explanation_type='phenomenon', # wrt real target
                      node_mask_type='attributes', # mask each feature across all nodes
                      model_config=dict(mode='regression',
                                        task_level='node',
                                        return_type='raw'))

# Get results on all batches, average over all nodes
explanations = []
for i, batch in enumerate(tqdm(test_loader, desc="Explaining batches")):
    explanation = explainer(x=batch.x,
                            edge_index=batch.edge_index,
                            target=batch.y,
                            data=batch)
    explanations.append(explanation)

feature_labels = [feat.name for feat in EdgeFeatures]

# Average feature importance over all explanations (batches)
feature_importances = np.mean([exp.get('node_mask').sum(dim=0).cpu().detach().numpy() for exp in explanations], axis=0)

# Sort
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_labels = [feature_labels[i] for i in sorted_indices]
sorted_feature_importances = feature_importances[sorted_indices]

# Plot feature importance
plt.figure(figsize=(14, 7))
plt.bar(sorted_feature_labels, sorted_feature_importances)
plt.title('Feature Importance')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(sorted_feature_importances):
    plt.text(i, v, str(round(v)), ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(run_path, 'feature_importance.png'))