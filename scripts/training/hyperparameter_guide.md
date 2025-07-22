# Hyperparameter Tuning Guide for GNN Traffic Prediction

## Quick Start

### 1. Install Required Dependencies
```bash
pip install optuna plotly  # Add to your requirements.txt
```

### 2. Run Automated Search
```bash
# Example: Tune GATv2 for 50 trials
cd scripts/training
python hyperparameter_tuning.py --gnn_arch gatv2 --n_trials 50 --study_name gatv2_hex500_search

# Example: Tune TransConv with more trials
python hyperparameter_tuning.py --gnn_arch trans_conv --n_trials 100 --study_name transconv_hex500_search
```

## Search Space Breakdown

### Common Parameters (All Architectures)
- **Learning Rate**: `1e-4` to `1e-1` (log scale)
- **Epochs**: `50` to `500` (step 50)
- **Batch Size**: `[1, 2, 4, 8]`
- **Dropout**: `0.1` to `0.7` (step 0.1)
- **Early Stopping**: `15` to `50` patience

### Architecture-Specific Parameters

#### GAT/GATv2/GATv3
- **Hidden Channels**: `[32, 64, 128, 256]`
- **Layers**: `2` to `6`
- **Attention Heads**: `[1, 2, 4, 8]`
- **Concat**: `[True, False]`
- **Edge Dimension**: `[None, 16, 32, 64]`

#### TransConv
- **Hidden Channels**: `[64, 128, 256, 512]`
- **Layers**: `2` to `6`
- **Heads**: `[1, 2, 4, 8]`
- **Beta (learnable skip)**: `[True, False]`

#### GraphSAGE
- **Hidden Channels**: `[64, 128, 256, 512]`
- **Aggregation**: `['mean', 'max', 'add']`
- **Normalization**: `[True, False]`

## Practical Workflow

### Phase 1: Quick Architecture Comparison (20-30 trials each)
```bash
# Test all architectures with limited trials
for arch in gatv2 trans_conv graphSAGE eign; do
    python hyperparameter_tuning.py --gnn_arch $arch --n_trials 30 --study_name ${arch}_quick_test
done
```

### Phase 2: Deep Tuning for Best Architectures (100+ trials)
```bash
# Focus on top 2-3 performers from Phase 1
python hyperparameter_tuning.py --gnn_arch gatv2 --n_trials 100 --study_name gatv2_deep_search
```

### Phase 3: Feature Engineering with Best Hyperparameters
Use the best parameters from Phase 2, then manually test:
- Different normalization strategies
- Independent feature inclusion/exclusion
- SO3 equivariant features

## Manual Testing for Quick Experiments

For quick tests when you have an idea about specific parameters:

```bash
# Test specific configuration
python run_models.py \
  --gnn_arch gatv2 \
  --unique_model_description manual_test_gatv2_256_hidden \
  --in_channels 5 \
  --use_all_features False \
  --num_epochs 100 \
  --lr 0.005 \
  --early_stopping_patience 20 \
  --use_dropout True \
  --dropout 0.3 \
  --model_kwargs model_configs/gatv2_256_hidden.json
```

## Expected Performance Ranges

Based on traffic prediction literature:

### Learning Rates
- **Too high** (>0.01): Training instability
- **Sweet spot** (0.001-0.01): Most architectures
- **Too low** (<0.0001): Slow convergence

### Hidden Dimensions
- **Small models** (32-64): Fast but may underfit
- **Medium models** (128-256): Good balance
- **Large models** (512+): Risk of overfitting with limited data

### Layers
- **Shallow** (2-3): Good for simple patterns
- **Medium** (4-5): Captures complex spatial relationships
- **Deep** (6+): Risk of over-smoothing in GNNs

## Analyzing Results

### View Optuna Dashboard
```bash
# Install dashboard
pip install optuna-dashboard

# Launch dashboard (run in background)
optuna-dashboard sqlite:///gatv2_hex500_search.db
```

### Load Best Parameters Programmatically
```python
import optuna
study = optuna.load_study(study_name="gatv2_hex500_search", 
                         storage="sqlite:///gatv2_hex500_search.db")
print("Best params:", study.best_params)
print("Best value:", study.best_value)
```

## Pro Tips

### 1. Start Small, Scale Up
- Begin with 20-30 trials per architecture
- Identify top 2-3 performers
- Deep tune only the promising ones

### 2. Monitor Computational Cost
- Track trial duration: `study.trials[i].duration`
- Set reasonable `n_trials` based on compute budget
- Use early stopping aggressively

### 3. Architecture-Specific Insights
- **GATv2**: Usually benefits from more attention heads
- **TransConv**: Often works well with larger hidden dimensions
- **GraphSAGE**: 'mean' aggregation is often most stable
- **EIGN**: Sensitive to number of eigenvalues

### 4. Transfer Learning
- Use best parameters from hex 500 as starting point for hex 1000/2000
- Fine-tune only learning rate and regularization for larger scales

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or hidden dimensions
2. **NaN Losses**: Lower learning rate, add gradient clipping
3. **No Convergence**: Increase epochs or patience
4. **Overfitting**: Increase dropout, reduce model size

### Debug Mode
Add `--num_epochs 10` for quick testing during hyperparameter development. 