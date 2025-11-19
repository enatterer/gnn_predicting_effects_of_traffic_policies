# Validation Loss Improvement Heatmap

## Overview

This script creates a heatmap visualization showing the percentage reduction in validation loss when finetuning from a pretrained model versus training from scratch, across different data availability scenarios (20/40/60 graphs per city).

## Output

The script generates a single heatmap:
- **val_loss_improvement_heatmap.png/pdf** - Heatmap showing validation loss reduction percentage
  - Rows: Cities
  - Columns: Data availability scenarios (20/40/60 graphs)
  - Values: Percentage reduction in validation loss (higher is better)

## Usage

### Extract Results from WandB

```bash
python scripts/evaluation/visualize_finetuning_results.py \
    --wandb_project tum-traffic-engineering/GNN_Transductive \
    --output_dir results/visualizations \
    --export_json results/results_export.json
```

### Use Pre-extracted JSON Results

If you've already exported results to JSON:

```bash
python scripts/evaluation/visualize_finetuning_results.py \
    --results_json results/results_export.json \
    --output_dir results/visualizations
```

### Use Results from Directory

```bash
python scripts/evaluation/visualize_finetuning_results.py \
    --results_dir results/json_files \
    --output_dir results/visualizations
```

## JSON Format

Results should be in JSON format with the following structure:

```json
[
  {
    "data_availability": 20,
    "city": "regensburg",
    "method": "finetune",
    "metrics": {
      "r2": 0.8888,
      "val_loss": 44.52,
      "spearman": 0.5905,
      "pearson": 0.9429
    }
  },
  {
    "data_availability": 20,
    "city": "regensburg",
    "method": "scratch",
    "metrics": {
      "r2": 0.6902,
      "val_loss": 119.85,
      "spearman": 0.3993,
      "pearson": 0.8362
    }
  }
]
```

## Paper Presentation

The heatmap clearly shows:
- **Consistency**: Finetuning consistently reduces validation loss across all cities
- **Data efficiency**: The improvement is visible even with limited data (20 graphs)
- **Scalability**: How the improvement changes as more data becomes available (20→40→60 graphs)

### Example Figure Caption:
"Validation loss reduction achieved by finetuning from a pretrained model versus training from scratch. The heatmap shows the percentage reduction in validation loss (MSE) across different cities and data availability scenarios. Higher values (darker green) indicate greater improvement from finetuning."

## Troubleshooting

### No results found
- Check that WandB project name is correct (format: `entity/project`)
- Verify run names match expected patterns: `finetune_{city}__parent-...` or `run_from_scratch_{city}__parent-...`
- Ensure `limit_train_graphs` is set in the config (should be 20, 40, or 60)

### Missing metrics
- The script requires `val_loss` metric for both finetune and scratch methods
- Other metrics (R², Spearman, Pearson) are not used but can be present in the data

### City extraction issues
- The script extracts city names from run names
- If city names don't match, check the run naming convention in `run_and_finetune_all_cities.py`
