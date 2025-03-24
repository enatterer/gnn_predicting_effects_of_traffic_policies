#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2880
#SBATCH --job-name=wannabe_best_6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run srun with container
srun --container-image ~/pyg.sqsh \
     --container-mounts=/dss/dssfs04/lwp-dss-0002/t5621/t5621-dss-0001/Development/gnn_predicting_effects_of_traffic_policies/:/mnt/repo \
     --overlap \
     bash -c "cd /mnt/repo/scripts/training && python run_models.py --gat_conv_layer_structure 128,256,512 --batch_size 8 --lr 5e-4 --num_epochs 750 --early_stopping_patience 40"