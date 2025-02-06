#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1800
#SBATCH --job-name=ensemble_5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run srun with container
srun --container-image=~/pyg.sqsh \
     --container-mounts=/dss/dssfs04/lwp-dss-0002/t5621/t5621-dss-0001/Development/gnn_predicting_effects_of_traffic_policies/:/mnt/repo \
     --overlap \
     bash -c "cd /mnt/repo/scripts/training && python run_models.py --gat_conv_layer_structure 128,256,512 --use_dropout True --dropout 0.1 --use_bootrapping True --lr 2e-3 --batch_size 32 --num_epochs 1000 --early_stopping_patience 25"