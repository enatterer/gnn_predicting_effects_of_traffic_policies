#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2880
#SBATCH --job-name=trial_run_gatv2_transductive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=ankit.basu@tum.de

# Run srun with container
srun --container-image ~/pyg.sqsh \
     --container-mounts=/dss/dsshome1/03/go69tef2/gnn_predicting_effects_of_traffic_policies:/mnt/repo \
     --overlap \
     bash -c "cd /mnt/repo/scripts/training && python run_models.py --gnn_arch trans_conv --unique_model_description trans_conv_F5_100_T --in_channels 5 --use_all_features True --num_epochs 100 --batch_size 8 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3"