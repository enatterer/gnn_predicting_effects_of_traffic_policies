#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2880
#SBATCH --job-name=te_gnn_2_10_part_2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run srun with container
srun --container-image ~/pyg.sqsh \
     --container-mounts=/dss/dssfs04/lwp-dss-0002/t5621/t5621-dss-0001/Development/gnn_predicting_effects_of_traffic_policies/:/mnt/repo \
     --overlap \
     bash -c "cd /mnt/repo/scripts/training && python run_models.py --unique_model_description "te_gnn_2_10_part_2" --gnn_arch "trans_encoder" --batch_size 8 --lr 3e-4 --num_epochs 500 --early_stopping_patience 25 --model_kwargs "../../logs/te_gnn_2_10.json" --continue_training True --base_checkpoint_path "../../data/TR-C_Benchmarks/te_gnn_2_10/trained_model/checkpoints/checkpoint_epoch_100.pt""