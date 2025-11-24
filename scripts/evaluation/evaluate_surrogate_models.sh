#!/bin/bash
# Script to evaluate both surrogate models (2 cities and 6 cities) on test cities

# Change to project root directory (two levels up from scripts/evaluation/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Base parameters
PROJECT_NAME="GNN_Transductive"
CITIES="bamberg,erlangen,muenchen,neuulm"
GRAPHS_PER_CITY=200
DATASET_PATH="data/bavaria/inductive_data/training_data/kreisfreistadt"
DEVICE_NR=0
GNN_ARCH="trans_encoder"

# Pretrained runs to evaluate
PRETRAIN_RUNS=("general_surrogate_2_cities" "general_surrogate_6_cities" "general_surrogate_10_cities")

# Run evaluation for each pretrained model
for PRETRAIN_RUN in "${PRETRAIN_RUNS[@]}"; do
    # Extract number of cities from run name (e.g., "general_surrogate_2_cities" -> "2")
    NUM_CITIES=$(echo ${PRETRAIN_RUN} | grep -oE '[0-9]+')
    
    # Create distinct output directory based on number of cities in training
    OUTPUT_DIR="data/analysis_results_${NUM_CITIES}_cities_in_training"
    mkdir -p ${OUTPUT_DIR}
    
    echo "=========================================="
    echo "Evaluating: ${PRETRAIN_RUN}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "=========================================="
    
    # Run evaluation for each city separately with nohup
    for CITY in ${CITIES}; do
        echo "Starting evaluation for ${PRETRAIN_RUN} on ${CITY}..."
        nohup python scripts/evaluation/evaluate_pretrained_on_cities.py \
            --pretrain_run_name ${PRETRAIN_RUN} \
            --project_name ${PROJECT_NAME} \
            --cities ${CITY} \
            --graphs_per_city ${GRAPHS_PER_CITY} \
            --dataset_path ${DATASET_PATH} \
            --output_dir ${OUTPUT_DIR} \
            --gnn_arch ${GNN_ARCH} \
            --device_nr ${DEVICE_NR} \
            > ${OUTPUT_DIR}/evaluation_${PRETRAIN_RUN}_${CITY}.log 2>&1 &
        
        echo "Started evaluation for ${PRETRAIN_RUN} - ${CITY} (PID: $!)"
        echo "Log file: ${OUTPUT_DIR}/evaluation_${PRETRAIN_RUN}_${CITY}.log"
        echo ""
        
        # Wait a bit between starting each city to avoid resource conflicts
        sleep 5
    done
done

echo "=========================================="
echo "All evaluation jobs started!"
echo "=========================================="
echo "Check logs in: data/analysis_results_*_cities_in_training/evaluation_*.log"
echo "Check results in: data/analysis_results_*_cities_in_training/evaluation_*.csv"
echo ""
echo "To check running jobs: ps aux | grep evaluate_pretrained_on_cities"
echo "To check GPU usage: watch -n 1 nvidia-smi"


# For other runs, use following paths:
# # Base command parameters
# PRETRAIN_RUN="general_surrogate_v0"
# PROJECT_NAME="Bavaria_Test"
# GRAPHS_PER_CITY=200
# DATASET_PATH="data/bavaria/inductive_data/training_data/kreisfreistadt"
# OUTPUT_DIR="data/analysis_results"
# DEVICE_NR=0
# GNN_ARCH="trans_encoder"