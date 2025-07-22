#!/usr/bin/env python3
"""
Comprehensive hyperparameter tuning for GNN architectures.
Integrates with existing run_models.py and provides automated search using Optuna.

Usage:
python hyperparameter_tuning.py --gnn_arch gatv2 --n_trials 100 --study_name gatv2_hp_search
"""

import optuna
import os
import sys
import json
import subprocess
import tempfile
from typing import Dict, Any
import argparse

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

def get_search_space(gnn_arch: str, trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define architecture-specific hyperparameter search spaces.
    Based on literature best practices and your specific traffic prediction task.
    """
    
    # Common hyperparameters for all architectures
    common_params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        'num_epochs': trial.suggest_int('num_epochs', 50, 500, step=50),
        'batch_size': trial.suggest_categorical('batch_size', [1, 2, 4, 8]),
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 15, 50, step=5),
        'use_dropout': trial.suggest_categorical('use_dropout', [True, False]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.7, step=0.1) if trial.params.get('use_dropout', True) else 0.0,
        'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 3, 4]),
    }
    
    # Architecture-specific parameters
    arch_specific_params = {}
    
    if gnn_arch in ['gat', 'gatv2', 'gatv3']:
        arch_specific_params.update({
            'hidden_channels': trial.suggest_categorical('hidden_channels', [32, 64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'heads': trial.suggest_categorical('heads', [1, 2, 4, 8]),
            'concat': trial.suggest_categorical('concat', [True, False]),
            'edge_dim': trial.suggest_categorical('edge_dim', [None, 16, 32, 64]),
        })
    
    elif gnn_arch in ['gcn', 'gcn2']:
        arch_specific_params.update({
            'hidden_channels': trial.suggest_categorical('hidden_channels', [32, 64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 8),
            'alpha': trial.suggest_float('alpha', 0.1, 0.9, step=0.1) if gnn_arch == 'gcn2' else None,
            'theta': trial.suggest_float('theta', 0.5, 2.0, step=0.1) if gnn_arch == 'gcn2' else None,
        })
    
    elif gnn_arch == 'trans_conv':
        arch_specific_params.update({
            'hidden_channels': trial.suggest_categorical('hidden_channels', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'heads': trial.suggest_categorical('heads', [1, 2, 4, 8]),
            'concat': trial.suggest_categorical('concat', [True, False]),
            'beta': trial.suggest_categorical('beta', [True, False]),
            'edge_dim': trial.suggest_categorical('edge_dim', [None, 32, 64, 128]),
        })
    
    elif gnn_arch == 'graphSAGE':
        arch_specific_params.update({
            'hidden_channels': trial.suggest_categorical('hidden_channels', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'aggr': trial.suggest_categorical('aggr', ['mean', 'max', 'add']),
            'normalize': trial.suggest_categorical('normalize', [True, False]),
        })
    
    elif gnn_arch == 'eign':
        arch_specific_params.update({
            'hidden_channels': trial.suggest_categorical('hidden_channels', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 2, 5),
            'num_eigenvalues': trial.suggest_int('num_eigenvalues', 10, 50, step=5),
            'normalization': trial.suggest_categorical('normalization', ['sym', 'rw']),
        })
    
    # Remove None values
    arch_specific_params = {k: v for k, v in arch_specific_params.items() if v is not None}
    
    # Combine all parameters
    all_params = {**common_params, **arch_specific_params}
    
    return all_params

def create_model_kwargs_file(arch_specific_params: Dict[str, Any]) -> str:
    """Create a temporary JSON file with model-specific kwargs."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(arch_specific_params, temp_file, indent=2)
    temp_file.close()
    return temp_file.name

def objective(trial: optuna.Trial, gnn_arch: str, base_args: Dict[str, Any]) -> float:
    """
    Objective function for Optuna optimization.
    Runs the model with suggested hyperparameters and returns validation loss.
    """
    
    # Get hyperparameter suggestions
    params = get_search_space(gnn_arch, trial)
    
    # Separate common parameters from architecture-specific ones
    common_params = ['lr', 'num_epochs', 'batch_size', 'early_stopping_patience', 
                    'use_dropout', 'dropout', 'gradient_accumulation_steps']
    
    arch_params = {k: v for k, v in params.items() if k not in common_params}
    run_params = {k: v for k, v in params.items() if k in common_params}
    
    # Create model kwargs file if needed
    model_kwargs_file = None
    if arch_params:
        model_kwargs_file = create_model_kwargs_file(arch_params)
        run_params['model_kwargs'] = model_kwargs_file
    
    # Update unique model description for this trial
    run_params['unique_model_description'] = f"{base_args['unique_model_description']}_trial_{trial.number}"
    
    # Combine with base arguments
    final_args = {**base_args, **run_params}
    
    # Build command
    cmd = ['python', 'run_models.py']
    for key, value in final_args.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    try:
        # Run the training
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            print(f"Trial {trial.number} failed with error: {result.stderr}")
            raise optuna.TrialPruned()
        
        # Extract validation loss from output
        # This assumes your run_models.py prints the best validation loss
        output_lines = result.stdout.split('\n')
        best_val_loss = None
        
        for line in output_lines:
            if 'with validation loss:' in line:
                # Extract the validation loss value
                parts = line.split('with validation loss:')
                if len(parts) > 1:
                    try:
                        best_val_loss = float(parts[1].split()[0])
                        break
                    except (ValueError, IndexError):
                        continue
        
        if best_val_loss is None:
            print(f"Could not extract validation loss from trial {trial.number}")
            raise optuna.TrialPruned()
        
        return best_val_loss
        
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        raise optuna.TrialPruned()
    
    finally:
        # Cleanup temporary files
        if model_kwargs_file and os.path.exists(model_kwargs_file):
            os.unlink(model_kwargs_file)

def run_hyperparameter_search(gnn_arch: str, n_trials: int, study_name: str, 
                             base_args: Dict[str, Any]) -> optuna.Study:
    """
    Run the hyperparameter search using Optuna.
    """
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Create objective function with fixed arguments
    objective_with_args = lambda trial: objective(trial, gnn_arch, base_args)
    
    # Run optimization
    study.optimize(objective_with_args, n_trials=n_trials, timeout=None)
    
    return study

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for GNN models")
    
    # Hyperparameter search specific arguments
    parser.add_argument("--gnn_arch", type=str, required=True,
                       choices=["gat", "gatv2", "gatv3", "gcn", "gcn2", "trans_conv", "graphSAGE", "eign"],
                       help="GNN architecture to tune")
    parser.add_argument("--n_trials", type=int, default=100,
                       help="Number of hyperparameter trials")
    parser.add_argument("--study_name", type=str, required=True,
                       help="Name for the Optuna study")
    
    # Base arguments (that don't change during search)
    parser.add_argument("--project_name", type=str, default="HP_Tuning_Bavaria",
                       help="Project name for organizing results")
    parser.add_argument("--unique_model_description", type=str, 
                       default="hp_search_base", help="Base description for runs")
    parser.add_argument("--in_channels", type=int, default=5,
                       help="Number of input channels")
    parser.add_argument("--use_all_features", type=bool, default=False,
                       help="Whether to use all features")
    parser.add_argument("--out_channels", type=int, default=1,
                       help="Number of output channels")
    
    args = parser.parse_args()
    
    # Base arguments that remain constant across trials
    base_args = {
        'gnn_arch': args.gnn_arch,
        'project_name': args.project_name,
        'unique_model_description': args.unique_model_description,
        'in_channels': args.in_channels,
        'use_all_features': args.use_all_features,
        'out_channels': args.out_channels,
    }
    
    print(f"Starting hyperparameter search for {args.gnn_arch}")
    print(f"Running {args.n_trials} trials...")
    
    # Run the search
    study = run_hyperparameter_search(
        gnn_arch=args.gnn_arch,
        n_trials=args.n_trials,
        study_name=args.study_name,
        base_args=base_args
    )
    
    # Print results
    print("\nHyperparameter search completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_file = f"{args.study_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial_number': study.best_trial.number,
            'n_trials': len(study.trials)
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Optuna database saved to {args.study_name}.db")

if __name__ == '__main__':
    main() 