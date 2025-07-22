#!/usr/bin/env python3
"""
Analyze and compare hyperparameter tuning results from multiple Optuna studies.

Usage:
python analyze_hyperparameters.py --study_names gatv2_search trans_conv_search --top_n 5
"""

import optuna
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import os
from typing import List, Dict, Any
import json

def load_study(study_name: str) -> optuna.Study:
    """Load an Optuna study from SQLite database."""
    try:
        return optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{study_name}.db"
        )
    except Exception as e:
        print(f"Error loading study {study_name}: {e}")
        return None

def study_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Convert Optuna study to pandas DataFrame for analysis."""
    if not study:
        return pd.DataFrame()
    
    data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {
                'trial_number': trial.number,
                'value': trial.value,
                'duration': trial.duration.total_seconds() if trial.duration else None,
                **trial.params
            }
            data.append(row)
    
    return pd.DataFrame(data)

def compare_studies(study_names: List[str]) -> Dict[str, Any]:
    """Compare multiple studies and return summary statistics."""
    comparison = {}
    
    for study_name in study_names:
        study = load_study(study_name)
        if study:
            comparison[study_name] = {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
                'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            }
    
    return comparison

def plot_optimization_history(studies: Dict[str, optuna.Study]):
    """Plot optimization history for multiple studies."""
    fig = go.Figure()
    
    for study_name, study in studies.items():
        if study:
            values = [trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            trials = [trial.number for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            
            # Best value so far
            best_values = []
            best_so_far = float('inf')
            for value in values:
                if value < best_so_far:
                    best_so_far = value
                best_values.append(best_so_far)
            
            fig.add_trace(go.Scatter(
                x=trials,
                y=best_values,
                mode='lines+markers',
                name=f'{study_name} (best: {study.best_value:.4f})',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title='Optimization History Comparison',
        xaxis_title='Trial Number',
        yaxis_title='Best Validation Loss',
        hovermode='x unified'
    )
    
    return fig

def plot_parameter_importance(study: optuna.Study, study_name: str):
    """Plot parameter importance for a single study."""
    if not study or len(study.trials) < 10:
        print(f"Not enough trials for parameter importance analysis: {study_name}")
        return None
    
    try:
        importance = optuna.importance.get_param_importances(study)
        
        params = list(importance.keys())
        values = list(importance.values())
        
        fig = go.Figure(data=[
            go.Bar(x=values, y=params, orientation='h',
                  text=[f'{v:.3f}' for v in values],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title=f'Parameter Importance - {study_name}',
            xaxis_title='Importance',
            yaxis_title='Parameters',
            height=max(400, len(params) * 30)
        )
        
        return fig
    except Exception as e:
        print(f"Error computing parameter importance for {study_name}: {e}")
        return None

def analyze_best_hyperparameters(studies: Dict[str, optuna.Study], top_n: int = 5):
    """Analyze the best hyperparameters across studies."""
    analysis = {}
    
    for study_name, study in studies.items():
        if not study:
            continue
            
        df = study_to_dataframe(study)
        if df.empty:
            continue
        
        # Top N trials
        top_trials = df.nsmallest(top_n, 'value')
        
        # Parameter statistics
        param_stats = {}
        for param in df.columns:
            if param not in ['trial_number', 'value', 'duration']:
                if df[param].dtype in ['int64', 'float64']:
                    param_stats[param] = {
                        'mean_top': top_trials[param].mean(),
                        'std_top': top_trials[param].std(),
                        'mean_all': df[param].mean(),
                        'std_all': df[param].std(),
                        'best_value': top_trials.iloc[0][param]
                    }
                else:
                    # Categorical parameters
                    param_stats[param] = {
                        'most_common_top': top_trials[param].mode().iloc[0] if not top_trials[param].mode().empty else None,
                        'most_common_all': df[param].mode().iloc[0] if not df[param].mode().empty else None,
                        'best_value': top_trials.iloc[0][param]
                    }
        
        analysis[study_name] = {
            'best_loss': study.best_value,
            'top_trials': top_trials.to_dict('records'),
            'param_stats': param_stats
        }
    
    return analysis

def generate_recommendations(analysis: Dict[str, Any]) -> Dict[str, str]:
    """Generate hyperparameter recommendations based on analysis."""
    recommendations = {}
    
    for study_name, data in analysis.items():
        rec = []
        
        # Learning rate recommendations
        if 'lr' in data['param_stats']:
            lr_stats = data['param_stats']['lr']
            rec.append(f"Learning rate: {lr_stats['best_value']:.4f} (top trials avg: {lr_stats['mean_top']:.4f})")
        
        # Hidden channels recommendations
        if 'hidden_channels' in data['param_stats']:
            hc_stats = data['param_stats']['hidden_channels']
            rec.append(f"Hidden channels: {hc_stats['best_value']} (top trials avg: {hc_stats['mean_top']:.1f})")
        
        # Dropout recommendations
        if 'dropout' in data['param_stats']:
            dropout_stats = data['param_stats']['dropout']
            rec.append(f"Dropout: {dropout_stats['best_value']:.2f} (top trials avg: {dropout_stats['mean_top']:.2f})")
        
        # Architecture-specific recommendations
        for param in ['heads', 'num_layers', 'aggr']:
            if param in data['param_stats']:
                param_stats = data['param_stats'][param]
                if 'best_value' in param_stats:
                    rec.append(f"{param}: {param_stats['best_value']}")
        
        recommendations[study_name] = "; ".join(rec)
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter tuning results")
    parser.add_argument("--study_names", nargs='+', required=True,
                       help="Names of Optuna studies to analyze")
    parser.add_argument("--top_n", type=int, default=5,
                       help="Number of top trials to analyze")
    parser.add_argument("--output_dir", type=str, default="hyperparameter_analysis",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load studies
    studies = {}
    for study_name in args.study_names:
        study = load_study(study_name)
        if study:
            studies[study_name] = study
            print(f"Loaded study: {study_name} ({len(study.trials)} trials)")
        else:
            print(f"Failed to load study: {study_name}")
    
    if not studies:
        print("No studies loaded successfully!")
        return
    
    # 1. Basic comparison
    print("\n" + "="*50)
    print("STUDY COMPARISON")
    print("="*50)
    comparison = compare_studies(list(studies.keys()))
    for study_name, stats in comparison.items():
        print(f"\n{study_name}:")
        print(f"  Best validation loss: {stats['best_value']:.6f}")
        print(f"  Completed trials: {stats['n_complete']}/{stats['n_trials']}")
        print(f"  Best parameters: {json.dumps(stats['best_params'], indent=2)}")
    
    # 2. Detailed analysis
    print("\n" + "="*50)
    print("DETAILED ANALYSIS")
    print("="*50)
    analysis = analyze_best_hyperparameters(studies, args.top_n)
    
    # Save detailed analysis
    with open(os.path.join(args.output_dir, 'detailed_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # 3. Generate recommendations
    recommendations = generate_recommendations(analysis)
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    for study_name, rec in recommendations.items():
        print(f"\n{study_name}:")
        print(f"  {rec}")
    
    # Save recommendations
    with open(os.path.join(args.output_dir, 'recommendations.txt'), 'w') as f:
        for study_name, rec in recommendations.items():
            f.write(f"{study_name}:\n{rec}\n\n")
    
    # 4. Generate plots
    print(f"\nGenerating plots in {args.output_dir}/...")
    
    # Optimization history
    fig_history = plot_optimization_history(studies)
    fig_history.write_html(os.path.join(args.output_dir, 'optimization_history.html'))
    
    # Parameter importance plots
    for study_name, study in studies.items():
        fig_importance = plot_parameter_importance(study, study_name)
        if fig_importance:
            fig_importance.write_html(os.path.join(args.output_dir, f'parameter_importance_{study_name}.html'))
    
    print("Analysis complete! Check the output directory for detailed results.")

if __name__ == '__main__':
    main() 