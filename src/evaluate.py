"""
Evaluation script for fetching results from WandB and generating comparison plots.
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def _convert_to_json_serializable(obj):
    """
    Recursively convert WandB objects to JSON-serializable types.
    
    Args:
        obj: Any object from WandB (e.g., SummarySubDict, config dict, etc.)
    
    Returns:
        JSON-serializable version of the object
    """
    # [VALIDATOR FIX - Attempt 3]
    # [PROBLEM]: Error 'to_dict' when fetching W&B runs
    # [CAUSE]: Some W&B objects raise AttributeError when accessing 'to_dict', even with getattr.
    #          The error message is just the attribute name 'to_dict'.
    # [FIX]: Use try-except to safely attempt to_dict conversion, avoiding hasattr which can trigger the error
    #
    # [OLD CODE]:
    # if isinstance(obj, dict):
    #     # Recursively convert all dict values
    #     return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    # elif isinstance(obj, (list, tuple)):
    #     # Recursively convert all list/tuple elements
    #     return [_convert_to_json_serializable(item) for item in obj]
    # elif isinstance(obj, (int, float, str, bool, type(None))):
    #     # Already JSON-serializable
    #     return obj
    # elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict', None)):
    #     # If object has a callable to_dict method, use it
    #     return _convert_to_json_serializable(obj.to_dict())
    # elif hasattr(obj, '__dict__'):
    #     # For objects with __dict__, convert to dict
    #     return _convert_to_json_serializable(obj.__dict__)
    # else:
    #     # Fallback: convert to string
    #     return str(obj)
    #
    # [NEW CODE]:
    # [VALIDATOR FIX - Attempt 4]
    # [PROBLEM]: WandB objects raise AttributeError when accessing to_dict, even in try-except
    # [CAUSE]: Some WandB objects (SummarySubDict, Config) have dict-like interface but fail on to_dict access
    # [FIX]: Check for items() method first (dict-like), then try other conversions more carefully
    #
    # [OLD CODE]:
    # if isinstance(obj, dict):
    #     return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    # elif isinstance(obj, (list, tuple)):
    #     return [_convert_to_json_serializable(item) for item in obj]
    # elif isinstance(obj, (int, float, str, bool, type(None))):
    #     return obj
    # else:
    #     try:
    #         to_dict_method = obj.to_dict
    #         if callable(to_dict_method):
    #             return _convert_to_json_serializable(to_dict_method())
    #     except (AttributeError, TypeError):
    #         pass
    #     try:
    #         if hasattr(obj, '__dict__'):
    #             return _convert_to_json_serializable(obj.__dict__)
    #     except (AttributeError, TypeError):
    #         pass
    #     return str(obj)
    #
    # [NEW CODE]:
    if isinstance(obj, dict):
        # Recursively convert all dict values
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively convert all list/tuple elements
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        # Already JSON-serializable
        return obj
    else:
        # Try dict-like conversion first (for WandB objects with items() method)
        try:
            if hasattr(obj, 'items') and callable(obj.items):
                # Treat as dict-like object
                return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
        except Exception:
            pass
        
        # Try keys() method for dict-like objects without items()
        try:
            if hasattr(obj, 'keys') and callable(obj.keys):
                result = {}
                for k in obj.keys():
                    try:
                        result[k] = _convert_to_json_serializable(obj[k])
                    except Exception:
                        result[k] = str(obj[k])
                return result
        except Exception:
            pass
        
        # Try __dict__ attribute
        try:
            if hasattr(obj, '__dict__'):
                return _convert_to_json_serializable(obj.__dict__)
        except Exception:
            pass
        
        # Fallback: convert to string
        return str(obj)


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.
    
    Returns:
        Dictionary containing config, summary, and history
    """
    # [VALIDATOR FIX - Attempt 4]
    # [PROBLEM]: Error 'to_dict' when accessing WandB run objects
    # [CAUSE]: WandB's SummarySubDict and Config objects raise AttributeError('to_dict') 
    #          when accessed, even through try-except blocks
    # [FIX]: Manually extract data from WandB objects using safe key iteration with getattr
    #
    # [OLD CODE]:
    # api = wandb.Api()
    # run = api.run(f"{entity}/{project}/{run_id}")
    # config = _convert_to_json_serializable(run.config)
    # summary = _convert_to_json_serializable(run.summary)
    #
    # [NEW CODE]:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Manually extract config data to avoid AttributeError
    config = {}
    try:
        # WandB config objects support iteration but may fail on dict() conversion
        for key in run.config.keys():
            try:
                config[key] = run.config[key]
            except Exception as e:
                print(f"Warning: Could not fetch config key {key}: {e}")
                config[key] = None
    except Exception as e:
        print(f"Warning: Could not iterate config: {e}")
    
    # Manually extract summary data to avoid AttributeError  
    summary = {}
    try:
        # WandB summary objects support iteration but may fail on dict() conversion
        for key in run.summary.keys():
            try:
                summary[key] = run.summary[key]
            except Exception as e:
                print(f"Warning: Could not fetch summary key {key}: {e}")
                summary[key] = None
    except Exception as e:
        print(f"Warning: Could not iterate summary: {e}")
    
    # Convert extracted data to JSON-serializable types
    config = _convert_to_json_serializable(config)
    summary = _convert_to_json_serializable(summary)
    
    # [VALIDATOR FIX - Attempt 5]
    # [PROBLEM]: History rows are converted to strings instead of dicts
    # [CAUSE]: _convert_to_json_serializable fails on WandB history row objects and falls back to str()
    # [FIX]: Manually iterate over row keys and extract values to build dict, similar to config/summary
    #
    # [OLD CODE]:
    # history = []
    # try:
    #     for row in run.history():
    #         history.append(_convert_to_json_serializable(row))
    # except Exception as e:
    #     print(f"Warning: Could not fetch history for {run_id}: {e}")
    #
    # [NEW CODE]:
    # Get history (may be empty for single-step runs)
    history = []
    try:
        for row in run.history():
            # Manually extract row data to avoid conversion issues
            row_dict = {}
            try:
                # Try to get keys from the row object
                if hasattr(row, 'keys') and callable(row.keys):
                    for key in row.keys():
                        try:
                            row_dict[key] = row[key]
                        except Exception:
                            row_dict[key] = None
                else:
                    # Fallback: try to iterate as dict
                    for key in row:
                        try:
                            row_dict[key] = row[key]
                        except Exception:
                            row_dict[key] = None
            except Exception as e:
                print(f"Warning: Could not extract row data: {e}")
                continue
            
            # Convert to JSON-serializable
            history.append(_convert_to_json_serializable(row_dict))
    except Exception as e:
        print(f"Warning: Could not fetch history for {run_id}: {e}")
    
    return {
        'config': config,
        'summary': summary,
        'history': history,
        'run_id': run_id,
        'name': run.name,
        'url': run.url
    }


def export_run_metrics(run_data: Dict, output_dir: Path):
    """Export per-run metrics to JSON."""
    metrics = {
        'run_id': run_data['run_id'],
        'name': run_data['name'],
        'url': run_data['url'],
        'config': run_data['config'],
        'summary': run_data['summary']
    }
    
    output_file = output_dir / 'metrics.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics to {output_file}")
    return output_file


def create_run_figures(run_data: Dict, output_dir: Path):
    """Create per-run visualization figures."""
    history = run_data['history']
    
    if not history:
        print(f"No history data for {run_data['run_id']}, skipping figures")
        return []
    
    created_files = []
    
    # Extract numeric metrics from history
    numeric_metrics = {}
    for row in history:
        for key, value in row.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                if key not in numeric_metrics:
                    numeric_metrics[key] = []
                numeric_metrics[key].append(value)
    
    # Create a figure for each metric
    for metric_name, values in numeric_metrics.items():
        if metric_name == '_step' or metric_name.startswith('_'):
            continue
        
        plt.figure(figsize=(8, 6))
        plt.plot(values, marker='o')
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f"{run_data['run_id']}: {metric_name}")
        plt.grid(True, alpha=0.3)
        
        output_file = output_dir / f"{metric_name.replace('/', '_')}.pdf"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        created_files.append(output_file)
        print(f"Created figure: {output_file}")
    
    return created_files


def aggregate_metrics(runs_data: List[Dict]) -> Dict:
    """
    Aggregate metrics across runs.
    
    Returns:
        Dictionary with aggregated metrics and comparisons
    """
    # Collect metrics by run_id
    metrics_by_run = {}
    for run_data in runs_data:
        run_id = run_data['run_id']
        summary = run_data['summary']
        
        # Extract primary metric (accuracy for this task)
        primary_metric = summary.get('accuracy', None)
        
        metrics_by_run[run_id] = {
            'accuracy': primary_metric,
            'all_metrics': summary
        }
    
    # Identify proposed vs baseline runs
    proposed_runs = [rid for rid in metrics_by_run if 'proposed' in rid]
    baseline_runs = [rid for rid in metrics_by_run if 'comparative' in rid or 'baseline' in rid]
    
    # Find best proposed and best baseline
    best_proposed = None
    best_proposed_acc = -float('inf')
    for run_id in proposed_runs:
        acc = metrics_by_run[run_id]['accuracy']
        if acc is not None and acc > best_proposed_acc:
            best_proposed_acc = acc
            best_proposed = run_id
    
    best_baseline = None
    best_baseline_acc = -float('inf')
    for run_id in baseline_runs:
        acc = metrics_by_run[run_id]['accuracy']
        if acc is not None and acc > best_baseline_acc:
            best_baseline_acc = acc
            best_baseline = run_id
    
    # Compute gap
    gap = None
    if best_proposed_acc != -float('inf') and best_baseline_acc != -float('inf'):
        gap = best_proposed_acc - best_baseline_acc
    
    aggregated = {
        'primary_metric': 'accuracy',
        'metrics_by_run': metrics_by_run,
        'best_proposed': best_proposed,
        'best_proposed_accuracy': best_proposed_acc if best_proposed_acc != -float('inf') else None,
        'best_baseline': best_baseline,
        'best_baseline_accuracy': best_baseline_acc if best_baseline_acc != -float('inf') else None,
        'gap': gap
    }
    
    return aggregated


def create_comparison_figures(runs_data: List[Dict], output_dir: Path):
    """
    Create comparison figures overlaying all runs.
    """
    created_files = []
    
    # Group runs by method for comparison
    runs_by_method = {}
    for run_data in runs_data:
        method_type = run_data['config'].get('method', {}).get('type', 'unknown')
        if method_type not in runs_by_method:
            runs_by_method[method_type] = []
        runs_by_method[method_type].append(run_data)
    
    # Create bar chart comparing accuracy across runs
    run_ids = [rd['run_id'] for rd in runs_data]
    accuracies = [rd['summary'].get('accuracy', 0) for rd in runs_data]
    
    plt.figure(figsize=(12, 6))
    colors = ['green' if 'proposed' in rid else 'blue' for rid in run_ids]
    bars = plt.bar(range(len(run_ids)), accuracies, color=colors, alpha=0.7)
    plt.xlabel('Run ID')
    plt.ylabel('Accuracy')
    plt.title('Comparison: Accuracy across runs')
    plt.xticks(range(len(run_ids)), run_ids, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend([bars[0], bars[-1]], ['Proposed' if 'proposed' in run_ids[0] else 'Baseline', 
                                      'Baseline' if 'comparative' in run_ids[-1] else 'Proposed'])
    
    output_file = output_dir / 'comparison_accuracy.pdf'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    created_files.append(output_file)
    print(f"Created comparison figure: {output_file}")
    
    # If runs have history data, create line plots
    # (For inference tasks, history may be minimal)
    all_metrics = set()
    for run_data in runs_data:
        for row in run_data['history']:
            all_metrics.update(row.keys())
    
    # Filter to numeric metrics
    numeric_metrics = []
    for metric in all_metrics:
        if metric.startswith('_'):
            continue
        # Check if at least one run has numeric values
        for run_data in runs_data:
            for row in run_data['history']:
                if metric in row and isinstance(row[metric], (int, float)):
                    numeric_metrics.append(metric)
                    break
            if metric in numeric_metrics:
                break
    
    # Create overlay plots for each metric
    for metric_name in numeric_metrics:
        plt.figure(figsize=(10, 6))
        
        for run_data in runs_data:
            run_id = run_data['run_id']
            values = [row.get(metric_name) for row in run_data['history'] 
                     if metric_name in row and isinstance(row[metric_name], (int, float))]
            
            if values:
                color = 'green' if 'proposed' in run_id else 'blue'
                linestyle = '-' if 'proposed' in run_id else '--'
                plt.plot(values, label=run_id, color=color, linestyle=linestyle, marker='o')
        
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f'Comparison: {metric_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        output_file = output_dir / f"comparison_{metric_name.replace('/', '_')}.pdf"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        created_files.append(output_file)
        print(f"Created comparison figure: {output_file}")
    
    return created_files


def main():
    """Main evaluation script."""
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: evaluate.py called with Hydra-style args (key=value) but uses argparse (--key value)
    # [CAUSE]: Workflow passes results_dir=".research/results" run_ids='[...]' without -- flags
    # [FIX]: Parse both argparse-style and Hydra-style arguments
    #
    # [OLD CODE]:
    # parser = argparse.ArgumentParser(description='Evaluate experiment runs from WandB')
    # parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    # parser.add_argument('--run_ids', type=str, required=True, help='JSON string list of run IDs')
    # parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (or use WANDB_ENTITY env)')
    # parser.add_argument('--wandb_project', type=str, default=None, help='WandB project (or use WANDB_PROJECT env)')
    # args = parser.parse_args()
    #
    # [NEW CODE]:
    import sys
    
    # Check if arguments are in Hydra format (key=value)
    hydra_args = {}
    regular_args = []
    for arg in sys.argv[1:]:
        if '=' in arg and not arg.startswith('--'):
            # Hydra-style: key=value
            key, value = arg.split('=', 1)
            hydra_args[key] = value
        else:
            regular_args.append(arg)
    
    # If we have Hydra-style args, convert them to argparse format
    if hydra_args:
        sys.argv = [sys.argv[0]]
        for key, value in hydra_args.items():
            sys.argv.extend([f'--{key}', value])
    
    parser = argparse.ArgumentParser(description='Evaluate experiment runs from WandB')
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    parser.add_argument('--run_ids', type=str, required=False, default=None, help='JSON string list of run IDs (or auto-discover)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (or use WANDB_ENTITY env)')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project (or use WANDB_PROJECT env)')
    
    args = parser.parse_args()
    
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: WandB entity and project not provided to evaluate.py during visualization stage
    # [CAUSE]: The workflow file calls evaluate.py without passing wandb_entity/wandb_project arguments,
    #          and does not set WANDB_ENTITY/WANDB_PROJECT environment variables
    # [FIX]: Read wandb config from config/config.yaml if not provided via args or env vars
    #
    # [OLD CODE]:
    # entity = args.wandb_entity or os.getenv('WANDB_ENTITY')
    # project = args.wandb_project or os.getenv('WANDB_PROJECT')
    # 
    # if not entity or not project:
    #     raise ValueError("WandB entity and project must be specified via arguments or environment variables")
    #
    # [NEW CODE]:
    entity = args.wandb_entity or os.getenv('WANDB_ENTITY')
    project = args.wandb_project or os.getenv('WANDB_PROJECT')
    
    # Fallback: read from config.yaml if not provided
    if not entity or not project:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if not entity and config.get('wandb', {}).get('entity'):
                    entity = config['wandb']['entity']
                    print(f"Using WandB entity from config.yaml: {entity}")
                if not project and config.get('wandb', {}).get('project'):
                    project = config['wandb']['project']
                    print(f"Using WandB project from config.yaml: {project}")
    
    if not entity or not project:
        raise ValueError("WandB entity and project must be specified via arguments, environment variables, or config.yaml")
    
    print(f"WandB: {entity}/{project}")
    
    # Parse run_ids or auto-discover
    if args.run_ids:
        run_ids = json.loads(args.run_ids)
        print(f"Evaluating {len(run_ids)} runs: {run_ids}")
    else:
        # Auto-discover runs from WandB project
        print("No run_ids provided, auto-discovering from WandB project...")
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        # Filter to finished runs from main experiments (not sanity checks)
        run_ids = []
        for run in runs:
            # Skip sanity check runs
            if 'sanity' in run.name.lower() or run.config.get('mode') == 'sanity_check':
                continue
            # Only include runs with completed state
            if run.state == 'finished':
                run_ids.append(run.id)
        
        print(f"Auto-discovered {len(run_ids)} runs: {run_ids}")
        
        if not run_ids:
            print("WARNING: No finished runs found in WandB project. Nothing to visualize.")
            print("Creating empty results directory.")
            results_dir = Path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            return
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch runs from WandB
    runs_data = []
    for run_id in run_ids:
        print(f"\nFetching run: {run_id}")
        try:
            run_data = fetch_wandb_run(entity, project, run_id)
            runs_data.append(run_data)
            
            # Export per-run metrics
            run_dir = results_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            export_run_metrics(run_data, run_dir)
            
            # Create per-run figures
            create_run_figures(run_data, run_dir)
        except Exception as e:
            print(f"Error fetching run {run_id}: {e}")
            continue
    
    if not runs_data:
        print("No runs successfully fetched")
        return
    
    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregated = aggregate_metrics(runs_data)
    
    # Export aggregated metrics
    comparison_dir = results_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_file = comparison_dir / 'aggregated_metrics.json'
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Exported aggregated metrics to {aggregated_file}")
    
    # Create comparison figures
    print("\nCreating comparison figures...")
    comparison_files = create_comparison_figures(runs_data, comparison_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Primary metric: {aggregated['primary_metric']}")
    print(f"Best proposed: {aggregated['best_proposed']} ({aggregated['best_proposed_accuracy']:.4f})")
    print(f"Best baseline: {aggregated['best_baseline']} ({aggregated['best_baseline_accuracy']:.4f})")
    if aggregated['gap'] is not None:
        print(f"Gap (proposed - baseline): {aggregated['gap']:.4f}")
    print("="*60)
    
    print("\nGenerated files:")
    for run_id in run_ids:
        print(f"  {results_dir / run_id / 'metrics.json'}")
    print(f"  {aggregated_file}")
    for comp_file in comparison_files:
        print(f"  {comp_file}")


if __name__ == '__main__':
    main()
