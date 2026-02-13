"""
Main orchestrator for running experiments.
Handles mode-specific configurations and invokes the appropriate execution script.
"""

import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main orchestrator for experiment execution.
    
    This script:
    1. Loads Hydra configuration
    2. Applies mode-specific overrides (sanity_check, main, pilot)
    3. Invokes the appropriate execution script (inference.py for this task)
    """
    
    # Get execution mode
    mode = cfg.get('mode', 'main')
    print(f"Running in {mode} mode")
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Results directory: {cfg.results_dir}")
    
    # Apply mode-specific overrides
    if mode == 'sanity_check':
        # For inference tasks, reduce dataset size
        cfg.dataset.n_total = 20
        cfg.dataset.n_calibration = 5
        cfg.dataset.n_evaluation = 15
        cfg.method.k_samples = 3
        cfg.wandb.mode = 'online'
        print("Sanity check mode: Using reduced dataset and samples")
    
    # Determine task type based on config
    # This is an inference-only task (prompt tuning, no training)
    task_type = 'inference'
    
    print(f"Task type: {task_type}")
    
    # Invoke the appropriate script
    if task_type == 'inference':
        # Run inference as a subprocess
        cmd = [
            sys.executable, '-u', '-m', 'src.inference'
        ]
        
        # Pass all config as Hydra CLI args
        # Note: We pass the entire config through environment or by re-running with same args
        # For simplicity, we import and call directly
        print("Starting inference...")
        from src.inference import main as inference_main
        inference_main(cfg)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    print(f"Execution complete for run {cfg.run.run_id}")


if __name__ == '__main__':
    main()
