"""
Dataset loading and preprocessing for GSM8K and SVAMP.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
from datasets import load_dataset as hf_load_dataset


def extract_numeric_answer(answer_text: str) -> float:
    """Extract numeric answer from answer text."""
    # Look for #### pattern (common in GSM8K)
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', answer_text)
    if match:
        return float(match.group(1).replace(',', ''))
    
    # Fallback: extract last number
    numbers = re.findall(r'([+-]?[\d,]+\.?\d*)', answer_text)
    if numbers:
        return float(numbers[-1].replace(',', ''))
    
    raise ValueError(f"Could not extract numeric answer from: {answer_text}")


def load_gsm8k(split: str = 'test', n_total: int = 300, cache_dir: str = '.cache') -> List[Dict]:
    """Load GSM8K dataset."""
    dataset = hf_load_dataset('gsm8k', 'main', split=split, cache_dir=cache_dir)
    
    examples = []
    for idx, item in enumerate(dataset):
        if idx >= n_total:
            break
        
        try:
            answer = extract_numeric_answer(item['answer'])
            examples.append({
                'question': item['question'],
                'answer': answer,
                'raw_answer': item['answer']
            })
        except Exception as e:
            print(f"Warning: Could not parse example {idx}: {e}")
            continue
    
    return examples


def load_svamp(split: str = 'test', n_total: int = 300, cache_dir: str = '.cache') -> List[Dict]:
    """Load SVAMP dataset."""
    # SVAMP is typically loaded from a JSON file or HuggingFace dataset
    # Using ChilleD/SVAMP dataset from HuggingFace
    try:
        dataset = hf_load_dataset('ChilleD/SVAMP', split='test', cache_dir=cache_dir)
    except Exception:
        # Fallback to another source
        print("Warning: Using alternative SVAMP source")
        dataset = hf_load_dataset('svamp', split='test', cache_dir=cache_dir)
    
    examples = []
    for idx, item in enumerate(dataset):
        if idx >= n_total:
            break
        
        try:
            # SVAMP typically has 'Body', 'Question', and 'Answer' fields
            if 'Body' in item and 'Question' in item:
                question = f"{item['Body']} {item['Question']}"
            elif 'question' in item:
                question = item['question']
            else:
                question = str(item)
            
            # Extract answer
            if 'Answer' in item:
                answer = float(item['Answer'])
            elif 'answer' in item:
                answer = float(item['answer'])
            else:
                raise ValueError("No answer field found")
            
            examples.append({
                'question': question,
                'answer': answer,
                'raw_answer': str(answer)
            })
        except Exception as e:
            print(f"Warning: Could not parse example {idx}: {e}")
            continue
    
    return examples


def load_dataset(cfg: DictConfig, mode: str = 'main') -> List[Dict]:
    """
    Load dataset based on configuration.
    
    Args:
        cfg: Hydra configuration
        mode: Execution mode (main or sanity_check)
    
    Returns:
        List of dataset examples
    """
    dataset_name = cfg.dataset.name.lower()
    split = cfg.dataset.split
    n_total = cfg.dataset.n_total
    cache_dir = cfg.get('inference', {}).get('cache_dir', '.cache')
    
    print(f"Loading {dataset_name} dataset (split={split}, n_total={n_total})...")
    
    if dataset_name == 'gsm8k':
        return load_gsm8k(split=split, n_total=n_total, cache_dir=cache_dir)
    elif dataset_name == 'svamp':
        return load_svamp(split=split, n_total=n_total, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == '__main__':
    # Test loading
    from omegaconf import OmegaConf
    
    # Test GSM8K
    cfg = OmegaConf.create({
        'dataset': {
            'name': 'gsm8k',
            'split': 'test',
            'n_total': 10
        }
    })
    examples = load_dataset(cfg)
    print(f"Loaded {len(examples)} GSM8K examples")
    print(f"Example: {examples[0]}")
