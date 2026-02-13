"""
Inference script for LLM-based Chain-of-Thought reasoning with Self-Consistency.
Supports both baseline (majority vote) and proposed (calibrated metacognitive) methods.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import Counter, defaultdict
from omegaconf import DictConfig, OmegaConf
import wandb

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def extract_answer(text: str) -> Optional[float]:
    """Extract numeric answer from LLM response."""
    # Look for "Final answer:" pattern
    match = re.search(r'Final answer:\s*([+-]?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        answer_str = match.group(1).replace(',', '')
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    # Fallback: extract last number in the text
    numbers = re.findall(r'([+-]?[\d,]+\.?\d*)', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    
    return None


def extract_confidence(text: str) -> Optional[float]:
    """Extract confidence score from LLM response."""
    match = re.search(r'Updated confidence:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if match:
        try:
            conf = float(match.group(1))
            return max(0.0, min(1.0, conf))  # Clamp to [0, 1]
        except ValueError:
            pass
    return None


def calibrate_binning(confidences: List[float], correctness: List[int], n_bins: int = 10, 
                      laplace_smooth: float = 1.0) -> Dict[int, float]:
    """
    Learn calibration mapping using equal-width binning with Laplace smoothing.
    
    Args:
        confidences: List of raw confidence scores
        correctness: List of binary correctness indicators (1=correct, 0=wrong)
        n_bins: Number of bins
        laplace_smooth: Laplace smoothing parameter
    
    Returns:
        Dictionary mapping bin_id -> calibrated probability
    """
    bins = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for conf, correct in zip(confidences, correctness):
        bin_id = min(int(conf * n_bins), n_bins - 1)
        bins[bin_id]['correct'] += correct
        bins[bin_id]['total'] += 1
    
    # Apply Laplace smoothing and compute calibrated probabilities
    calibration_map = {}
    for bin_id in range(n_bins):
        correct = bins[bin_id]['correct']
        total = bins[bin_id]['total']
        calibrated_prob = (correct + laplace_smooth) / (total + 2 * laplace_smooth)
        calibration_map[bin_id] = calibrated_prob
    
    return calibration_map


def calibrate_isotonic(confidences: List[float], correctness: List[int]) -> Any:
    """
    Learn calibration mapping using isotonic regression.
    
    Returns:
        Fitted isotonic regression model (or None if sklearn not available)
    """
    try:
        from sklearn.isotonic import IsotonicRegression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(confidences, correctness)
        return iso_reg
    except ImportError:
        print("Warning: sklearn not available, falling back to binning")
        return None


def apply_calibration(confidence: float, calibration_map: Optional[Dict[int, float]] = None,
                     isotonic_model: Any = None, n_bins: int = 10) -> float:
    """Apply calibration to a confidence score."""
    if isotonic_model is not None:
        return float(isotonic_model.predict([confidence])[0])
    elif calibration_map is not None:
        bin_id = min(int(confidence * n_bins), n_bins - 1)
        return calibration_map.get(bin_id, 0.5)
    else:
        return confidence


def aggregate_majority_vote(samples: List[Dict]) -> Tuple[Optional[float], Dict]:
    """Aggregate samples using simple majority vote."""
    answers = [s['answer'] for s in samples if s['answer'] is not None]
    if not answers:
        return None, {'method': 'majority_vote', 'num_valid_samples': 0}
    
    counter = Counter(answers)
    predicted_answer, vote_count = counter.most_common(1)[0]
    
    stats = {
        'method': 'majority_vote',
        'num_valid_samples': len(answers),
        'vote_count': vote_count,
        'unique_answers': len(counter)
    }
    
    return predicted_answer, stats


def aggregate_calibrated_msc(samples: List[Dict], calibration_map: Optional[Dict] = None,
                            isotonic_model: Any = None, n_bins: int = 10) -> Tuple[Optional[float], Dict]:
    """Aggregate samples using calibrated metacognitive self-consistency."""
    valid_samples = [s for s in samples if s['answer'] is not None and s['confidence'] is not None]
    if not valid_samples:
        return None, {'method': 'cal-msc', 'num_valid_samples': 0}
    
    # Apply calibration
    calibrated_probs = []
    for sample in valid_samples:
        calibrated_prob = apply_calibration(
            sample['confidence'], 
            calibration_map=calibration_map,
            isotonic_model=isotonic_model,
            n_bins=n_bins
        )
        sample['calibrated_prob'] = calibrated_prob
        calibrated_probs.append(calibrated_prob)
    
    # Aggregate using log-odds pooling
    answer_scores = defaultdict(float)
    for sample in valid_samples:
        answer = sample['answer']
        p = sample['calibrated_prob']
        # Avoid log(0) by clamping
        p = max(0.01, min(0.99, p))
        log_odds = np.log(p / (1 - p))
        answer_scores[answer] += log_odds
    
    predicted_answer = max(answer_scores, key=answer_scores.get)
    
    stats = {
        'method': 'cal-msc',
        'num_valid_samples': len(valid_samples),
        'unique_answers': len(answer_scores),
        'mean_raw_confidence': float(np.mean([s['confidence'] for s in valid_samples])),
        'mean_calibrated_prob': float(np.mean(calibrated_probs)),
        'predicted_score': float(answer_scores[predicted_answer])
    }
    
    return predicted_answer, stats


def query_llm(client: Any, question: str, cfg: DictConfig, seed: Optional[int] = None) -> str:
    """Query LLM with a question."""
    if client is None:
        raise RuntimeError("OpenAI client not initialized")
    
    messages = [
        {"role": "system", "content": cfg.prompt.system_message},
        {"role": "user", "content": f"{cfg.prompt.cot_instruction}\n\nQuestion: {question}"}
    ]
    
    response = client.chat.completions.create(
        model=cfg.model.name,
        messages=messages,
        temperature=cfg.method.temperature,
        max_tokens=cfg.model.max_tokens,
        seed=seed
    )
    
    return response.choices[0].message.content


def run_inference(cfg: DictConfig, dataset: List[Dict], mode: str = 'main') -> Dict:
    """
    Run inference on dataset.
    
    Args:
        cfg: Hydra config
        dataset: List of dataset examples
        mode: Execution mode (main or sanity_check)
    
    Returns:
        Dictionary of results
    """
    # Initialize OpenAI client
    if OpenAI is None:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Get method configuration
    method_type = cfg.method.type
    k_samples = cfg.method.k_samples
    
    # Adjust for sanity check mode
    if mode == 'sanity_check':
        k_samples = min(k_samples, 3)  # Reduce samples for sanity check
    
    # Split dataset into calibration and evaluation sets
    n_cal = cfg.dataset.n_calibration
    if method_type == 'proposed':
        calibration_set = dataset[:n_cal]
        evaluation_set = dataset[n_cal:]
    else:
        # Baseline doesn't use calibration, but we still split for consistency
        calibration_set = []
        evaluation_set = dataset[n_cal:]
    
    # Run calibration phase for proposed method
    calibration_map = None
    isotonic_model = None
    
    if method_type == 'proposed' and len(calibration_set) > 0:
        print(f"Running calibration on {len(calibration_set)} questions...")
        cal_confidences = []
        cal_correctness = []
        
        for idx, example in enumerate(calibration_set):
            question = example['question']
            gold_answer = example['answer']
            
            for sample_idx in range(k_samples):
                try:
                    response = query_llm(client, question, cfg, seed=idx * k_samples + sample_idx)
                    answer = extract_answer(response)
                    confidence = extract_confidence(response)
                    
                    if answer is not None and confidence is not None:
                        is_correct = 1 if abs(answer - gold_answer) < 0.01 else 0
                        cal_confidences.append(confidence)
                        cal_correctness.append(is_correct)
                except Exception as e:
                    print(f"Error in calibration question {idx}, sample {sample_idx}: {e}")
                    continue
        
        print(f"Collected {len(cal_confidences)} calibration samples")
        
        # Learn calibration mapping
        if cfg.method.calibration_method == 'isotonic':
            isotonic_model = calibrate_isotonic(cal_confidences, cal_correctness)
            if isotonic_model is None:
                calibration_map = calibrate_binning(
                    cal_confidences, cal_correctness,
                    n_bins=cfg.method.n_bins,
                    laplace_smooth=cfg.method.laplace_smoothing
                )
        else:
            calibration_map = calibrate_binning(
                cal_confidences, cal_correctness,
                n_bins=cfg.method.n_bins,
                laplace_smooth=cfg.method.laplace_smoothing
            )
    
    # Run inference on evaluation set
    print(f"Running inference on {len(evaluation_set)} questions...")
    results = []
    num_correct = 0
    
    for idx, example in enumerate(evaluation_set):
        question = example['question']
        gold_answer = example['answer']
        
        # Sample K completions
        samples = []
        for sample_idx in range(k_samples):
            try:
                response = query_llm(client, question, cfg, seed=(n_cal + idx) * k_samples + sample_idx)
                answer = extract_answer(response)
                confidence = extract_confidence(response) if method_type == 'proposed' else None
                
                samples.append({
                    'response': response,
                    'answer': answer,
                    'confidence': confidence
                })
            except Exception as e:
                print(f"Error in question {idx}, sample {sample_idx}: {e}")
                samples.append({
                    'response': None,
                    'answer': None,
                    'confidence': None
                })
        
        # Aggregate answers
        if method_type == 'proposed':
            predicted_answer, agg_stats = aggregate_calibrated_msc(
                samples, 
                calibration_map=calibration_map,
                isotonic_model=isotonic_model,
                n_bins=cfg.method.n_bins
            )
        else:
            predicted_answer, agg_stats = aggregate_majority_vote(samples)
        
        # Check correctness
        is_correct = False
        if predicted_answer is not None:
            is_correct = abs(predicted_answer - gold_answer) < 0.01
            if is_correct:
                num_correct += 1
        
        results.append({
            'question_idx': idx,
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'samples': samples,
            'aggregation_stats': agg_stats
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(evaluation_set)} questions, accuracy: {num_correct / (idx + 1):.3f}")
    
    # Compute final metrics
    accuracy = num_correct / len(evaluation_set) if len(evaluation_set) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'num_correct': num_correct,
        'num_total': len(evaluation_set),
        'results': results,
        'calibration_map': calibration_map if calibration_map is not None else {},
        'config': OmegaConf.to_container(cfg, resolve=True)
    }


def main(cfg: DictConfig):
    """Main inference execution."""
    from src.preprocess import load_dataset
    
    # Get mode
    mode = cfg.get('mode', 'main')
    
    # Set wandb project for sanity check
    wandb_project = cfg.wandb.project
    if mode == 'sanity_check':
        wandb_project = f"{cfg.wandb.project}-sanity"
    
    # Initialize WandB
    if cfg.wandb.mode == 'online':
        wandb.init(
            entity=cfg.wandb.entity,
            project=wandb_project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume='allow'
        )
        print(f"WandB run: {wandb.run.url}")
    
    # Load dataset
    dataset = load_dataset(cfg, mode=mode)
    print(f"Loaded {len(dataset)} examples from {cfg.dataset.name}")
    
    # Run inference
    results = run_inference(cfg, dataset, mode=mode)
    
    # Log metrics to WandB
    if cfg.wandb.mode == 'online':
        wandb.log({
            'accuracy': results['accuracy'],
            'num_correct': results['num_correct'],
            'num_total': results['num_total']
        })
        wandb.summary['accuracy'] = results['accuracy']
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFinal accuracy: {results['accuracy']:.4f} ({results['num_correct']}/{results['num_total']})")
    print(f"Results saved to {results_dir}")
    
    # Sanity validation for inference tasks
    if mode == 'sanity_check':
        num_samples = results['num_total']
        
        # Check: at least 5 samples processed
        samples_ok = num_samples >= 5
        
        # Check: all outputs are valid (not all identical)
        unique_predictions = len(set(
            r['predicted_answer'] for r in results['results'] 
            if r['predicted_answer'] is not None
        ))
        outputs_valid = unique_predictions > 1
        
        # Check: accuracy is finite
        accuracy_finite = np.isfinite(results['accuracy'])
        
        # Overall validation
        passed = samples_ok and outputs_valid and accuracy_finite
        
        # Print validation summary
        summary = {
            'samples': num_samples,
            'outputs_valid': outputs_valid,
            'outputs_unique': unique_predictions,
            'accuracy': results['accuracy']
        }
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
        
        if passed:
            print("SANITY_VALIDATION: PASS")
        else:
            if not samples_ok:
                print("SANITY_VALIDATION: FAIL reason=insufficient_samples")
            elif not outputs_valid:
                print("SANITY_VALIDATION: FAIL reason=invalid_outputs")
            elif not accuracy_finite:
                print("SANITY_VALIDATION: FAIL reason=non_finite_metrics")
            else:
                print("SANITY_VALIDATION: FAIL reason=unknown")
    
    if cfg.wandb.mode == 'online':
        wandb.finish()


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(config_path="../config", config_name="config", version_base=None)
    def hydra_main(cfg: DictConfig):
        main(cfg)
    
    hydra_main()
