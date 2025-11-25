"""Evaluation metrics for question answering."""
import re
import string
from typing import List, Union


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Compute exact match score."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 1.0 if len(pred_tokens) == len(gold_tokens) else 0.0
    
    common = set(pred_tokens) & set(gold_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def compute_metrics(predictions: List[str], gold_answers: List[Union[str, List[str]]]) -> tuple[float, float]:
    """
    Compute EM and F1 scores over predictions.
    
    Args:
        predictions: List of predicted answer strings
        gold_answers: List of gold answers (can be string or list of strings)
    
    Returns:
        Tuple of (EM score, F1 score) as percentages
    """
    em_scores = []
    f1_scores = []
    
    for pred, gold in zip(predictions, gold_answers):
        # Handle multiple gold answers
        if isinstance(gold, list):
            # Take best match across all gold answers
            em = max(exact_match(pred, g) for g in gold)
            f1 = max(f1_score(pred, g) for g in gold)
        else:
            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)
        
        em_scores.append(em)
        f1_scores.append(f1)
    
    avg_em = sum(em_scores) / len(em_scores) * 100 if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0.0
    
    return avg_em, avg_f1

