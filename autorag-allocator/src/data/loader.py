"""Dataset loading utilities for NQ-Open."""
import random
from typing import List, Dict, Any
from datasets import load_dataset


def load_nq_open(n_samples: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load Natural Questions Open dataset subset.
    
    Args:
        n_samples: Number of samples to return
        seed: Random seed for reproducibility
    
    Returns:
        List of dicts with keys: 'question', 'answer' (list), 'context' (optional)
    """
    random.seed(seed)
    
    # Load NQ-Open dataset from HuggingFace
    dataset = load_dataset("nq_open", split="train")
    
    # Convert to list and sample
    all_data = list(dataset)
    sampled = random.sample(all_data, min(n_samples, len(all_data)))
    
    # Normalize format
    results = []
    for item in sampled:
        # NQ-Open format: 'question' and 'answer' fields
        question = item.get('question', '')
        
        # Handle multiple answer formats
        answer = item.get('answer', [])
        if isinstance(answer, str):
            answer = [answer]
        elif not isinstance(answer, list):
            answer = [str(answer)]
        
        # Extract context if available (some items have 'context' field)
        context = item.get('context', '')
        
        results.append({
            'question': question,
            'answer': answer,
            'context': context
        })
    
    return results


def load_hotpotqa(n_samples: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load HotpotQA dataset subset.
    
    Args:
        n_samples: Number of samples to return
        seed: Random seed for reproducibility
    
    Returns:
        List of dicts with keys: 'question', 'answer' (list), 'context' (optional)
    """
    random.seed(seed)
    
    # Load HotpotQA dataset from HuggingFace
    dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
    
    # Convert to list and sample
    all_data = list(dataset)
    sampled = random.sample(all_data, min(n_samples, len(all_data)))
    
    # Normalize format
    results = []
    for item in sampled:
        question = item.get('question', '')
        
        # HotpotQA format: 'answer' field
        answer = item.get('answer', '')
        if isinstance(answer, str):
            answer = [answer]
        elif not isinstance(answer, list):
            answer = [str(answer)]
        
        # HotpotQA has 'context' field with supporting paragraphs
        context = item.get('context', [])
        if isinstance(context, list) and len(context) > 0:
            # Join context paragraphs
            context = ' '.join([para.get('text', '') if isinstance(para, dict) else str(para) for para in context])
        else:
            context = ''
        
        results.append({
            'question': question,
            'answer': answer,
            'context': context
        })
    
    return results

