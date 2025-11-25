"""RAG pipeline executor."""
import time
import random
import numpy as np
from typing import List, Dict, Any

from ..models.base import BaseRetriever, BaseGenerator, BaseVerifier
from .metrics import compute_metrics


def run_pipeline(
    retriever: BaseRetriever,
    generator: BaseGenerator,
    verifier: BaseVerifier,
    dataset: List[Dict[str, Any]],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run full RAG pipeline on dataset.
    
    Args:
        retriever: Retriever instance
        generator: Generator instance
        verifier: Verifier instance
        dataset: List of query dicts with 'question', 'answer', 'context'
        seed: Random seed for reproducibility
    
    Returns:
        Dict with keys: em, f1, latency_ms, cost_cents, predictions
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Reset model statistics
    retriever.reset_stats()
    generator.reset_stats()
    verifier.reset_stats()
    
    predictions = []
    total_time = 0.0
    total_cost = 0.0
    
    for item in dataset:
        query = item['question']
        gold_answer = item['answer']
        
        # Step 1: Retrieve
        start = time.time()
        docs = retriever.retrieve(query, k=8)
        retrieve_time = time.time() - start
        
        # Step 2: Generate
        start = time.time()
        answer = generator.generate(query, docs)
        generate_time = time.time() - start
        
        # Step 3: Verify
        start = time.time()
        verified = verifier.verify(query, answer, docs)
        verify_time = time.time() - start
        
        # Track metrics
        predictions.append({
            'query': query,
            'answer': answer,
            'gold': gold_answer,
            'verified': verified,
            'docs': docs
        })
        
        total_time += (retrieve_time + generate_time + verify_time)
        total_cost += (retriever.cost + generator.cost + verifier.cost)
    
    # Compute EM/F1
    pred_answers = [p['answer'] for p in predictions]
    gold_answers = [p['gold'] for p in predictions]
    em, f1 = compute_metrics(pred_answers, gold_answers)
    
    avg_latency = (total_time * 1000) / len(dataset) if dataset else 0.0  # ms per query
    avg_cost = total_cost / len(dataset) if dataset else 0.0  # Already in cents
    
    return {
        'em': em,
        'f1': f1,
        'latency_ms': avg_latency,
        'cost_cents': avg_cost,
        'predictions': predictions
    }

