"""Adaptive profiling for model performance estimation."""
import random
import time
from typing import List, Dict, Any

from ..models.base import BaseRetriever, BaseGenerator, BaseVerifier
from ..eval.pipeline import run_pipeline


def adaptive_profile(
    model_banks: Dict[str, List],
    dataset: List[Dict[str, Any]],
    probe_size: int = 15,
    seed: int = 42
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Profile each model independently on probe set.
    
    Args:
        model_banks: Dict with keys 'retriever', 'generator', 'verifier', each containing list of models
        dataset: List of query dicts
        probe_size: Number of queries to profile on
        seed: Random seed
    
    Returns:
        perf_map: dict mapping module_type -> model_name -> {avg_lat, avg_cost}
    """
    random.seed(seed)
    probe_set = random.sample(dataset, min(probe_size, len(dataset)))
    
    perf_map = {}
    
    # Profile retrievers
    if 'retriever' in model_banks:
        perf_map['retriever'] = {}
        for retriever in model_banks['retriever']:
            print(f"  Profiling retriever: {retriever.name}")
            latencies, costs = [], []
            
            for query_item in probe_set:
                query = query_item['question']
                start = time.time()
                docs = retriever.retrieve(query, k=8)
                latency_ms = (time.time() - start) * 1000
                
                latencies.append(latency_ms)
                costs.append(retriever.cost)
            
            perf_map['retriever'][retriever.name] = {
                'avg_lat': sum(latencies) / len(latencies) if latencies else 0.0,
                'avg_cost': sum(costs) / len(costs) if costs else 0.0
            }
            retriever.reset_stats()
    
    # Profile generators (need dummy docs)
    if 'generator' in model_banks:
        perf_map['generator'] = {}
        for generator in model_banks['generator']:
            print(f"  Profiling generator: {generator.name}")
            latencies, costs = [], []
            
            for query_item in probe_set:
                query = query_item['question']
                # Use dummy docs for profiling
                dummy_docs = ["This is a dummy document for profiling purposes."] * 3
                
                start = time.time()
                answer = generator.generate(query, dummy_docs)
                latency_ms = (time.time() - start) * 1000
                
                latencies.append(latency_ms)
                costs.append(generator.cost)
            
            perf_map['generator'][generator.name] = {
                'avg_lat': sum(latencies) / len(latencies) if latencies else 0.0,
                'avg_cost': sum(costs) / len(costs) if costs else 0.0
            }
            generator.reset_stats()
    
    # Profile verifiers (need dummy answer and docs)
    if 'verifier' in model_banks:
        perf_map['verifier'] = {}
        for verifier in model_banks['verifier']:
            print(f"  Profiling verifier: {verifier.name}")
            latencies, costs = [], []
            
            for query_item in probe_set:
                query = query_item['question']
                dummy_answer = "This is a dummy answer for profiling."
                dummy_docs = ["This is a dummy document for profiling purposes."] * 3
                
                start = time.time()
                verified = verifier.verify(query, dummy_answer, dummy_docs)
                latency_ms = (time.time() - start) * 1000
                
                latencies.append(latency_ms)
                costs.append(verifier.cost)
            
            perf_map['verifier'][verifier.name] = {
                'avg_lat': sum(latencies) / len(latencies) if latencies else 0.0,
                'avg_cost': sum(costs) / len(costs) if costs else 0.0
            }
            verifier.reset_stats()
    
    return perf_map

