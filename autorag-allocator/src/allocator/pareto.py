"""Pareto frontier optimization."""
from typing import List, Dict, Any


def pareto_front(
    results: List[Dict[str, Any]],
    q_metric: str = "em",
    c_metric: str = "cost_cents"
) -> List[Dict[str, Any]]:
    """
    Find Pareto-optimal configurations.
    
    Args:
        results: List of dicts with quality and cost metrics
        q_metric: Quality metric key (e.g., "em")
        c_metric: Cost metric key (e.g., "cost_cents")
    
    Returns:
        List of non-dominated configurations
    """
    pareto = []
    
    for r in results:
        dominated = False
        for s in results:
            # s dominates r if s is better on both metrics (or equal on one, better on other)
            if (s[q_metric] >= r[q_metric] and s[c_metric] <= r[c_metric] and 
                (s[q_metric] > r[q_metric] or s[c_metric] < r[c_metric])):
                dominated = True
                break
        
        if not dominated:
            pareto.append(r)
    
    # Sort by quality (descending), then cost (ascending)
    pareto.sort(key=lambda x: (-x[q_metric], x[c_metric]))
    
    return pareto

