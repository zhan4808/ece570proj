"""Budget-aware triplet selection."""
from typing import List, Dict, Any


def select_triplets(
    perf_map: Dict[str, Dict[str, Dict[str, float]]],
    budget_lat: float,
    budget_cost: float,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Select top-k triplets satisfying budget constraints.
    
    Args:
        perf_map: Output from adaptive_profile
        budget_lat: Max latency in ms
        budget_cost: Max cost in cents
        top_k: Number of candidates to return
    
    Returns:
        List of triplet configs: [{'R': name, 'G': name, 'V': name, 'est_lat': float, 'est_cost': float}]
    """
    candidates = []
    
    retrievers = perf_map.get('retriever', {})
    generators = perf_map.get('generator', {})
    verifiers = perf_map.get('verifier', {})
    
    for R_name, R_perf in retrievers.items():
        for G_name, G_perf in generators.items():
            for V_name, V_perf in verifiers.items():
                est_lat = (R_perf['avg_lat'] + 
                          G_perf['avg_lat'] + 
                          V_perf['avg_lat'])
                est_cost = (R_perf['avg_cost'] + 
                           G_perf['avg_cost'] + 
                           V_perf['avg_cost'])
                
                if est_lat <= budget_lat and est_cost <= budget_cost:
                    candidates.append({
                        'R': R_name,
                        'G': G_name,
                        'V': V_name,
                        'est_lat': est_lat,
                        'est_cost': est_cost
                    })
    
    # Sort by cost and return top-k cheapest
    candidates.sort(key=lambda x: x['est_cost'])
    return candidates[:top_k]

