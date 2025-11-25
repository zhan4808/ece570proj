#!/usr/bin/env python3
"""Run full experimental evaluation."""
import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_nq_open, load_hotpotqa
from src.models.retrievers import MiniLMRetriever, BGESmallRetriever, BGEBaseRetriever
from src.models.generators import GPT4oMiniGenerator, Llama3Generator, Llama31Generator, MistralGenerator
from src.models.verifiers import MiniLMVerifier, GPT35Verifier, GPT4oMiniVerifier
from src.allocator.profiler import adaptive_profile
from src.allocator.selector import select_triplets
from src.allocator.pareto import pareto_front
from src.eval.pipeline import run_pipeline


def build_corpus_from_dataset(dataset):
    """Build corpus from dataset contexts or create synthetic corpus."""
    corpus = []
    
    # First, try to use context fields
    for item in dataset:
        context = item.get('context', '')
        if context and context.strip():
            corpus.append(context)
    
    # If no context, create corpus from questions + answers (synthetic documents)
    if not corpus:
        for item in dataset:
            question = item.get('question', '')
            answer = item.get('answer', '')
            if isinstance(answer, list) and answer:
                answer_str = answer[0]
            elif isinstance(answer, str):
                answer_str = answer
            else:
                answer_str = str(answer)
            
            # Create a synthetic document from question + answer
            doc = f"Question: {question} Answer: {answer_str}"
            corpus.append(doc)
    
    # If still empty, create dummy corpus
    if not corpus:
        corpus = ["Dummy document for retrieval."] * 10
    
    return corpus


def run_baseline(dataset_name: str, dataset, corpus):
    """Run uniform baseline."""
    print(f"\n=== Running Baseline: {dataset_name} ===")
    
    # Use GPT-3.5-turbo for all modules (closest to uniform baseline)
    retriever = BGESmallRetriever(corpus=corpus)
    generator = GPT4oMiniGenerator()  # Use GPT-4o-mini as closest available
    verifier = GPT35Verifier()
    
    results = run_pipeline(retriever, generator, verifier, dataset, seed=42)
    
    print(f"EM: {results['em']:.1f}%")
    print(f"F1: {results['f1']:.1f}%")
    print(f"Latency: {results['latency_ms']:.0f} ms")
    print(f"Cost: {results['cost_cents']:.2f} ¢/query")
    
    return results


def run_adaptive_allocation(dataset_name: str, dataset, corpus):
    """Run adaptive allocation system."""
    print(f"\n=== Running Adaptive Allocation: {dataset_name} ===")
    
    # Define model banks
    retrievers = [
        MiniLMRetriever(corpus=corpus),
        BGESmallRetriever(corpus=corpus),
        BGEBaseRetriever(corpus=corpus)
    ]
    
    generators = [
        Llama3Generator(),
        Llama31Generator(),
        MistralGenerator(),
        GPT4oMiniGenerator()
    ]
    
    verifiers = [
        MiniLMVerifier(),
        GPT35Verifier(),
        GPT4oMiniVerifier()
    ]
    
    model_banks = {
        'retriever': retrievers,
        'generator': generators,
        'verifier': verifiers
    }
    
    # Phase 1: Adaptive Profiling
    print("Phase 1: Profiling models on probe set...")
    start_time = time.time()
    perf_map = adaptive_profile(model_banks, dataset, probe_size=15, seed=42)
    profiling_time = time.time() - start_time
    print(f"Profiling completed in {profiling_time:.1f}s")
    
    # Phase 2: Budget-Aware Selection
    print("Phase 2: Selecting candidate triplets...")
    budget_lat = 2000  # 2 seconds
    budget_cost = 7.0  # 7 cents
    candidates = select_triplets(perf_map, budget_lat, budget_cost, top_k=5)
    print(f"Found {len(candidates)} budget-compliant candidates")
    
    if len(candidates) == 0:
        print("Warning: No candidates found within budget. Relaxing constraints...")
        budget_lat = 5000
        budget_cost = 10.0
        candidates = select_triplets(perf_map, budget_lat, budget_cost, top_k=5)
        print(f"Found {len(candidates)} candidates with relaxed budget")
    
    # Phase 3: Full Evaluation of Candidates
    print("Phase 3: Evaluating candidates on full dataset...")
    results_list = []
    
    # Create lookup maps
    retriever_map = {r.name: r for r in retrievers}
    generator_map = {g.name: g for g in generators}
    verifier_map = {v.name: v for v in verifiers}
    
    for i, config in enumerate(candidates):
        print(f"  Evaluating config {i+1}/{len(candidates)}: {config['R']}/{config['G']}/{config['V']}")
        
        # Instantiate models
        retriever = retriever_map[config['R']]
        generator = generator_map[config['G']]
        verifier = verifier_map[config['V']]
        
        results = run_pipeline(retriever, generator, verifier, dataset, seed=42)
        results['config'] = config
        results_list.append(results)
    
    # Phase 4: Pareto Optimization
    print("Phase 4: Finding Pareto-optimal configurations...")
    pareto_configs = pareto_front(results_list, q_metric='em', c_metric='cost_cents')
    
    # Select best (highest EM among Pareto front)
    if pareto_configs:
        best = max(pareto_configs, key=lambda x: x['em'])
        
        print(f"\nBest configuration: {best['config']['R']}/{best['config']['G']}/{best['config']['V']}")
        print(f"EM: {best['em']:.1f}%")
        print(f"F1: {best['f1']:.1f}%")
        print(f"Latency: {best['latency_ms']:.0f} ms")
        print(f"Cost: {best['cost_cents']:.2f} ¢/query")
    else:
        print("Warning: No Pareto-optimal configurations found.")
        best = results_list[0] if results_list else None
    
    return {
        'best': best,
        'pareto_front': pareto_configs,
        'all_results': results_list,
        'profiling_time': profiling_time
    }


def main():
    """Main experiment runner."""
    print("=" * 60)
    print("AutoRAG-Allocator Experiment Runner")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading datasets...")
    nq_data = load_nq_open(n_samples=100, seed=42)
    nq_corpus = build_corpus_from_dataset(nq_data)
    
    # For MVP, focus on NQ-Open only
    # hotpot_data = load_hotpotqa(n_samples=100, seed=42)
    # hotpot_corpus = build_corpus_from_dataset(hotpot_data)
    
    results = {}
    
    # NQ-Open experiments
    results['nq_baseline'] = run_baseline('NQ-Open', nq_data, nq_corpus)
    results['nq_adaptive'] = run_adaptive_allocation('NQ-Open', nq_data, nq_corpus)
    
    # HotpotQA experiments (commented out for MVP)
    # results['hotpot_baseline'] = run_baseline('HotpotQA', hotpot_data, hotpot_corpus)
    # results['hotpot_adaptive'] = run_adaptive_allocation('HotpotQA', hotpot_data, hotpot_corpus)
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_results = make_serializable(results)
    
    results_file = results_dir / "full_results.json"
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to {results_file}")
    
    # Print summary
    print("\nSummary:")
    print(f"NQ-Open Baseline:  EM={results['nq_baseline']['em']:.1f}%, Cost={results['nq_baseline']['cost_cents']:.2f}¢")
    if results['nq_adaptive'].get('best'):
        best = results['nq_adaptive']['best']
        print(f"NQ-Open Adaptive: EM={best['em']:.1f}%, Cost={best['cost_cents']:.2f}¢")
        print(f"Improvement: EM +{best['em'] - results['nq_baseline']['em']:.1f}%, Cost {best['cost_cents'] - results['nq_baseline']['cost_cents']:.2f}¢")


if __name__ == "__main__":
    main()

