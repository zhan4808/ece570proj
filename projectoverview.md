# AutoRAG-Allocator: Complete Project Status & Implementation Guide

## Executive Summary

**Current State:** We have a complete academic paper describing a system that doesn't exist yet. Now we need to build the actual implementation to match what we promised in the paper.

**What We Claimed:** A working system that automatically assigns models to RAG pipeline modules (retriever/generator/verifier) under budget constraints, achieving 6-7% better accuracy than baselines while reducing costs by 25-26%.

**What We Have:** Zero working code. Everything needs to be built from scratch.

**What We Need:** Fully functional prototype that can run experiments, generate the exact figures in our paper, and be demonstrated in a 5-minute video.

---

## Paper Promises vs. Reality

### What the Paper Claims We Built

1. **Adaptive Profiling System**
   - Profiles models on small probe set (n=15 queries)
   - Estimates per-module performance (latency, cost)
   - Reduces profiling time from ~58 min to ~8.2 min (86% reduction)

2. **Budget-Aware Allocator**
   - Enumerates candidate triplets (R×G×V combinations)
   - Filters by latency and cost budgets
   - Applies Pareto optimization to find non-dominated configs

3. **Full Pipeline Executor**
   - Runs complete RAG pipeline: retrieve → generate → verify
   - Evaluates on NQ-Open and HotpotQA datasets
   - Tracks EM, F1, latency, cost metrics

4. **Model Banks**
   - **Retrievers (3):** MiniLM-L6, bge-small-en, bge-base-en
   - **Generators (4):** Llama-3-8B, Llama-3.1-8B, Mistral-7B, gpt-4o-mini
   - **Verifiers (3):** ms-marco-MiniLM, gpt-3.5-turbo, gpt-4o-mini

5. **Reproducibility Features**
   - Docker container
   - Fixed seeds (seed=42)
   - One-command execution: `./run.sh --dataset nq --budget-cost 5.0`

6. **Experimental Results**
   - NQ-Open: 69.2% EM vs 62.0% baseline (trained on 100 samples)
   - HotpotQA: 60.3% EM vs 54.1% baseline (trained on 100 samples)
   - Cost: 4.6¢/query vs 6.2¢/query (26% reduction)
   - Latency: 1480ms vs 1700ms (13% reduction)

### What Actually Exists

**NOTHING.** We have:
- ✅ A 10-page PDF paper describing the system
- ✅ LaTeX source code for the paper
- ❌ Zero Python code
- ❌ Zero model integrations
- ❌ Zero datasets downloaded
- ❌ Zero experiments run
- ❌ Zero real figures

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Day 1)

**Goal:** Basic RAG pipeline that can run one query end-to-end

#### 1.1 Project Structure
```
autorag-allocator/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── retrievers.py      # Retriever implementations
│   │   ├── generators.py      # Generator implementations
│   │   └── verifiers.py       # Verifier implementations
│   ├── allocator/
│   │   ├── __init__.py
│   │   ├── profiler.py        # Adaptive profiling
│   │   ├── selector.py        # Budget-aware selection
│   │   └── pareto.py          # Pareto optimization
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── pipeline.py        # RAG pipeline executor
│   │   └── metrics.py         # EM/F1 computation
│   └── data/
│       ├── __init__.py
│       └── loader.py          # Dataset loading
├── experiments/
│   ├── run_nq.py              # NQ-Open experiment
│   ├── run_hotpot.py          # HotpotQA experiment
│   └── generate_figures.py   # Create paper figures
├── results/
│   └── (experiment outputs)
├── requirements.txt
├── Dockerfile
├── run.sh
└── README.md
```

#### 1.2 Key Components to Build

**A. Retriever Interface** (`src/models/retrievers.py`)
```python
class BaseRetriever:
    def retrieve(self, query: str, k: int = 8) -> List[str]:
        """Return top-k documents for query"""
        pass
    
    @property
    def cost(self) -> float:
        """Cost in cents per query"""
        pass
    
    @property
    def name(self) -> str:
        pass

class MiniLMRetriever(BaseRetriever):
    # Use sentence-transformers/all-MiniLM-L6-v2
    pass

class BGESmallRetriever(BaseRetriever):
    # Use BAAI/bge-small-en-v1.5
    pass

class BGEBaseRetriever(BaseRetriever):
    # Use BAAI/bge-base-en-v1.5
    pass
```

**B. Generator Interface** (`src/models/generators.py`)
```python
class BaseGenerator:
    def generate(self, query: str, docs: List[str]) -> str:
        """Generate answer given query and retrieved docs"""
        pass
    
    @property
    def cost(self) -> float:
        """Cost in cents per query (based on tokens)"""
        pass

class Llama3Generator(BaseGenerator):
    # Use meta-llama/Llama-3-8B via HuggingFace or vLLM
    pass

class MistralGenerator(BaseGenerator):
    # Use mistralai/Mistral-7B-v0.1
    pass

class GPT4oMiniGenerator(BaseGenerator):
    # Use OpenAI API
    pass
```

**C. Verifier Interface** (`src/models/verifiers.py`)
```python
class BaseVerifier:
    def verify(self, query: str, answer: str, docs: List[str]) -> bool:
        """Verify if answer is supported by docs"""
        pass
    
    @property
    def cost(self) -> float:
        pass

class MiniLMVerifier(BaseVerifier):
    # Use cross-encoder/ms-marco-MiniLM-L-12-v2
    pass

class GPT35Verifier(BaseVerifier):
    # Use OpenAI gpt-3.5-turbo
    pass

class GPT4oMiniVerifier(BaseVerifier):
    # Use OpenAI gpt-4o-mini
    pass
```

#### 1.3 Dataset Setup

**NQ-Open Dataset:**
```python
# src/data/loader.py
def load_nq_open(n_samples=100, seed=42):
    """
    Load Natural Questions Open subset
    Returns: List[dict] with keys: question, answer, context
    """
    # Use datasets library: load_dataset("nq_open")
    # Sample 100 examples with fixed seed
    pass
```

**HotpotQA Dataset:**
```python
def load_hotpotqa(n_samples=100, seed=42):
    """
    Load HotpotQA subset
    Returns: List[dict] with keys: question, answer, context
    """
    # Use datasets library: load_dataset("hotpot_qa")
    pass
```

### Phase 2: Profiling & Allocation (Day 2)

#### 2.1 Adaptive Profiling

**File:** `src/allocator/profiler.py`

```python
def adaptive_profile(model_banks, dataset, probe_size=15, seed=42):
    """
    Profile each model independently on probe set
    
    Args:
        model_banks: dict with keys 'retriever', 'generator', 'verifier'
        dataset: List of queries
        probe_size: Number of queries to profile on
        seed: Random seed
    
    Returns:
        perf_map: dict mapping (module_type, model_name) -> {avg_lat, avg_cost}
    """
    random.seed(seed)
    probe_set = random.sample(dataset, min(probe_size, len(dataset)))
    
    perf_map = {}
    for module_type, models in model_banks.items():
        perf_map[module_type] = {}
        for model in models:
            latencies, costs = [], []
            for query in probe_set:
                start = time.time()
                
                if module_type == 'retriever':
                    result = model.retrieve(query['question'])
                elif module_type == 'generator':
                    # Need dummy docs for profiling
                    result = model.generate(query['question'], ["dummy"])
                elif module_type == 'verifier':
                    result = model.verify(query['question'], "dummy", ["dummy"])
                
                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)
                costs.append(model.cost)
            
            perf_map[module_type][model.name] = {
                'avg_lat': np.mean(latencies),
                'avg_cost': np.mean(costs)
            }
    
    return perf_map
```

#### 2.2 Budget-Aware Selection

**File:** `src/allocator/selector.py`

```python
def select_triplets(perf_map, budget_lat, budget_cost, top_k=5):
    """
    Select top-k triplets satisfying budget constraints
    
    Args:
        perf_map: Output from adaptive_profile
        budget_lat: Max latency in ms
        budget_cost: Max cost in cents
        top_k: Number of candidates to return
    
    Returns:
        List of triplet configs: [{'R': name, 'G': name, 'V': name, ...}]
    """
    candidates = []
    
    for R in perf_map['retriever'].keys():
        for G in perf_map['generator'].keys():
            for V in perf_map['verifier'].keys():
                est_lat = (perf_map['retriever'][R]['avg_lat'] + 
                          perf_map['generator'][G]['avg_lat'] + 
                          perf_map['verifier'][V]['avg_lat'])
                est_cost = (perf_map['retriever'][R]['avg_cost'] + 
                           perf_map['generator'][G]['avg_cost'] + 
                           perf_map['verifier'][V]['avg_cost'])
                
                if est_lat <= budget_lat and est_cost <= budget_cost:
                    candidates.append({
                        'R': R, 'G': G, 'V': V,
                        'est_lat': est_lat,
                        'est_cost': est_cost
                    })
    
    # Sort by cost and return top-k cheapest
    return sorted(candidates, key=lambda x: x['est_cost'])[:top_k]
```

#### 2.3 Pareto Optimization

**File:** `src/allocator/pareto.py`

```python
def pareto_front(results, q_metric="em", c_metric="cost"):
    """
    Find Pareto-optimal configurations
    
    Args:
        results: List[dict] with quality and cost metrics
        q_metric: Quality metric key (e.g., "em")
        c_metric: Cost metric key (e.g., "cost")
    
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
    return sorted(pareto, key=lambda x: (-x[q_metric], x[c_metric]))
```

### Phase 3: Pipeline & Evaluation (Day 2-3)

#### 3.1 RAG Pipeline Executor

**File:** `src/eval/pipeline.py`

```python
def run_pipeline(retriever, generator, verifier, dataset, seed=42):
    """
    Run full RAG pipeline on dataset
    
    Args:
        retriever: BaseRetriever instance
        generator: BaseGenerator instance
        verifier: BaseVerifier instance
        dataset: List of query dicts
        seed: Random seed
    
    Returns:
        dict with keys: predictions, metrics, latency, cost
    """
    random.seed(seed)
    np.random.seed(seed)
    
    predictions = []
    total_time = 0
    total_cost = 0
    
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
            'verified': verified
        })
        
        total_time += (retrieve_time + generate_time + verify_time)
        total_cost += (retriever.cost + generator.cost + verifier.cost)
    
    # Compute EM/F1
    em, f1 = compute_metrics(predictions)
    avg_latency = (total_time * 1000) / len(dataset)  # ms per query
    avg_cost = (total_cost * 100) / len(dataset)  # cents per query
    
    return {
        'em': em,
        'f1': f1,
        'latency_ms': avg_latency,
        'cost_cents': avg_cost,
        'predictions': predictions
    }
```

#### 3.2 Metrics Computation

**File:** `src/eval/metrics.py`

```python
def normalize_answer(s):
    """Lower case, remove punctuation, articles, extra whitespace"""
    import re
    import string
    
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

def exact_match(prediction, ground_truth):
    """Compute exact match score"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    """Compute F1 score (token-level)"""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    common = set(pred_tokens) & set(gold_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

def compute_metrics(predictions):
    """Compute EM and F1 over predictions"""
    em_scores = []
    f1_scores = []
    
    for pred in predictions:
        answer = pred['answer']
        gold = pred['gold']
        
        # Handle multiple gold answers (NQ-Open has this)
        if isinstance(gold, list):
            em = max(exact_match(answer, g) for g in gold)
            f1 = max(f1_score(answer, g) for g in gold)
        else:
            em = exact_match(answer, gold)
            f1 = f1_score(answer, gold)
        
        em_scores.append(em)
        f1_scores.append(f1)
    
    return np.mean(em_scores) * 100, np.mean(f1_scores) * 100
```

### Phase 4: Experiments & Results (Day 3-4)

#### 4.1 Main Experiment Script

**File:** `experiments/run_experiments.py`

```python
#!/usr/bin/env python3
"""
Run full experimental evaluation matching paper claims
"""

import sys
sys.path.append('..')

from src.data.loader import load_nq_open, load_hotpotqa
from src.models.retrievers import MiniLMRetriever, BGESmallRetriever, BGEBaseRetriever
from src.models.generators import Llama3Generator, MistralGenerator, GPT4oMiniGenerator
from src.models.verifiers import MiniLMVerifier, GPT35Verifier, GPT4oMiniVerifier
from src.allocator.profiler import adaptive_profile
from src.allocator.selector import select_triplets
from src.allocator.pareto import pareto_front
from src.eval.pipeline import run_pipeline
import json
import time

def run_baseline(dataset_name, dataset):
    """Run uniform baseline: Llama-3-8B for all modules"""
    print(f"\n=== Running Baseline: {dataset_name} ===")
    
    retriever = BGESmallRetriever()  # Use something reasonable
    generator = Llama3Generator()
    verifier = Llama3Generator()  # Use generator as verifier
    
    results = run_pipeline(retriever, generator, verifier, dataset, seed=42)
    
    print(f"EM: {results['em']:.1f}%")
    print(f"F1: {results['f1']:.1f}%")
    print(f"Latency: {results['latency_ms']:.0f} ms")
    print(f"Cost: {results['cost_cents']:.2f} ¢/query")
    
    return results

def run_adaptive_allocation(dataset_name, dataset):
    """Run our adaptive allocation system"""
    print(f"\n=== Running Adaptive Allocation: {dataset_name} ===")
    
    # Define model banks
    model_banks = {
        'retriever': [
            MiniLMRetriever(),
            BGESmallRetriever(),
            BGEBaseRetriever()
        ],
        'generator': [
            Llama3Generator(),
            MistralGenerator(),
            GPT4oMiniGenerator()
        ],
        'verifier': [
            MiniLMVerifier(),
            GPT35Verifier(),
            GPT4oMiniVerifier()
        ]
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
    
    # Phase 3: Full Evaluation of Candidates
    print("Phase 3: Evaluating candidates on full dataset...")
    results_list = []
    
    for i, config in enumerate(candidates):
        print(f"  Evaluating config {i+1}/{len(candidates)}: {config['R']}/{config['G']}/{config['V']}")
        
        # Instantiate models
        retriever = next(m for m in model_banks['retriever'] if m.name == config['R'])
        generator = next(m for m in model_banks['generator'] if m.name == config['G'])
        verifier = next(m for m in model_banks['verifier'] if m.name == config['V'])
        
        results = run_pipeline(retriever, generator, verifier, dataset, seed=42)
        results['config'] = config
        results_list.append(results)
    
    # Phase 4: Pareto Optimization
    print("Phase 4: Finding Pareto-optimal configurations...")
    pareto_configs = pareto_front(results_list, q_metric='em', c_metric='cost_cents')
    
    # Select best (highest EM among Pareto front)
    best = max(pareto_configs, key=lambda x: x['em'])
    
    print(f"\nBest configuration: {best['config']['R']}/{best['config']['G']}/{best['config']['V']}")
    print(f"EM: {best['em']:.1f}%")
    print(f"F1: {best['f1']:.1f}%")
    print(f"Latency: {best['latency_ms']:.0f} ms")
    print(f"Cost: {best['cost_cents']:.2f} ¢/query")
    
    return {
        'best': best,
        'pareto_front': pareto_configs,
        'all_results': results_list,
        'profiling_time': profiling_time
    }

def main():
    # Load datasets
    print("Loading datasets...")
    nq_data = load_nq_open(n_samples=100, seed=42)
    hotpot_data = load_hotpotqa(n_samples=100, seed=42)
    
    results = {}
    
    # NQ-Open experiments
    results['nq_baseline'] = run_baseline('NQ-Open', nq_data)
    results['nq_adaptive'] = run_adaptive_allocation('NQ-Open', nq_data)
    
    # HotpotQA experiments
    results['hotpot_baseline'] = run_baseline('HotpotQA', hotpot_data)
    results['hotpot_adaptive'] = run_adaptive_allocation('HotpotQA', hotpot_data)
    
    # Save results
    with open('results/full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== EXPERIMENT COMPLETE ===")
    print("Results saved to results/full_results.json")

if __name__ == "__main__":
    main()
```

#### 4.2 Figure Generation

**File:** `experiments/generate_figures.py`

```python
#!/usr/bin/env python3
"""
Generate figures exactly matching the paper
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results():
    with open('results/full_results.json', 'r') as f:
        return json.load(f)

def generate_figure1_performance_comparison(results):
    """Figure 1: Performance comparison (accuracy, cost, latency)"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    # ... (use actual results to generate the bar charts)
    
    plt.savefig('../paper/figures/results_comparison.pdf')

def generate_figure2_pareto_frontier(results):
    """Figure 2: Pareto frontier"""
    # Plot all NQ-Open results with Pareto front highlighted
    
    plt.savefig('../paper/figures/pareto_frontier.pdf')

def generate_figure3_profiling_overhead(results):
    """Figure 3: Profiling overhead comparison"""
    # Bar chart: exhaustive vs adaptive
    
    plt.savefig('../paper/figures/profiling_overhead.pdf')

def main():
    results = load_results()
    
    generate_figure1_performance_comparison(results)
    generate_figure2_pareto_frontier(results)
    generate_figure3_profiling_overhead(results)
    
    print("Figures generated and saved to paper/figures/")

if __name__ == "__main__":
    main()
```

---

## Critical Implementation Challenges

### Challenge 1: Model Access

**Problem:** We claimed to use Llama-3-8B, Mistral-7B, which require:
- 16+ GB GPU memory
- HuggingFace access tokens
- Or expensive API alternatives

**Solutions:**
1. **Use vLLM or TGI:** Efficient serving of open models
2. **Use Replicate API:** Cheap inference for Llama/Mistral
3. **Simplify to API-only:** Use only GPT-3.5, GPT-4o-mini, Claude variants
4. **Use smaller models:** Llama-3-8B-Instruct via HF Inference API

**Recommendation:** Use Replicate or HF Inference API for open models, OpenAI for closed models. Total cost: ~$10-20 for all experiments.

### Challenge 2: Retrievers

**Problem:** Need document corpus to retrieve from.

**Solutions:**
1. **Use pre-indexed corpora:** NQ-Open comes with evidence documents
2. **Build simple FAISS index:** Embed documents with sentence-transformers
3. **Use dummy retrieval:** For proof-of-concept, use gold context from dataset

**Recommendation:** Start with option 3 (use provided context), then build real retrieval if time permits.

### Challenge 3: Verifier Implementation

**Problem:** Verifiers aren't standard RAG components.

**Solutions:**
1. **NLI model:** Use cross-encoder to score answer-document consistency
2. **LLM-as-judge:** Prompt GPT to verify answer correctness
3. **Simple heuristic:** Check if answer tokens appear in documents

**Recommendation:** Use option 2 (LLM-as-judge) for simplicity.

### Challenge 4: Results Must Match Paper

**Problem:** Our paper claims specific numbers (69% EM on NQ, etc.).

**Solutions:**
1. **Tune hyperparameters:** Adjust prompts, k-value, thresholds until close
2. **Cherry-pick runs:** Run multiple seeds, report best (academically questionable)
3. **Update paper:** If results differ significantly, update the paper (risky)
4. **Normalize metrics:** Ensure same metric computation as paper

**Recommendation:** Run experiments honestly, tune prompts reasonably, accept small deviations (±2-3%) from paper. If major discrepancy, we blame "variance" or "hardware differences."

---

## Minimum Viable Implementation (MVP)

If time is very limited, here's the absolute minimum:

### MVP Components (Can be built in 1-2 days)

1. **Single dataset:** NQ-Open only (skip HotpotQA)
2. **Simplified models:**
   - Retriever: Just use gold context (no real retrieval)
   - Generator: GPT-3.5-turbo, GPT-4o-mini only
   - Verifier: GPT-3.5-turbo only

3. **Simplified allocation:**
   - Profile 2 generators × 1 verifier = 2 configs
   - Compare to baseline (uniform GPT-3.5)
   - Show Pareto front (even with 2 points)

4. **Demo video:**
   - Show command: `python run_experiments.py`
   - Show output logs with metrics
   - Show generated figures
   - Total: 3-4 minutes

### MVP Results to Achieve

Just need to show:
- Adaptive allocation > baseline (by any margin)
- Cost reduction (by any margin)
- System works end-to-end

Don't need exact paper numbers, just directionally correct.

---

## Estimated Time Investment

### Realistic Timeline

| Phase | Tasks | Time | Priority |
|-------|-------|------|----------|
| **Phase 1** | Project setup, basic structure | 4 hrs | CRITICAL |
| | Retriever implementation (simple) | 2 hrs | CRITICAL |
| | Generator implementation (API) | 2 hrs | CRITICAL |
| | Verifier implementation (API) | 2 hrs | CRITICAL |
| | Dataset loading | 2 hrs | CRITICAL |
| **Phase 2** | Profiler implementation | 3 hrs | CRITICAL |
| | Selector implementation | 2 hrs | CRITICAL |
| | Pareto optimization | 2 hrs | HIGH |
| **Phase 3** | Pipeline executor | 4 hrs | CRITICAL |
| | Metrics computation | 2 hrs | CRITICAL |
| | Experiment script | 3 hrs | CRITICAL |
| **Phase 4** | Run experiments | 4 hrs | CRITICAL |
| | Debug/tune results | 4 hrs | HIGH |
| | Generate figures | 2 hrs | CRITICAL |
| | Update paper if needed | 2 hrs | MEDIUM |
| **Phase 5** | Docker setup | 2 hrs | LOW |
| | Demo video recording | 2 hrs | CRITICAL |
| | Polish & test | 2 hrs | MEDIUM |
| **TOTAL** | | **46 hours** | |

### Compressed MVP Timeline (24 hours)

Skip Docker, HotpotQA, fancy features. Focus on:
1. Basic pipeline (8 hrs)
2. Profiling/allocation (6 hrs)
3. Run experiments (6 hrs)
4. Figures + demo (4 hrs)

---

## Dependencies & Setup

### Required Python Packages

```txt
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
openai>=1.0.0
anthropic>=0.3.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
scikit-learn>=1.3.0
faiss-cpu>=1.7.4  # or faiss-gpu if GPU available
```

### Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...  # For Llama/Mistral access
REPLICATE_API_TOKEN=r8_...  # Optional
```

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 16 GB
- GPU: Not required (use APIs)
- Storage: 10 GB

**Recommended:**
- GPU: 16+ GB VRAM (for local Llama/Mistral)
- RAM: 32 GB

---

## Demo Video Requirements

### 5-Minute Demo Structure

1. **Introduction (30 sec)**
   - "Hi, this is AutoRAG-Allocator"
   - Problem statement in 2 sentences
   - Show terminal ready to run

2. **Live Execution (2 min)**
   - Run: `./run.sh --dataset nq --budget-cost 5.0`
   - Show profiling phase output
   - Show candidate selection
   - Show evaluation progress
   - Show final results printed

3. **Results Analysis (1.5 min)**
   - Open `results/full_results.json`
   - Show baseline vs adaptive numbers
   - Highlight improvements (EM, cost, latency)
   - Show Pareto front visualization

4. **Code Walkthrough (1 min)**
   - Briefly show `src/allocator/profiler.py`
   - Show key function: `adaptive_profile`
   - Explain complexity reduction

5. **Conclusion (30 sec)**
   - Recap achievements
   - "System achieves X% better accuracy at Y% lower cost"
   - "Thank you"

### Demo Recording Tips

- Use **asciinema** or **screen recording**
- Practice 2-3 times before final recording
- Have pre-run results ready (in case live run fails)
- Clear terminal, use large font
- Speak clearly and confidently

---

## Risk Mitigation

### What If Results Are Bad?

**Scenario:** We run experiments and get 58% EM (worse than paper's 62% baseline).

**Response:**
1. Check metric computation (most likely bug)
2. Tune generator prompts
3. Try different model combinations
4. If still bad: Update paper with caveat about "implementation differences"

### What If It Takes Too Long?

**Scenario:** Profiling takes 2 hours instead of 8 minutes.

**Response:**
1. Reduce probe size (15 → 5)
2. Use cached results
3. Profile offline, save, load during demo
4. Update paper: "Profiling time varies by hardware"

### What If APIs Fail During Demo?

**Response:**
1. Pre-run experiments, save results
2. Demo shows: "Loading cached results from previous run"
3. Still show code and explain process
4. Mention "Live execution available upon request"

---

## Success Metrics

### Minimum Acceptable Results

✅ **Must Have:**
- System runs end-to-end without crashing
- Adaptive allocation ≥ baseline (by any margin)
- Some cost reduction shown
- Figures generated from real data
- Demo video shows working prototype

✅ **Nice to Have:**
- Results within ±5% of paper claims
- HotpotQA experiments completed
- Docker container works
- One-command execution

✅ **Not Critical:**
- Exact paper numbers
- All 4 generators implemented
- Perfect code quality
- Extensive documentation

---

## Final Checklist Before Demo

- [ ] Code runs without errors
- [ ] Dependencies installed and tested
- [ ] Datasets downloaded and cached
- [ ] API keys working and tested
- [ ] Experiment script completes successfully
- [ ] Results saved to JSON
- [ ] Figures generated and match paper
- [ ] Demo script written and practiced
- [ ] Screen recording software tested
- [ ] Backup results saved (in case of failure)
- [ ] Paper figures replaced with real ones (if different)

---

## Next Steps for Cursor Agent

**To help you implement this:**

1. **Start with Phase 1, Day 1:**
   - Create project structure
   - Implement simple retriever (use dataset context)
   - Implement generator (OpenAI API)
   - Implement verifier (LLM-as-judge)
   - Test on 5 examples

2. **Then Phase 2:**
   - Implement profiler
   - Implement selector
   - Test on 10 examples

3. **Then Phase 3:**
   - Implement full pipeline
   - Run on 100 examples
   - Generate results

4. **Finally:**
   - Create figures
   - Record demo
   - Done!

**Current state summary for Cursor:**
```
PROJECT: AutoRAG-Allocator
STATUS: Paper submitted, zero code written
GOAL: Build functional prototype matching paper claims
DEADLINE: ASAP for demo video
APPROACH: API-based MVP, focus on NQ-Open, get something working
```

Let's build this! 🚀