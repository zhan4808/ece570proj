# AutoRAG-Allocator

Budget-aware model assignment for compound RAG pipelines.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API keys:
- `OPENAI_API_KEY`: For GPT-3.5-turbo and GPT-4o-mini
- `REPLICATE_API_TOKEN`: For Llama-3 and Mistral models

3. Run experiments:
```bash
./run.sh --dataset nq --budget-cost 5.0
```

Or use Python directly:
```bash
python experiments/run_experiments.py
```

## Project Structure

- `src/models/`: Model implementations (retrievers, generators, verifiers)
- `src/allocator/`: Profiling and allocation logic
- `src/eval/`: Evaluation pipeline and metrics
- `src/data/`: Dataset loading utilities
- `experiments/`: Experiment scripts and figure generation
- `results/`: Output directory for results and figures

## Usage

The system automatically:
1. Profiles models on a small probe set (15 queries)
2. Selects budget-compliant configurations
3. Evaluates top candidates on full dataset
4. Applies Pareto optimization
5. Reports best configuration

## Cost Estimation

- Profiling: ~$0.50-1.00 (15 queries × 10 models)
- Full evaluation: ~$2-5 (100 queries × 2-3 configs)
- Total: ~$3-6 for complete experiment

## Reproducibility

All experiments use fixed seed (42) for reproducibility. Results are saved to `results/full_results.json`.

