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
- `GROQ_API_KEY`: For Llama-3 and Mistral models (via Groq API)

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
- `src/data/`: Dataset loading utilities (including Wikipedia corpus loader)
- `experiments/`: Experiment scripts and figure generation
- `results/`: Output directory for results and figures
- `data/cache/`: Cached corpus files (created automatically)

## Usage

The system automatically:
1. Loads Wikipedia passage corpus (10,000 passages from DPR dataset)
2. Profiles models on a small probe set (15 queries)
3. Selects budget-compliant configurations
4. Evaluates top candidates on full dataset
5. Applies Pareto optimization
6. Reports best configuration

**Note**: First run will download the Wikipedia corpus (~500MB). Subsequent runs use cached corpus for faster startup.

## Cost Estimation

- Profiling: ~$0.50-1.00 (15 queries × 10 models)
- Full evaluation: ~$2-5 (100 queries × 2-3 configs)
- Total: ~$3-6 for complete experiment

## Reproducibility

All experiments use fixed seed (42) for reproducibility. Results are saved to `results/full_results.json`.

