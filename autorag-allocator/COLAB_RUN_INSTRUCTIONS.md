# How to Run Experiments on Colab

## Quick Start

### 1. Setup (One Time)

**Upload Files:**
- Upload your `autorag-allocator` folder to Colab
- Files should be at: `/content/autorag-allocator/ece570proj/autorag-allocator/`

**Set Runtime:**
- Runtime > Change runtime type > **GPU (A100)** for Colab Pro
- Or **GPU (T4)** for free tier (slower, but works)

**Set Secrets (🔑 icon in sidebar):**
- `OPENAI_API_KEY`: Your OpenAI key
- `GROQ_API_KEY`: Your Groq key (optional, for fallback)
- `HF_TOKEN`: Your HuggingFace token (for local Llama models)

### 2. Run Notebook Cells in Order

**Cell 1: Setup and Installation**
```python
# Installs dependencies, sets up directory
# Just run it - should complete in 2-3 minutes
```

**Cell 2: Configure API Keys**
```python
# Loads API keys from Colab secrets
# Should show ✅ for all keys
```

**Cell 3: Verify GPU**
```python
# Checks if GPU is available
# Should show A100 or T4 GPU
```

**Cell 4: Login to HuggingFace**
```python
# Logs into HuggingFace for Llama models
# Should show ✅ HuggingFace login successful!
```

**Cell 5: Test Setup**
```python
# Tests corpus loading
# Should load 100 passages successfully
```

**Cell 6: Run Full Experiment** ⭐ **THIS IS THE MAIN ONE**
```python
# Runs the complete experiment
# Takes 30-45 minutes for full corpus (10000 passages)
# Or 10-15 minutes for quick test (1000 passages)
```

**Cell 7: View Results**
```python
# Shows experiment results
# Displays EM, F1, Cost, Latency metrics
```

**Cell 8: Generate Figures**
```python
# Creates PDF figures for paper
# Generates: results_comparison.pdf, pareto_frontier.pdf, profiling_overhead.pdf
```

**Cell 9: Download Results**
```python
# Downloads results JSON and figures
# Click to download files to your computer
```

## Quick Test vs Full Experiment

### Quick Test (Recommended First)
In **Cell 6**, change:
```python
corpus_size = 1000  # Quick test
```
- Takes ~10-15 minutes
- Tests everything works
- Good for debugging

### Full Experiment (Paper Results)
In **Cell 6**, use:
```python
corpus_size = 10000  # Full experiment
```
- Takes ~30-45 minutes
- Matches paper corpus size
- Generates final results

## Command-Line Alternative

If you prefer command line instead of notebook cells:

```python
# In a new cell, run:
import sys
import os
from pathlib import Path

# Find project
project_dir = '/content/autorag-allocator/ece570proj/autorag-allocator'
sys.path.insert(0, project_dir)
os.chdir(project_dir)

# Set environment
os.environ['CORPUS_SIZE'] = '10000'  # or '1000' for quick test

# Run experiment
from experiments.run_experiments import main
main()
```

## What Happens During Experiment

1. **Loads datasets**: NQ-Open (100 samples)
2. **Builds corpus**: From NQ-Open dataset (10000 passages)
3. **Profiles models**: Tests all 10 models on 15 probe queries (~5-8 min)
4. **Selects candidates**: Finds budget-compliant configurations
5. **Evaluates**: Runs full pipeline on selected configs (~15-20 min)
6. **Pareto optimization**: Finds optimal trade-offs
7. **Saves results**: To `results/full_results.json`

## Expected Output

```
=== Running Baseline: NQ-Open ===
EM: 62.0%
F1: 69.2%
Latency: 1700 ms
Cost: 6.2 ¢/query

=== Running Adaptive Allocation: NQ-Open ===
Phase 1: Profiling models on probe set...
  Profiling retriever: MiniLM-L6
  Profiling generator: Llama-3-8B
  ...
Profiling completed in 8.2s

Phase 2: Selecting candidate triplets...
Found 5 budget-compliant candidates

Phase 3: Evaluating candidates on full dataset...
  Evaluating config 1/5: bge-small-en/Llama-3-8B/gpt-3.5-turbo
  ...

Phase 4: Finding Pareto-optimal configurations...

Best configuration: bge-small-en/Mistral-7B/gpt-4o-mini
EM: 69.2%
F1: 76.4%
Latency: 1480 ms
Cost: 4.6 ¢/query
```

## Troubleshooting

### "Could not find project directory"
- **Fix**: Make sure files are uploaded to `/content/autorag-allocator/ece570proj/autorag-allocator/`

### "HF_TOKEN not set"
- **Fix**: Add `HF_TOKEN` to Colab secrets (🔑 icon)
- **Note**: Local models will fall back to API if not set

### "GPU not available"
- **Fix**: Runtime > Change runtime type > GPU (A100)
- **Note**: Free tier has T4 (slower but works)

### "Out of memory"
- **Fix**: Use smaller corpus (`corpus_size = 1000`)
- Or use API models only (set `HF_TOKEN` to empty)

### Experiment takes too long
- **Fix**: Use `corpus_size = 1000` for testing
- Or run in smaller chunks (modify `run_experiments.py`)

## Tips

1. **First run**: Use `corpus_size = 1000` to test everything works
2. **Monitor progress**: Check GPU memory with `!nvidia-smi` in a cell
3. **Save checkpoints**: Results are saved to JSON automatically
4. **Download results**: Use Cell 9 to download everything
5. **Re-run**: Just re-run Cell 6 to run experiment again (uses cached corpus)

## Time Estimates

| Step | Quick Test (1K) | Full (10K) |
|------|-----------------|------------|
| Corpus build | 2-3 min | 10-15 min |
| Profiling | 5-8 min | 5-8 min |
| Evaluation | 5-10 min | 15-20 min |
| **Total** | **12-21 min** | **30-43 min** |

## Next Steps After Experiment

1. ✅ View results (Cell 7)
2. ✅ Generate figures (Cell 8)
3. ✅ Download results (Cell 9)
4. ✅ Compare to paper results
5. ✅ Update paper if needed

That's it! Just run the cells in order. 🚀

