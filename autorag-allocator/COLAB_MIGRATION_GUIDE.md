# Google Colab Pro Migration Guide

## Overview

This guide helps you migrate AutoRAG-Allocator from local Apple Silicon Mac to Google Colab Pro with A100 GPUs. This solves:
- ✅ **Storage limits**: Colab provides 200GB+ disk space
- ✅ **Memory limits**: A100 GPUs have 40GB+ VRAM
- ✅ **Apple Silicon compatibility**: No more M1/M2/M3 issues
- ✅ **Full corpus support**: Can download 100GB+ datasets

## Prerequisites

1. **Google Colab Pro subscription** ($10/month)
   - Provides A100 GPU access (up to 50 hours/week)
   - 200GB+ disk space
   - Faster runtime connections

2. **API Keys** (same as before):
   - `OPENAI_API_KEY`: For GPT models
   - `GROQ_API_KEY`: For Llama/Mistral models

## Migration Strategy

### Phase 1: Setup Colab Environment

1. **Create new Colab notebook**: `AutoRAG_Allocator.ipynb`
2. **Mount Google Drive** (optional, for persistent storage)
3. **Clone repository** or upload code
4. **Install dependencies**
5. **Set environment variables**

### Phase 2: Adapt Code for Colab

Key changes needed:
- ✅ Use Colab's GPU runtime
- ✅ Handle file paths (Colab uses `/content/`)
- ✅ Use Colab's disk space for corpus caching
- ✅ Adapt to Colab's Python environment

### Phase 3: Run Experiments

- Full corpus download (10K passages)
- All model profiling
- Complete evaluation pipeline

## Step-by-Step Setup

### Step 1: Create Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create new notebook: `File > New notebook`
3. Rename to `AutoRAG_Allocator.ipynb`

### Step 2: Initial Setup Cell

```python
# Cell 1: Setup and Installation
!git clone https://github.com/yourusername/ece570proj.git
%cd ece570proj/autorag-allocator

# Install dependencies
!pip install -q -r requirements.txt

# Install additional Colab-specific packages
!pip install -q faiss-gpu  # Use GPU version instead of faiss-cpu
```

### Step 3: Environment Variables

```python
# Cell 2: Set API Keys
import os
from google.colab import userdata

# Get API keys from Colab secrets (recommended) or set directly
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')

# Verify keys are set
print("API keys configured:", bool(os.getenv('OPENAI_API_KEY')) and bool(os.getenv('GROQ_API_KEY')))
```

**To set secrets in Colab:**
1. Click key icon (🔑) in left sidebar
2. Add secrets: `OPENAI_API_KEY` and `GROQ_API_KEY`
3. Access via `userdata.get('KEY_NAME')`

### Step 4: Verify GPU

```python
# Cell 3: Check GPU
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU detected. Make sure Runtime > Change runtime type > GPU (A100)")
```

### Step 5: Test Corpus Download

```python
# Cell 4: Test Corpus Loading
import sys
sys.path.insert(0, '/content/ece570proj/autorag-allocator')

from src.data.corpus import load_wikipedia_corpus

# Test with small corpus first
print("Testing corpus loading...")
corpus = load_wikipedia_corpus(n_passages=100, seed=42)
print(f"✅ Loaded {len(corpus)} passages")
print(f"Sample passage: {corpus[0][:200]}...")
```

### Step 6: Run Full Experiment

```python
# Cell 5: Run Experiments
import sys
sys.path.insert(0, '/content/ece570proj/autorag-allocator')

from experiments.run_experiments import main

# Set corpus size for full experiment
import os
os.environ['CORPUS_SIZE'] = '10000'  # Full corpus

# Run experiment
main()
```

## Code Adaptations Needed

### 1. Update File Paths

**File**: `src/data/corpus.py`
- Colab uses `/content/` as root
- Cache directory should be `/content/ece570proj/autorag-allocator/data/cache/`

**No changes needed** - code uses relative paths that work in Colab.

### 2. Use GPU for FAISS

**File**: `src/models/retrievers.py`

Change:
```python
# OLD (CPU only)
import faiss

# NEW (GPU support)
import faiss
# Colab will use faiss-gpu automatically
```

Actually, **no code changes needed** - just install `faiss-gpu` instead of `faiss-cpu`.

### 3. Handle Colab Timeouts

Colab sessions timeout after 90 minutes of inactivity. For long experiments:

**Option A: Use Colab Pro** (no timeout limits)
**Option B: Add checkpointing** (save progress periodically)

```python
# Add to run_experiments.py
import pickle
from pathlib import Path

def save_checkpoint(results, checkpoint_file):
    """Save intermediate results."""
    checkpoint_file = Path(checkpoint_file)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Checkpoint saved to {checkpoint_file}")

def load_checkpoint(checkpoint_file):
    """Load intermediate results."""
    with open(checkpoint_file, 'rb') as f:
        return pickle.load(f)
```

## Full Colab Notebook Template

See `AutoRAG_Allocator_Colab.ipynb` for complete notebook.

## Storage Management

### Colab Disk Space

- **Free tier**: ~80GB (shared)
- **Colab Pro**: 200GB+ (dedicated)
- **Colab Pro+**: 500GB+ (premium)

### Corpus Caching Strategy

1. **First run**: Downloads full corpus (~100GB)
2. **Cache location**: `/content/ece570proj/autorag-allocator/data/cache/wiki_corpus_10000_42.pkl`
3. **Subsequent runs**: Loads from cache (~500MB pickle file)

### Free Up Space

```python
# Clear HuggingFace cache if needed
!rm -rf ~/.cache/huggingface/datasets/wiki_dpr*

# Keep only the pickled corpus
!ls -lh /content/ece570proj/autorag-allocator/data/cache/
```

## Running Experiments

### Quick Test (Small Corpus)

```python
# Test with 1000 passages
os.environ['CORPUS_SIZE'] = '1000'
main()
```

### Full Experiment (10K Passages)

```python
# Full corpus
os.environ['CORPUS_SIZE'] = '10000'
main()
```

### Monitor Progress

```python
# Check disk usage
!df -h /content

# Check GPU memory
!nvidia-smi

# View results as they're generated
!tail -f /content/ece570proj/autorag-allocator/results/full_results.json
```

## Troubleshooting

### Issue: "No space left on device"

**Solution**: 
```python
# Check disk usage
!df -h

# Clear cache
!rm -rf ~/.cache/huggingface/datasets/*

# Use smaller corpus temporarily
os.environ['CORPUS_SIZE'] = '5000'
```

### Issue: "GPU not available"

**Solution**:
1. Runtime > Change runtime type
2. Select "GPU" (or "A100" if Pro)
3. Restart runtime

### Issue: "API key not found"

**Solution**:
```python
# Check if keys are set
import os
print("OPENAI:", bool(os.getenv('OPENAI_API_KEY')))
print("GROQ:", bool(os.getenv('GROQ_API_KEY')))

# Set directly if needed
os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['GROQ_API_KEY'] = 'gsk_...'
```

### Issue: "Session timeout"

**Solution**:
- Use Colab Pro (no timeouts)
- Or add checkpointing to resume
- Or run in smaller chunks

## Expected Performance

### Colab Pro A100

- **Corpus download**: ~10-15 minutes (first time)
- **Profiling**: ~5-8 minutes (15 queries × 10 models)
- **Full evaluation**: ~15-20 minutes (100 queries × 3 configs)
- **Total time**: ~30-45 minutes

### Storage Usage

- **Corpus download**: ~100GB (temporary)
- **Cached corpus**: ~500MB (permanent)
- **Model downloads**: ~5GB (HuggingFace cache)
- **Results**: ~10MB

## Next Steps

1. ✅ Create Colab notebook
2. ✅ Set up API keys
3. ✅ Test small corpus (100 passages)
4. ✅ Run full experiment (10K passages)
5. ✅ Generate figures
6. ✅ Download results

## Benefits of Colab Migration

1. **No storage limits**: Download full corpus
2. **GPU acceleration**: Faster embeddings/retrieval
3. **No Apple Silicon issues**: Standard x86_64 environment
4. **Reproducible**: Same environment every time
5. **Shareable**: Easy to share notebook with collaborators

## Cost Comparison

| Resource | Local Mac | Colab Pro |
|----------|-----------|-----------|
| Storage | Limited (16GB free) | 200GB+ |
| GPU | None (M-series) | A100 (40GB) |
| Cost | $0 (hardware owned) | $10/month |
| Setup time | Hours (debugging) | Minutes (ready) |

**Recommendation**: Use Colab Pro for experiments, keep local for development.

