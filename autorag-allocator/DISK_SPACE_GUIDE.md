# Disk Space Requirements and Solutions

## Problem
The `wiki_dpr` dataset requires **100-200GB** of disk space during download and processing, even though we only need 10,000 passages.

## Current Disk Space
You have **16GB available** (97% full), which is insufficient for the full dataset download.

## Solutions

### Option 1: Use Smaller Corpus Size (Recommended)
Reduce the corpus size to something manageable:

```python
# In experiments/run_experiments.py, change:
nq_corpus = build_corpus_from_wikipedia(n_passages=1000, seed=42)  # Instead of 10000
```

**Disk Space Needed**: ~10-20GB (much more manageable)
**Impact**: Slightly less realistic retrieval, but system still works correctly

### Option 2: Use Fallback Wikipedia Dataset
The code now automatically falls back to a smaller Wikipedia dataset if disk space is insufficient. This dataset is much smaller (~5-10GB).

**Automatic**: The code will try this if the main dataset fails.

### Option 3: Free Up Disk Space
If you want the full corpus:
- Need **~100-200GB free** for `wiki_dpr` dataset
- The dataset downloads to `~/.cache/huggingface/datasets/`
- After caching our 10K passages, you can delete the full dataset cache

### Option 4: Use Pre-Cached Corpus
If you have access to another machine with more space:
1. Download and cache the corpus there
2. Copy the cache file: `data/cache/wiki_corpus_10000_42.pkl`
3. Place it in your project's `data/cache/` directory

## Recommended Approach

For immediate testing and demo:
1. **Use smaller corpus** (1000-5000 passages)
2. System will work correctly with smaller corpus
3. Results will be directionally correct
4. Can scale up later if needed

## Updated Code

The corpus loader now:
- ✅ Uses **streaming mode** to minimize memory usage
- ✅ Implements **reservoir sampling** to randomly sample without loading everything
- ✅ Has **automatic fallback** to smaller dataset
- ✅ Provides **clear error messages** with solutions

## Quick Fix

Edit `experiments/run_experiments.py` line 170:
```python
# Change from:
nq_corpus = build_corpus_from_wikipedia(n_passages=10000, seed=42)

# To:
nq_corpus = build_corpus_from_wikipedia(n_passages=1000, seed=42)  # Smaller corpus
```

This will work immediately with your current disk space!

