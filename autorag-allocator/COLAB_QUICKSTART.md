# Colab Quick Start Guide

## 5-Minute Setup

### 1. Open Colab Notebook
- Go to [Google Colab](https://colab.research.google.com/)
- Upload `AutoRAG_Allocator_Colab.ipynb` or create new notebook

### 2. Set Runtime
- Runtime > Change runtime type
- Select: **GPU** (or **A100** if Colab Pro)
- Click Save

### 3. Set API Keys (Colab Secrets)
- Click 🔑 icon in left sidebar
- Add secrets:
  - `OPENAI_API_KEY`: Your OpenAI key
  - `GROQ_API_KEY`: Your Groq key

### 4. Run First Cell
```python
# Clone repo (or upload manually)
!git clone https://github.com/yourusername/ece570proj.git
%cd ece570proj/autorag-allocator
!pip install -q -r requirements.txt
!pip install -q faiss-gpu
```

### 5. Run Experiment
```python
import sys
import os
sys.path.insert(0, '/content/ece570proj/autorag-allocator')

os.environ['CORPUS_SIZE'] = '10000'  # Full corpus
from experiments.run_experiments import main
main()
```

## Quick Test (Small Corpus)

For faster testing, use smaller corpus:

```python
os.environ['CORPUS_SIZE'] = '1000'  # 1K passages instead of 10K
main()
```

## Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Setup | 2 min | Install dependencies |
| Corpus download | 10-15 min | First time only |
| Profiling | 5-8 min | 15 queries × 10 models |
| Evaluation | 15-20 min | 100 queries × 3 configs |
| **Total** | **30-45 min** | Full experiment |

## Troubleshooting

**"No GPU"**: Runtime > Change runtime type > GPU

**"No space"**: Use smaller corpus (`CORPUS_SIZE=1000`)

**"API key error"**: Check Colab secrets (🔑 icon)

**"Import error"**: Run setup cell again

## Download Results

```python
from google.colab import files
files.download('/content/ece570proj/autorag-allocator/results/full_results.json')
```

## Next Steps

1. ✅ Run quick test (1000 passages)
2. ✅ Verify results look reasonable
3. ✅ Run full experiment (10000 passages)
4. ✅ Generate figures
5. ✅ Download results

