# Paper-Accurate Implementation

This document describes the implementation that matches the paper exactly, using local models on Colab A100 GPUs.

## Models Matching Paper Specifications

### Generators (4 models, as in paper)
1. **Llama-3-8B** (`meta-llama/Llama-3-8B-Instruct`)
   - Local on Colab A100 (primary)
   - Groq API fallback if local unavailable
   
2. **Llama-3.1-8B** (`meta-llama/Llama-3.1-8B-Instruct`)
   - Local on Colab A100 (primary)
   - Groq API fallback if local unavailable
   
3. **Mistral-7B** (`mistralai/Mistral-7B-Instruct-v0.2`)
   - Local on Colab A100 (primary)
   - Groq API fallback if local unavailable
   
4. **gpt-4o-mini** (OpenAI API)
   - Always uses API (no local version)

### Verifiers (3 models, as in paper)
1. **ms-marco-MiniLM** (`cross-encoder/ms-marco-MiniLM-L-12-v2`)
   - Local cross-encoder model
   
2. **gpt-3.5-turbo** (OpenAI API)
   - Always uses API
   
3. **gpt-4o-mini** (OpenAI API)
   - Always uses API

### Retrievers (3 models, as in paper)
1. **MiniLM-L6** (`sentence-transformers/all-MiniLM-L6-v2`)
2. **bge-small-en** (`BAAI/bge-small-en-v1.5`)
3. **bge-base-en** (`BAAI/bge-base-en-v1.5`)

## Key Changes from Previous Implementation

### 1. Local Model Support
- **New**: `LocalLLMGenerator` class for running models locally on GPU
- Automatically detects GPU availability (A100 in Colab)
- Uses HuggingFace Transformers with float16 quantization
- Falls back to API if GPU unavailable

### 2. Paper-Accurate Models
- **Before**: Used Groq API models (Llama-3.3-70B, Qwen3-32B) - not in paper
- **Now**: Uses exact paper models (Llama-3-8B, Llama-3.1-8B, Mistral-7B)

### 3. Baseline Configuration
- **Before**: Used Llama-3.3-70B for verifier (not in paper)
- **Now**: Uses GPT-3.5-turbo for verifier (matches paper's verifier options)

### 4. Corpus Loading
- **Before**: Tried to download wiki_dpr (deprecated, caused errors)
- **Now**: Builds corpus from NQ-Open/HotpotQA datasets themselves (simpler, matches paper)

## Requirements

### For Local Models (Colab A100)
- `transformers>=4.30.0` - HuggingFace model loading
- `accelerate>=0.20.0` - Efficient model loading
- `torch>=2.0.0` - PyTorch for GPU inference
- GPU with >10GB VRAM (A100 has 40GB, perfect)

### For API Models
- `openai>=1.0.0` - OpenAI API
- `GROQ_API_KEY` - Optional, for fallback

## Usage

### On Colab with A100
```python
from src.models.generators import Llama3Generator, Llama31Generator, MistralGenerator

# Automatically uses local models if GPU available
generator = Llama3Generator()  # Loads Llama-3-8B locally on A100
```

### Fallback Behavior
If local models fail (no GPU, insufficient memory), automatically falls back to Groq API:
```python
# Will print: "⚠️  Local Llama-3-8B unavailable, using Groq API fallback..."
generator = Llama3Generator()  # Uses Groq API instead
```

## Performance Expectations

### Local Models (A100)
- **Loading time**: 2-5 minutes per model (first time, cached after)
- **Inference latency**: 200-500ms per query (8B models)
- **Memory usage**: ~15-20GB VRAM per 8B model (with float16)
- **Cost**: ~$0.01 per query (GPU compute time estimate)

### API Models
- **Loading time**: Instant (no model loading)
- **Inference latency**: 500-2000ms per query (network + API)
- **Cost**: $0.05-0.27 per query (actual API pricing)

## Benefits of Paper-Accurate Implementation

1. **Exact Reproducibility**: Matches paper models exactly
2. **Better Performance**: Local models faster than API (no network latency)
3. **Cost Savings**: Local inference cheaper than API calls
4. **No API Limits**: No rate limiting with local models
5. **Full Control**: Can modify prompts, temperature, etc. directly

## Notes

- **HuggingFace Access**: Llama models require HuggingFace authentication
  - Set `HF_TOKEN` in Colab secrets
  - Or use `huggingface-cli login` in Colab
  
- **Model Size**: 8B models fit comfortably in A100 (40GB)
  - Uses float16 quantization
  - ~15-20GB per model
  
- **First Run**: First time loading each model takes 2-5 minutes
  - Models are cached by HuggingFace
  - Subsequent runs are faster

## Verification

To verify models match paper:
```python
from src.models.generators import Llama3Generator, Llama31Generator, MistralGenerator
from src.models.verifiers import MiniLMVerifier, GPT35Verifier, GPT4oMiniVerifier

# Check generator names
g1 = Llama3Generator()
print(g1.name)  # Should be "Llama-3-8B"

g2 = Llama31Generator()
print(g2.name)  # Should be "Llama-3.1-8B"

g3 = MistralGenerator()
print(g3.name)  # Should be "Mistral-7B"

# Verifiers
v1 = MiniLMVerifier()
print(v1.name)  # Should be "ms-marco-MiniLM"

v2 = GPT35Verifier()
print(v2.name)  # Should be "gpt-3.5-turbo"

v3 = GPT4oMiniVerifier()
print(v3.name)  # Should be "gpt-4o-mini"
```

All models now match the paper exactly! 🎯

