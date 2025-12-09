# Troubleshooting Guide

## Issue: Llama Models Not Loading

### Error: "is not a valid model identifier"

**Cause**: HuggingFace token not being used or model access not granted

**Solutions**:

1. **Verify HF_TOKEN is set**:
   ```python
   import os
   print("HF_TOKEN:", "✅ Set" if os.getenv('HF_TOKEN') else "❌ Missing")
   ```

2. **Request access to gated models**:
   - Go to https://huggingface.co/meta-llama/Llama-3-8B-Instruct
   - Click "Agree and access repository"
   - Accept license terms
   - Wait for approval (usually instant)

3. **Verify login worked**:
   ```python
   from huggingface_hub import whoami
   try:
       user = whoami()
       print(f"✅ Logged in as: {user['name']}")
   except:
       print("❌ Not logged in")
   ```

4. **Test model access**:
   ```python
   from huggingface_hub import model_info
   try:
       info = model_info("meta-llama/Llama-3-8B-Instruct", token=os.getenv('HF_TOKEN'))
       print("✅ Model accessible")
   except Exception as e:
       print(f"❌ Cannot access model: {e}")
   ```

## Issue: Groq API Connection Errors

**Cause**: Network issues or API downtime

**Solutions**:

1. **Check network connection**:
   ```python
   import requests
   try:
       r = requests.get("https://api.groq.com", timeout=5)
       print("✅ Groq API reachable")
   except:
       print("❌ Cannot reach Groq API")
   ```

2. **Verify API key**:
   ```python
   import os
   groq_key = os.getenv('GROQ_API_KEY')
   if groq_key:
       print(f"✅ GROQ_API_KEY set: {groq_key[:10]}...")
   else:
       print("❌ GROQ_API_KEY not set")
   ```

3. **Use local models instead**: If Groq is down, ensure HF_TOKEN is set and use local models

## Issue: trust_remote_code Warning

**Cause**: Old datasets library version or deprecated parameter

**Solution**: Already fixed in `corpus.py` - removed `trust_remote_code` parameter

## Quick Diagnostic Script

Run this in Colab to check everything:

```python
import os
import torch
from huggingface_hub import whoami

print("=== Environment Check ===\n")

# Check API keys
print("API Keys:")
print(f"  OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
print(f"  GROQ_API_KEY: {'✅' if os.getenv('GROQ_API_KEY') else '❌'}")
print(f"  HF_TOKEN: {'✅' if os.getenv('HF_TOKEN') else '❌'}")

# Check GPU
print(f"\nGPU:")
print(f"  Available: {'✅' if torch.cuda.is_available() else '❌'}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check HuggingFace login
print(f"\nHuggingFace:")
try:
    user = whoami()
    print(f"  Logged in as: ✅ {user.get('name', 'Unknown')}")
except:
    print(f"  Login status: ❌ Not logged in")

# Check model access
print(f"\nModel Access:")
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    try:
        from huggingface_hub import model_info
        info = model_info("meta-llama/Llama-3-8B-Instruct", token=hf_token)
        print(f"  Llama-3-8B-Instruct: ✅ Accessible")
    except Exception as e:
        print(f"  Llama-3-8B-Instruct: ❌ {str(e)[:100]}")
else:
    print(f"  Llama-3-8B-Instruct: ❌ HF_TOKEN not set")
```

## Common Fixes

### Fix 1: Re-login to HuggingFace
```python
from huggingface_hub import login
import os
from google.colab import userdata

token = userdata.get('HF_TOKEN')
if token:
    login(token=token)
    print("✅ Re-logged in")
```

### Fix 2: Use API Models Only (Skip Local)
If local models keep failing, the code will automatically use Groq API. Just make sure `GROQ_API_KEY` is set.

### Fix 3: Check Model Names
The exact model names are:
- `meta-llama/Llama-3-8B-Instruct` (not `Llama-3-8B`)
- `meta-llama/Llama-3.1-8B-Instruct` (not `Llama-3.1-8B`)
- `mistralai/Mistral-7B-Instruct-v0.2`

Make sure you've requested access to these exact model names on HuggingFace.

