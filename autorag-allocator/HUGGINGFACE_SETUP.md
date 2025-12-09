# HuggingFace Token Setup for Colab

## Token Type Needed

**Answer: Classic token with "Read" access is sufficient.**

You don't need:
- ❌ Write access (we're not uploading models)
- ❌ Fine-grained tokens (classic tokens work fine)
- ❌ Special permissions (just read access)

## Step-by-Step Setup

### 1. Request Access to Llama Models

Llama models are "gated" - you need to request access first:

1. Go to model pages and click "Agree and access repository":
   - [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Llama-3-8B-Instruct)
   - [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   - [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

2. Accept the license terms for each model
3. Wait for approval (usually instant for academic use)

### 2. Create HuggingFace Token

1. Go to [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Choose:
   - **Name**: `colab-autorag` (or any name)
   - **Type**: **Classic token** (not fine-grained)
   - **Permissions**: **Read** (this is enough!)
4. Click "Generate token"
5. **Copy the token immediately** (you won't see it again!)

### 3. Add Token to Colab

**Option A: Colab Secrets (Recommended)**
1. In Colab, click the 🔑 (key) icon in left sidebar
2. Click "Add new secret"
3. Name: `HF_TOKEN`
4. Value: Paste your HuggingFace token
5. Click "Add secret"

**Option B: Environment Variable**
```python
import os
os.environ['HF_TOKEN'] = 'hf_your_token_here'
```

### 4. Verify Token Works

In Colab, test the token:
```python
from huggingface_hub import login
import os
from google.colab import userdata

# Get token from Colab secrets
try:
    token = userdata.get('HF_TOKEN')
    login(token=token)
    print("✅ HuggingFace login successful!")
except Exception as e:
    print(f"❌ Login failed: {e}")
    print("Make sure HF_TOKEN is set in Colab secrets")
```

## Token Permissions Explained

| Permission | Needed? | Why |
|------------|---------|-----|
| **Read** | ✅ **YES** | To download models from HuggingFace Hub |
| **Write** | ❌ No | Only needed if uploading models (we're not) |
| Fine-grained | ❌ Optional | Classic tokens work fine, fine-grained is more secure but not required |

## Troubleshooting

### Error: "401 Unauthorized"
- **Cause**: Token not set or invalid
- **Fix**: Check token is in Colab secrets as `HF_TOKEN`

### Error: "403 Forbidden" or "Repository not found"
- **Cause**: Haven't requested access to gated models
- **Fix**: Go to model pages and click "Agree and access repository"

### Error: "Repository is gated"
- **Cause**: Access request pending or not approved
- **Fix**: Check your HuggingFace account - you should see "Access granted" on model pages

## Quick Checklist

- [ ] Created HuggingFace account
- [ ] Requested access to Llama-3-8B-Instruct
- [ ] Requested access to Llama-3.1-8B-Instruct  
- [ ] Requested access to Mistral-7B-Instruct-v0.2
- [ ] Created classic token with "Read" permission
- [ ] Added token to Colab secrets as `HF_TOKEN`
- [ ] Tested login in Colab

## Security Note

- **Never commit tokens to git**
- **Never share tokens publicly**
- Colab secrets are encrypted and only accessible in your Colab session
- Tokens can be revoked and recreated if compromised

## Alternative: Use API Models Only

If you don't want to deal with HuggingFace tokens, the code will automatically fall back to Groq API models when local models fail. However, for paper-accurate results, you'll want the local models.

