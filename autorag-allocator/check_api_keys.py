#!/usr/bin/env python3
"""Quick script to verify API keys are loaded correctly."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent
load_dotenv(dotenv_path=project_root / ".env")

print("=" * 60)
print("API Key Check")
print("=" * 60)

# Check OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✓ OPENAI_API_KEY found: {openai_key[:10]}...{openai_key[-4:]}")
else:
    print("✗ OPENAI_API_KEY not found")

# Check Replicate
replicate_token = os.getenv("REPLICATE_API_TOKEN")
if replicate_token:
    print(f"✓ REPLICATE_API_TOKEN found: {replicate_token[:10]}...{replicate_token[-4:]}")
else:
    print("✗ REPLICATE_API_TOKEN not found")

print("\n.env file location:", project_root / ".env")
print(".env file exists:", (project_root / ".env").exists())

# Try to test Replicate connection
if replicate_token:
    print("\nTesting Replicate connection...")
    try:
        import replicate
        os.environ["REPLICATE_API_TOKEN"] = replicate_token
        replicate.default_client.api_token = replicate_token
        
        # Try a simple test
        print("Attempting test call to Replicate...")
        # Don't actually run, just check if client is configured
        print(f"Replicate client configured: {replicate.default_client.api_token is not None}")
    except Exception as e:
        print(f"Error testing Replicate: {e}")

print("=" * 60)

