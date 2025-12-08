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

# Check Groq
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    print(f"✓ GROQ_API_KEY found: {groq_key[:10]}...{groq_key[-4:]}")
else:
    print("✗ GROQ_API_KEY not found")

print("\n.env file location:", project_root / ".env")
print(".env file exists:", (project_root / ".env").exists())

# Try to test Groq connection
if groq_key:
    print("\nTesting Groq connection...")
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key
        )
        
        # Try a simple test
        print("Attempting test call to Groq...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=5
        )
        result = response.choices[0].message.content.strip()
        print(f"✓ Groq connection successful! Response: '{result}'")
    except Exception as e:
        print(f"✗ Error testing Groq: {e}")

print("=" * 60)

