#!/usr/bin/env python3
"""Test Groq models individually to verify API connection and model names."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.generators import Llama3Generator, Llama31Generator, MistralGenerator

def test_model(generator_class, model_name):
    """Test a single generator model."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        gen = generator_class()
        print(f"✓ Model initialized: {gen.name}")
        print(f"  Using Groq model: {gen.model_name}")
        
        # Test with a simple query
        test_query = "What is 2+2?"
        test_docs = ["Mathematics is the study of numbers.", "Addition is a basic operation."]
        
        print(f"Testing generation with query: '{test_query}'")
        answer = gen.generate(test_query, test_docs)
        
        print(f"✓ Generation successful!")
        print(f"  Answer: {answer[:100]}...")
        print(f"  Cost: {gen.cost:.4f} ¢/query")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all Groq models."""
    print("="*60)
    print("Groq Model Testing")
    print("="*60)
    
    results = {}
    
    # Test Llama-3-8B
    results['llama3'] = test_model(Llama3Generator, "Llama-3-8B")
    
    # Test Llama-3.1-8B
    results['llama31'] = test_model(Llama31Generator, "Llama-3.1-8B")
    
    # Test Qwen3-32B (replaces Mistral-7B)
    results['mistral'] = test_model(MistralGenerator, "qwen3-32b")
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:15} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All Groq models working correctly!")
    else:
        print("\n✗ Some models failed. Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

