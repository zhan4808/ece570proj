"""Generator model implementations matching paper specifications.

Paper models:
- Llama-3-8B (meta-llama/Llama-3-8B-Instruct)
- Llama-3.1-8B (meta-llama/Llama-3.1-8B-Instruct)
- Mistral-7B (mistralai/Mistral-7B-Instruct-v0.2)
- gpt-4o-mini (OpenAI API)

On Colab with A100, we can run local models. Falls back to API if needed.
"""
import os
import time
import torch
from typing import List, Optional
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseGenerator

project_root = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=project_root / ".env")


def _check_gpu_available() -> bool:
    """Check if GPU is available for local model inference."""
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 10e9  # >10GB


class LocalLLMGenerator(BaseGenerator):
    """Base class for local LLM generators using HuggingFace Transformers."""
    
    def __init__(self, model_name: str, display_name: str, hf_model_path: str):
        super().__init__()
        self.model_name = model_name
        self.display_name = display_name
        self.hf_model_path = hf_model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer. Only loads if GPU available."""
        if not _check_gpu_available():
            raise RuntimeError(f"GPU not available for local model {self.display_name}. Use API version instead.")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading {self.display_name} locally (this may take a few minutes)...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with 8-bit quantization to fit in A100 (40GB)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"✅ {self.display_name} loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.display_name} locally: {e}. Use API version instead.")
    
    def _create_prompt(self, query: str, docs: List[str]) -> str:
        """Create RAG prompt from query and documents."""
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
        prompt = f"""Given the following context documents, answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def generate(self, query: str, docs: List[str]) -> str:
        """Generate answer using local model."""
        if self.model is None:
            raise RuntimeError(f"Model {self.display_name} not loaded")
        
        prompt = self._create_prompt(query, docs)
        
        start_time = time.time()
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # Estimate cost (local models: compute cost, estimate as ~$0.01 per query for 8B models)
            # This is a rough estimate - actual cost depends on GPU time
            estimated_cost = 0.01  # $0.01 per query for local inference
            
            latency_ms = (time.time() - start_time) * 1000
            self._record_query(estimated_cost, latency_ms)
            
            return answer
        except Exception as e:
            print(f"Error in {self.display_name}: {e}")
            latency_ms = (time.time() - start_time) * 1000
            self._record_query(0.0, latency_ms)
            return "Error generating answer."
    
    @property
    def name(self) -> str:
        return self.display_name


class OpenAIGenerator(BaseGenerator):
    """Base class for OpenAI generators."""
    
    def __init__(self, model_name: str, display_name: str):
        super().__init__()
        self.model_name = model_name
        self.display_name = display_name
        
        project_root = Path(__file__).parent.parent.parent
        load_dotenv(dotenv_path=project_root / ".env")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)
        
        self.input_price_per_1m = 0.0
        self.output_price_per_1m = 0.0
    
    def _create_prompt(self, query: str, docs: List[str]) -> str:
        """Create RAG prompt from query and documents."""
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
        prompt = f"""Given the following context documents, answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def generate(self, query: str, docs: List[str]) -> str:
        """Generate answer using OpenAI API."""
        prompt = self._create_prompt(query, docs)
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens / 1_000_000 * self.input_price_per_1m) + \
                   (output_tokens / 1_000_000 * self.output_price_per_1m)
            
            latency_ms = (time.time() - start_time) * 1000
            self._record_query(cost, latency_ms)
            
            return answer
        except Exception as e:
            print(f"Error in {self.display_name}: {e}")
            latency_ms = (time.time() - start_time) * 1000
            self._record_query(0.0, latency_ms)
            return "Error generating answer."
    
    @property
    def name(self) -> str:
        return self.display_name


class GroqGenerator(BaseGenerator):
    """Base class for Groq API generators (fallback when local models unavailable)."""
    
    def __init__(self, model_name: str, display_name: str, input_price_per_1m: float = 0.10, output_price_per_1m: float = 0.10):
        super().__init__()
        self.model_name = model_name
        self.display_name = display_name
        
        project_root = Path(__file__).parent.parent.parent
        load_dotenv(dotenv_path=project_root / ".env")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(f"GROQ_API_KEY not found in environment")
        
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m
    
    def _create_prompt(self, query: str, docs: List[str]) -> str:
        """Create RAG prompt from query and documents."""
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
        prompt = f"""Given the following context documents, answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def generate(self, query: str, docs: List[str]) -> str:
        """Generate answer using Groq API."""
        prompt = self._create_prompt(query, docs)
        start_time = time.time()
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=200
                )
                
                answer = response.choices[0].message.content.strip()
                
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = (input_tokens / 1_000_000 * self.input_price_per_1m) + \
                       (output_tokens / 1_000_000 * self.output_price_per_1m)
                
                latency_ms = (time.time() - start_time) * 1000
                self._record_query(cost, latency_ms)
                
                return answer
                
            except Exception as e:
                error_str = str(e)
                is_bad_request = "400" in error_str or "401" in error_str or "404" in error_str
                
                if is_bad_request:
                    print(f"Error in {self.display_name}: {e}")
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_query(0.0, latency_ms)
                    return "Error generating answer (bad request)."
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Error for {self.display_name}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error in {self.display_name} after {max_retries} retries: {e}")
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_query(0.0, latency_ms)
                    return "Error generating answer."
        
        latency_ms = (time.time() - start_time) * 1000
        self._record_query(0.0, latency_ms)
        return "Error generating answer."
    
    @property
    def name(self) -> str:
        return self.display_name


# Paper-specified generators with automatic local/API selection
def Llama3Generator():
    """Llama-3-8B generator (paper specification).
    
    Tries local first (Colab A100), falls back to Groq API.
    """
    try:
        return LocalLLMGenerator(
            model_name="llama-3-8b",
            display_name="Llama-3-8B",
            hf_model_path="meta-llama/Llama-3-8B-Instruct"
        )
    except Exception as e:
        print(f"⚠️  Local Llama-3-8B unavailable ({e}), using Groq API fallback...")
        return GroqGenerator(
            model_name="llama-3.1-8b-instant",  # Groq doesn't have 3-8B, use closest
            display_name="Llama-3-8B (Groq)",
            input_price_per_1m=0.05,
            output_price_per_1m=0.08
        )


def Llama31Generator():
    """Llama-3.1-8B generator (paper specification).
    
    Tries local first (Colab A100), falls back to Groq API.
    """
    try:
        return LocalLLMGenerator(
            model_name="llama-3.1-8b",
            display_name="Llama-3.1-8B",
            hf_model_path="meta-llama/Llama-3.1-8B-Instruct"
        )
    except Exception as e:
        print(f"⚠️  Local Llama-3.1-8B unavailable ({e}), using Groq API fallback...")
        return GroqGenerator(
            model_name="llama-3.1-8b-instant",
            display_name="Llama-3.1-8B (Groq)",
            input_price_per_1m=0.05,
            output_price_per_1m=0.08
        )


def MistralGenerator():
    """Mistral-7B generator (paper specification).
    
    Tries local first (Colab A100), falls back to Groq API.
    """
    try:
        return LocalLLMGenerator(
            model_name="mistral-7b",
            display_name="Mistral-7B",
            hf_model_path="mistralai/Mistral-7B-Instruct-v0.2"
        )
    except Exception as e:
        print(f"⚠️  Local Mistral-7B unavailable ({e}), using Groq API fallback...")
        return GroqGenerator(
            model_name="mixtral-8x7b-32768",  # Groq doesn't have Mistral-7B, use Mixtral
            display_name="Mistral-7B (Groq Mixtral)",
            input_price_per_1m=0.24,
            output_price_per_1m=0.24
        )


class GPT4oMiniGenerator(OpenAIGenerator):
    """GPT-4o-mini generator (paper specification)."""
    
    def __init__(self):
        super().__init__(
            model_name="gpt-4o-mini",
            display_name="gpt-4o-mini"
        )
        # Pricing: $0.15/1M input, $0.60/1M output
        self.input_price_per_1m = 0.15
        self.output_price_per_1m = 0.60
