"""Verifier model implementations."""
import os
import time
from typing import List
from pathlib import Path
from sentence_transformers import CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseVerifier

project_root = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=project_root / ".env")


class MiniLMVerifier(BaseVerifier):
    """MiniLM cross-encoder verifier."""
    
    def __init__(self):
        super().__init__()
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.threshold = 0.5  # Score threshold for verification
    
    def verify(self, query: str, answer: str, docs: List[str]) -> bool:
        """Verify answer using cross-encoder scoring."""
        if len(docs) == 0:
            return False
        
        start_time = time.time()
        
        # Score answer against each document
        pairs = [(answer, doc) for doc in docs]
        scores = self.model.predict(pairs)
        
        # Verify if any document supports the answer
        max_score = float(max(scores))
        verified = max_score >= self.threshold
        
        latency_ms = (time.time() - start_time) * 1000
        # Local computation cost is negligible
        self._record_query(0.0, latency_ms)
        
        return verified
    
    @property
    def name(self) -> str:
        return "ms-marco-MiniLM"


class LLMVerifier(BaseVerifier):
    """Base class for LLM-as-judge verifiers."""
    
    def __init__(self, model_name: str, display_name: str):
        super().__init__()
        self.model_name = model_name
        self.display_name = display_name
        
        # Load .env explicitly
        project_root = Path(__file__).parent.parent.parent
        load_dotenv(dotenv_path=project_root / ".env")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY not found in environment. Check .env file at {project_root / '.env'}")
        self.client = OpenAI(api_key=api_key)
        
        # Pricing per 1M tokens (input, output)
        self.input_price_per_1m = 0.0
        self.output_price_per_1m = 0.0
    
    def _create_prompt(self, query: str, answer: str, docs: List[str]) -> str:
        """Create verification prompt."""
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
        prompt = f"""Does the following answer correctly address the question based on the provided context? Respond with only "Yes" or "No".

Context:
{context}

Question: {query}

Answer: {answer}

Verification:"""
        return prompt
    
    def verify(self, query: str, answer: str, docs: List[str]) -> bool:
        """Verify answer using LLM-as-judge."""
        prompt = self._create_prompt(query, answer, docs)
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a verification system. Respond with only 'Yes' or 'No'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            verified = result.startswith("yes")
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens / 1_000_000 * self.input_price_per_1m) + \
                   (output_tokens / 1_000_000 * self.output_price_per_1m)
            
            latency_ms = (time.time() - start_time) * 1000
            self._record_query(cost, latency_ms)
            
            return verified
        except Exception as e:
            print(f"Error in {self.display_name}: {e}")
            latency_ms = (time.time() - start_time) * 1000
            self._record_query(0.0, latency_ms)
            return False
    
    @property
    def name(self) -> str:
        return self.display_name


class GPT35Verifier(LLMVerifier):
    """GPT-3.5-turbo verifier."""
    
    def __init__(self):
        super().__init__(
            model_name="gpt-3.5-turbo",
            display_name="gpt-3.5-turbo"
        )
        # Pricing: $0.50/1M input, $1.50/1M output
        self.input_price_per_1m = 0.50
        self.output_price_per_1m = 1.50


class GPT4oMiniVerifier(LLMVerifier):
    """GPT-4o-mini verifier."""
    
    def __init__(self):
        super().__init__(
            model_name="gpt-4o-mini",
            display_name="gpt-4o-mini"
        )
        # Pricing: $0.15/1M input, $0.60/1M output
        self.input_price_per_1m = 0.15
        self.output_price_per_1m = 0.60


class GroqVerifier(BaseVerifier):
    """Base class for Groq API verifiers (LLM-as-judge)."""
    
    def __init__(self, model_name: str, display_name: str, input_price_per_1m: float = 0.27, output_price_per_1m: float = 0.27):
        super().__init__()
        self.model_name = model_name
        self.display_name = display_name
        
        # Load .env explicitly
        project_root = Path(__file__).parent.parent.parent
        load_dotenv(dotenv_path=project_root / ".env")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(f"GROQ_API_KEY not found in environment. Check .env file at {project_root / '.env'}")
        
        # Initialize Groq client using OpenAI-compatible API
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        
        # Groq pricing per 1M tokens
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m
    
    def _create_prompt(self, query: str, answer: str, docs: List[str]) -> str:
        """Create verification prompt."""
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
        prompt = f"""Does the following answer correctly address the question based on the provided context? Respond with only "Yes" or "No".

Context:
{context}

Question: {query}

Answer: {answer}

Verification:"""
        return prompt
    
    def verify(self, query: str, answer: str, docs: List[str]) -> bool:
        """Verify answer using Groq LLM-as-judge."""
        prompt = self._create_prompt(query, answer, docs)
        
        start_time = time.time()
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a verification system. Respond with only 'Yes' or 'No'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                
                result = response.choices[0].message.content.strip().lower()
                verified = result.startswith("yes")
                
                # Calculate cost from actual token usage
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = (input_tokens / 1_000_000 * self.input_price_per_1m) + \
                       (output_tokens / 1_000_000 * self.output_price_per_1m)
                
                latency_ms = (time.time() - start_time) * 1000
                self._record_query(cost, latency_ms)
                
                return verified
                
            except Exception as e:
                error_str = str(e)
                is_bad_request = "400" in error_str or "401" in error_str or "404" in error_str
                
                if is_bad_request:
                    print(f"Error in {self.display_name}: {e}")
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_query(0.0, latency_ms)
                    return False
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Error for {self.display_name}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error in {self.display_name} after {max_retries} retries: {e}")
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_query(0.0, latency_ms)
                    return False
        
        latency_ms = (time.time() - start_time) * 1000
        self._record_query(0.0, latency_ms)
        return False
    
    @property
    def name(self) -> str:
        return self.display_name


# Note: Paper specifies only 3 verifiers:
# 1. ms-marco-MiniLM (MiniLMVerifier) - ✅ Already implemented
# 2. gpt-3.5-turbo (GPT35Verifier) - ✅ Already implemented  
# 3. gpt-4o-mini (GPT4oMiniVerifier) - ✅ Already implemented
#
# No Llama verifier in paper - removed to match paper exactly

