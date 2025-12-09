"""Base classes for all model types."""
from abc import ABC, abstractmethod
from typing import List


class BaseRetriever(ABC):
    """Base class for retriever models."""
    
    def __init__(self):
        self._total_cost = 0.0
        self._total_latency_ms = 0.0
        self._query_count = 0
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 8) -> List[str]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
        
        Returns:
            List of document strings
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass
    
    @property
    def cost(self) -> float:
        """Cost per query in cents (average)."""
        if self._query_count == 0:
            return 0.0
        return (self._total_cost / self._query_count) * 100  # Convert to cents
    
    def _record_query(self, cost: float, latency_ms: float):
        """Record query cost and latency."""
        self._total_cost += cost
        self._total_latency_ms += latency_ms
        self._query_count += 1
    
    def reset_stats(self):
        """Reset statistics."""
        self._total_cost = 0.0
        self._total_latency_ms = 0.0
        self._query_count = 0


class BaseGenerator(ABC):
    """Base class for generator models."""
    
    def __init__(self):
        self._total_cost = 0.0
        self._total_latency_ms = 0.0
        self._query_count = 0
    
    @abstractmethod
    def generate(self, query: str, docs: List[str]) -> str:
        """
        Generate answer given query and retrieved documents.
        
        Args:
            query: Query string
            docs: List of retrieved document strings
        
        Returns:
            Generated answer string
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass
    
    @property
    def cost(self) -> float:
        """Cost per query in cents (average)."""
        if self._query_count == 0:
            return 0.0
        return (self._total_cost / self._query_count) * 100  # Convert to cents
    
    def _record_query(self, cost: float, latency_ms: float):
        """Record query cost and latency."""
        self._total_cost += cost
        self._total_latency_ms += latency_ms
        self._query_count += 1
    
    def reset_stats(self):
        """Reset statistics."""
        self._total_cost = 0.0
        self._total_latency_ms = 0.0
        self._query_count = 0


class BaseVerifier(ABC):
    """Base class for verifier models."""
    
    def __init__(self):
        self._total_cost = 0.0
        self._total_latency_ms = 0.0
        self._query_count = 0
    
    @abstractmethod
    def verify(self, query: str, answer: str, docs: List[str]) -> bool:
        """
        Verify if answer is supported by documents.
        
        Args:
            query: Query string
            answer: Generated answer string
            docs: List of retrieved document strings
        
        Returns:
            True if answer is verified, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass
    
    @property
    def cost(self) -> float:
        """Cost per query in cents (average)."""
        if self._query_count == 0:
            return 0.0
        return (self._total_cost / self._query_count) * 100  # Convert to cents
    
    def _record_query(self, cost: float, latency_ms: float):
        """Record query cost and latency."""
        self._total_cost += cost
        self._total_latency_ms += latency_ms
        self._query_count += 1
    
    def reset_stats(self):
        """Reset statistics."""
        self._total_cost = 0.0
        self._total_latency_ms = 0.0
        self._query_count = 0

