"""Wikipedia corpus loading utilities for RAG systems."""
import random
import os
import pickle
from typing import List, Optional
from pathlib import Path
from datasets import load_dataset


def load_wikipedia_corpus(n_passages: int = 10000, seed: int = 42, cache_dir: Optional[Path] = None) -> List[str]:
    """
    Load Wikipedia passage corpus for NQ-Open from DPR dataset.
    
    Uses the wiki_dpr dataset which contains Wikipedia passages used in
    DPR (Dense Passage Retrieval) and Natural Questions evaluation.
    
    Args:
        n_passages: Number of passages to return
        seed: Random seed for reproducibility
        cache_dir: Optional directory to cache corpus (defaults to project data dir)
    
    Returns:
        List of passage strings (text content only)
    """
    random.seed(seed)
    
    # Set up cache directory
    if cache_dir is None:
        project_root = Path(__file__).parent.parent.parent
        cache_dir = project_root / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache file name based on parameters
    cache_file = cache_dir / f"wiki_corpus_{n_passages}_{seed}.pkl"
    
    # Try to load from cache first
    if cache_file.exists():
        print(f"Loading Wikipedia corpus from cache ({n_passages} passages)...")
        try:
            with open(cache_file, 'rb') as f:
                corpus = pickle.load(f)
            print(f"Loaded {len(corpus)} Wikipedia passages from cache")
            print(f"Average passage length: {sum(len(p) for p in corpus) / len(corpus) if corpus else 0:.0f} characters")
            return corpus
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), downloading fresh corpus...")
    
    # Load from HuggingFace (requires network)
    print(f"Loading Wikipedia corpus from HuggingFace ({n_passages} passages)...")
    print("Note: This requires network access and may take several minutes on first run.")
    print("Using streaming mode to minimize disk space requirements.")
    
    # Try multiple dataset sources (wiki_dpr is deprecated in newer datasets library)
    # Use Wikipedia dataset which is well-maintained and works with streaming
    try:
        print("Loading from Wikipedia dataset (20220301.en)...")
        # Add trust_remote_code=True to handle any script dependencies
        dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
        
        # Reservoir sampling: randomly sample n_passages from stream
        corpus = []
        seen = 0
        
        for item in dataset:
            # Wikipedia format: 'text' field contains full article
            text = item.get('text', '').strip()
            
            if not text or len(text) < 100:  # Skip very short articles
                continue
            
            # Split long articles into passages (Wikipedia articles are typically long)
            # Use paragraph breaks first, then sentence boundaries if needed
            paragraphs = text.split('\n\n')  # Split by double newline (paragraphs)
            
            for para in paragraphs:
                para = para.strip()
                if len(para) < 100:  # Skip short paragraphs
                    continue
                
                # If paragraph is too long, split by sentences
                if len(para) > 800:
                    sentences = para.split('. ')
                    # Group sentences into chunks of ~200-400 chars
                    current_chunk = []
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if not sent.endswith('.'):
                            sent += '.'
                        
                        current_chunk.append(sent)
                        chunk_text = ' '.join(current_chunk)
                        
                        # When chunk is substantial, add it
                        if len(chunk_text) >= 200:
                            corpus.append(chunk_text)
                            seen += 1
                            
                            # Reservoir sampling
                            if len(corpus) > n_passages:
                                j = random.randint(0, seen - 1)
                                if j < n_passages:
                                    corpus[j] = chunk_text
                                else:
                                    corpus.pop()
                            
                            if len(corpus) >= n_passages:
                                break
                            
                            current_chunk = []
                    
                    # Add remaining chunk if substantial
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text) >= 100:
                            corpus.append(chunk_text)
                            seen += 1
                            if len(corpus) > n_passages:
                                j = random.randint(0, seen - 1)
                                if j < n_passages:
                                    corpus[j] = chunk_text
                                else:
                                    corpus.pop()
                else:
                    # Use paragraph as-is if it's a good size
                    corpus.append(para)
                    seen += 1
                    
                    # Reservoir sampling
                    if len(corpus) > n_passages:
                        j = random.randint(0, seen - 1)
                        if j < n_passages:
                            corpus[j] = para
                        else:
                            corpus.pop()
                
                if len(corpus) >= n_passages:
                    break
            
            if len(corpus) >= n_passages:
                break
        
        if len(corpus) < n_passages:
            print(f"⚠️  Only collected {len(corpus)} passages (requested {n_passages})")
            print("   This is okay - we'll use what we have")
        
        print(f"✅ Successfully loaded {len(corpus)} passages from Wikipedia")
        
    except Exception as e:
        last_error = e
        error_msg = str(e)
        print(f"⚠️  Error loading Wikipedia dataset: {error_msg[:200]}")
        
        # Fallback: Try simpler approach with Natural Questions
        try:
            print("Trying fallback: Natural Questions dataset...")
            dataset = load_dataset("nq_open", split="train", streaming=True, trust_remote_code=True)
            
            corpus = []
            seen = 0
            
            for item in dataset:
                # NQ-Open may have context or we can use the question+answer as a passage
                text = item.get('context', '')
                if not text:
                    # Create a passage from question and answer
                    question = item.get('question', '')
                    answer = item.get('answer', [])
                    if isinstance(answer, list) and len(answer) > 0:
                        answer = answer[0]
                    text = f"{question} {answer}".strip()
                
                if len(text) < 50:
                    continue
                
                corpus.append(text)
                seen += 1
                
                if len(corpus) > n_passages:
                    j = random.randint(0, seen - 1)
                    if j < n_passages:
                        corpus[j] = text
                    else:
                        corpus.pop()
                
                if len(corpus) >= n_passages:
                    break
            
            print(f"✅ Loaded {len(corpus)} passages from NQ-Open fallback")
            
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load corpus from all sources.\n"
                f"Wikipedia error: {error_msg[:200]}\n"
                f"NQ-Open error: {str(fallback_error)[:200]}\n"
                f"Please check your internet connection or use a pre-cached corpus."
            )
    
    if corpus is None or len(corpus) == 0:
        raise RuntimeError(
            f"Failed to load corpus from all sources. Last error: {last_error if 'last_error' in locals() else 'Unknown'}\n"
            f"Please check your internet connection or use a pre-cached corpus."
        )
    
    # Trim to exact number requested
    corpus = corpus[:n_passages]
    print(f"Streamed through dataset, sampled {len(corpus)} passages")
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(corpus, f)
        print(f"Cached corpus to {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to cache corpus ({e})")
    
    print(f"Loaded {len(corpus)} Wikipedia passages")
    print(f"Average passage length: {sum(len(p) for p in corpus) / len(corpus) if corpus else 0:.0f} characters")
    
    return corpus


def load_wikipedia_corpus_cached(n_passages: int = 10000, seed: int = 42, cache_dir: Path = None) -> List[str]:
    """
    Load Wikipedia corpus with optional caching.
    
    Args:
        n_passages: Number of passages to return
        seed: Random seed for reproducibility
        cache_dir: Directory to cache corpus (if None, uses default datasets cache)
    
    Returns:
        List of passage strings
    """
    # For now, just call the main function
    # In future, could add caching logic here
    return load_wikipedia_corpus(n_passages=n_passages, seed=seed)

