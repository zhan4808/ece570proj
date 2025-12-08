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
    
    try:
        # Use streaming to avoid loading entire dataset into memory/disk
        # This samples passages as we iterate, requiring minimal disk space
        dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact", split="train", streaming=True)
        
        # Reservoir sampling: randomly sample n_passages from stream
        # This avoids needing to know total size or load everything
        corpus = []
        seen = 0
        
        for item in dataset:
            text = item.get('text', '').strip()
            if not text:
                continue
            
            seen += 1
            
            # Reservoir sampling algorithm
            if len(corpus) < n_passages:
                corpus.append(text)
            else:
                # Randomly replace an existing item with probability n_passages/seen
                j = random.randint(0, seen - 1)
                if j < n_passages:
                    corpus[j] = text
        
        print(f"Streamed through {seen} passages, sampled {len(corpus)}")
        
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
        
    except Exception as e:
        error_msg = str(e)
        if "No space left" in error_msg or "disk" in error_msg.lower() or "space" in error_msg.lower():
            print(f"\n⚠️  Disk space error: Not enough space to download full dataset.")
            print(f"   Error: {error_msg[:200]}")
            print(f"\n   Solutions:")
            print(f"   1. Free up disk space (need ~50-100GB for wiki_dpr dataset)")
            print(f"   2. Use a smaller corpus size (e.g., n_passages=1000)")
            print(f"   3. Use alternative smaller dataset (see fallback below)")
            print(f"\n   Trying fallback: Using smaller Wikipedia dataset...")
            
            # Fallback: Use smaller Wikipedia dataset
            try:
                print("Loading smaller Wikipedia dataset (20220301.en subset)...")
                dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{n_passages*10}]", streaming=True)
                
                corpus = []
                for i, item in enumerate(dataset):
                    text = item.get('text', '').strip()
                    if text and len(text) > 100:  # Filter very short articles
                        corpus.append(text)
                        if len(corpus) >= n_passages:
                            break
                
                print(f"Loaded {len(corpus)} passages from fallback dataset")
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Both main and fallback corpus loading failed.\n"
                    f"Main error: {error_msg[:200]}\n"
                    f"Fallback error: {str(fallback_error)[:200]}\n"
                    f"Please free up disk space or use a pre-cached corpus."
                )
        elif "network" in error_msg.lower() or "connection" in error_msg.lower() or "resolve" in error_msg.lower():
            print(f"\n⚠️  Network error: Cannot download corpus right now.")
            print(f"   Error: {error_msg[:200]}")
            print(f"\n   Solutions:")
            print(f"   1. Check internet connection and try again")
            print(f"   2. If you have a cached corpus, it will be used automatically")
            print(f"   3. For offline testing, you can create a corpus file manually")
            raise ConnectionError(f"Network required to download corpus: {error_msg}")
        else:
            raise


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

