"""
Optimized Utility Functions
============================
Reusable functions untuk preprocessing, augmentation, training, dan evaluation
"""

import re
import numpy as np
import pickle
import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize Indonesian text
    
    Optimizations:
    - Single-pass regex operations
    - Efficient string operations
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphanumeric (keep Indonesian characters)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def build_vocabulary(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    """
    Build vocabulary from texts with frequency filtering
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency to include word in vocab
    
    Returns:
        word2idx: Dictionary mapping words to indices
    """
    from collections import Counter
    
    # Count word frequencies
    counter = Counter()
    for text in texts:
        cleaned = clean_text(text)
        counter.update(cleaned.split())
    
    # Build vocab with special tokens
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    
    # Add words that meet frequency threshold
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)
    
    return word2idx


def encode_text(text: str, word2idx: Dict[str, int], max_len: int) -> List[int]:
    """
    Encode text to sequence of indices
    
    Args:
        text: Input text
        word2idx: Vocabulary mapping
        max_len: Maximum sequence length
    
    Returns:
        Encoded sequence (padded or truncated to max_len)
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()
    
    # Convert to indices
    seq = [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]
    
    # Pad or truncate
    if len(seq) < max_len:
        seq = seq + [word2idx["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    
    return seq


def encode_batch(texts: List[str], word2idx: Dict[str, int], max_len: int) -> np.ndarray:
    """
    Encode batch of texts (optimized with list comprehension)
    """
    return np.array([encode_text(t, word2idx, max_len) for t in texts])


# ============================================================================
# EMBEDDING UTILITIES
# ============================================================================

def build_embedding_matrix(word2idx: Dict[str, int], w2v_model, embed_dim: int, 
                          oov_scale: float = 0.6) -> np.ndarray:
    """
    Build embedding matrix from Word2Vec model
    
    Args:
        word2idx: Vocabulary mapping
        w2v_model: Gensim Word2Vec model
        embed_dim: Embedding dimension
        oov_scale: Scale for OOV word initialization
    
    Returns:
        Embedding matrix of shape (vocab_size, embed_dim)
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    
    found = 0
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            found += 1
        else:
            # Random initialization for OOV words
            embedding_matrix[idx] = np.random.normal(0, oov_scale, embed_dim)
    
    coverage = (found / vocab_size) * 100
    print(f"Embedding coverage: {found}/{vocab_size} ({coverage:.2f}%)")
    
    return embedding_matrix


# ============================================================================
# AUGMENTATION QUALITY CONTROL
# ============================================================================

def is_valid_augmentation(original: str, augmented: str, 
                         min_sim: float = 0.6, max_sim: float = 0.95) -> bool:
    """
    Check if augmented text is valid using cosine similarity
    
    Args:
        original: Original text
        augmented: Augmented text
        min_sim: Minimum similarity threshold
        max_sim: Maximum similarity threshold
    
    Returns:
        True if augmentation is valid
    """
    if augmented is None or augmented == original:
        return False
    
    try:
        vectorizer = TfidfVectorizer()
        vecs = vectorizer.fit_transform([original, augmented])
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        return min_sim <= sim <= max_sim
    except:
        return False


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get optimal device"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    """Simple timer for benchmarking"""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __str__(self):
        mins = int(self.elapsed // 60)
        secs = int(self.elapsed % 60)
        return f"{mins}m {secs}s"


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   average: str = "macro") -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'micro', 'weighted')
    
    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics across folds
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Dictionary of (mean, std) for each metric
    """
    import pandas as pd
    df = pd.DataFrame(metrics_list)
    return {
        col: (df[col].mean(), df[col].std())
        for col in df.columns
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[Path] = None):
    """
    Plot training history (loss and metrics)
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_f1', etc.
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics plot
    for key in ['val_f1', 'val_acc', 'val_precision', 'val_recall']:
        if key in history:
            axes[1].plot(history[key], label=key.replace('val_', '').upper(), marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Training History - Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_kfold_results(fold_metrics: List[Dict[str, float]], save_path: Optional[Path] = None):
    """
    Plot K-Fold cross-validation results
    
    Args:
        fold_metrics: List of metric dictionaries for each fold
        save_path: Path to save figure
    """
    import pandas as pd
    
    df = pd.DataFrame(fold_metrics)
    df['fold'] = range(1, len(df) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = df['fold']
    width = 0.2
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax.bar(x + i * width, df[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('K-Fold Cross-Validation Results')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'Fold {i}' for i in x])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def save_pickle(obj, path: Path):
    """Save object as pickle"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved to {path}")


def load_pickle(path: Path):
    """Load pickle object"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(obj: dict, path: Path):
    """Save dictionary as JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)
    print(f"Saved to {path}")


def load_json(path: Path) -> dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================================
# MODEL CHECKPOINT UTILITIES
# ============================================================================

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict[str, float], path: Path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Metrics dictionary
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                   path: Path, device: torch.device):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        path: Path to checkpoint
        device: Device to load on
    
    Returns:
        epoch, metrics
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


# ============================================================================
# MEMORY OPTIMIZATION
# ============================================================================

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage() -> str:
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CPU mode - no GPU memory tracking"


if __name__ == "__main__":
    print("Utils module loaded successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
