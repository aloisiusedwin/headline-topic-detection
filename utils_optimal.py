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


class RuntimeLogger:
    """
    Track and log training runtime statistics
    
    Usage:
        logger = RuntimeLogger()
        logger.start_training()
        logger.log_epoch(epoch, train_loss, val_loss, val_f1)
        logger.end_training()
        logger.save("artifacts/logs/run_001.json")
    """
    def __init__(self, experiment_name: str = "training"):
        self.experiment_name = experiment_name
        self.start_time = None
        self.end_time = None
        self.epoch_logs = []
        self.total_epochs = 0
        self.best_epoch = 0
        self.best_metric = 0.0
    
    def start_training(self):
        """Start timing the training"""
        self.start_time = time.time()
        print(f"\n[RuntimeLogger] Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None, 
                  val_metrics: Dict[str, float] = None, memory_gb: float = None):
        """Log metrics for one epoch"""
        epoch_log = {
            'epoch': epoch,
            'timestamp': time.time(),
            'train_loss': train_loss,
        }
        
        if val_loss is not None:
            epoch_log['val_loss'] = val_loss
        
        if val_metrics:
            epoch_log.update(val_metrics)
            
            # Track best epoch
            if 'f1' in val_metrics and val_metrics['f1'] > self.best_metric:
                self.best_metric = val_metrics['f1']
                self.best_epoch = epoch
        
        if memory_gb is not None:
            epoch_log['memory_gb'] = memory_gb
        
        self.epoch_logs.append(epoch_log)
        self.total_epochs = epoch + 1
    
    def end_training(self):
        """End timing and compute statistics"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print(f"[RuntimeLogger] Training ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[RuntimeLogger] Total time: {self._format_time(total_time)}")
        print(f"[RuntimeLogger] Average time per epoch: {self._format_time(total_time / max(1, self.total_epochs))}")
        print(f"[RuntimeLogger] Best epoch: {self.best_epoch} (F1={self.best_metric:.4f})")
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.start_time or not self.end_time:
            return {}
        
        total_time = self.end_time - self.start_time
        
        return {
            'experiment_name': self.experiment_name,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)),
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time)),
            'total_time_seconds': total_time,
            'total_time_formatted': self._format_time(total_time),
            'total_epochs': self.total_epochs,
            'avg_time_per_epoch': total_time / max(1, self.total_epochs),
            'best_epoch': self.best_epoch,
            'best_f1': self.best_metric,
            'epoch_logs': self.epoch_logs
        }
    
    def save(self, save_path: Path):
        """Save logs to JSON file"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"[RuntimeLogger] Logs saved to {save_path}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human readable string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


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

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[Path] = None,
                         save_formats: List[str] = ['png'], dpi: int = 300, 
                         title_prefix: str = ""):
    """
    Plot training history (loss and metrics) with enhanced visualization
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_f1', etc.
        save_path: Path to save figure (without extension)
        save_formats: List of formats to save ['png', 'pdf', 'svg']
        dpi: Resolution for saved figures
        title_prefix: Prefix for plot titles (e.g., "Fold 1")
    """
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss plot
    if 'train_loss' in history and history['train_loss']:
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label='Train Loss', 
                    marker='o', linewidth=2, markersize=6, color='#2E86AB')
    
    if 'val_loss' in history and history['val_loss']:
        epochs = range(1, len(history['val_loss']) + 1)
        axes[0].plot(epochs, history['val_loss'], label='Val Loss', 
                    marker='s', linewidth=2, markersize=6, color='#A23B72')
    
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{title_prefix} Training History - Loss'.strip(), 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Metrics plot
    colors = {'val_f1': '#F18F01', 'val_acc': '#2E86AB', 
              'val_precision': '#A23B72', 'val_recall': '#C73E1D'}
    
    for key in ['val_f1', 'val_acc', 'val_precision', 'val_recall']:
        if key in history and history[key]:
            epochs = range(1, len(history[key]) + 1)
            label = key.replace('val_', '').upper()
            axes[1].plot(epochs, history[key], label=label, 
                        marker='o', linewidth=2, markersize=6, 
                        color=colors.get(key, None))
    
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{title_prefix} Training History - Metrics'.strip(), 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        # Save in multiple formats
        for fmt in save_formats:
            full_path = Path(str(save_path).replace('.png', '').replace('.pdf', '') + f'.{fmt}')
            full_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(full_path, dpi=dpi, bbox_inches='tight', format=fmt)
            print(f"✅ Plot saved: {full_path}")
    else:
        plt.show()
    
    plt.close()


def plot_kfold_results(fold_metrics: List[Dict[str, float]], save_path: Optional[Path] = None,
                      save_formats: List[str] = ['png'], dpi: int = 300):
    """
    Plot K-Fold cross-validation results with enhanced visualization
    
    Args:
        fold_metrics: List of metric dictionaries for each fold
        save_path: Path to save figure (without extension)
        save_formats: List of formats to save ['png', 'pdf', 'svg']
        dpi: Resolution for saved figures
    """
    import pandas as pd
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    df = pd.DataFrame(fold_metrics)
    df['fold'] = range(1, len(df) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = df['fold']
    width = 0.2
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = {'accuracy': '#2E86AB', 'precision': '#A23B72', 
              'recall': '#F18F01', 'f1': '#C73E1D'}
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            bars = ax.bar(x + i * width - width*1.5, df[metric], width, 
                         label=metric.capitalize(), color=colors.get(metric, None),
                         edgecolor='black', linewidth=1.2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('K-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in x])
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 1.05])
    
    # Add mean line for each metric
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            mean_val = df[metric].mean()
            ax.axhline(y=mean_val, color=colors.get(metric, 'gray'), 
                      linestyle=':', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        # Save in multiple formats
        for fmt in save_formats:
            full_path = Path(str(save_path).replace('.png', '').replace('.pdf', '') + f'.{fmt}')
            full_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(full_path, dpi=dpi, bbox_inches='tight', format=fmt)
            print(f"✅ Plot saved: {full_path}")
    else:
        plt.show()
    
    plt.close()
    else:
        plt.show()
    
    plt.close()


def plot_fold_comparison(all_fold_histories: List[Dict[str, List[float]]], 
                        save_path: Optional[Path] = None,
                        save_formats: List[str] = ['png'], dpi: int = 300):
    """
    Plot comparison of all folds training curves
    
    Args:
        all_fold_histories: List of history dicts for each fold
        save_path: Path to save figure (without extension)
        save_formats: List of formats to save
        dpi: Resolution
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    plot_configs = [
        ('train_loss', 'Train Loss', 0),
        ('val_loss', 'Validation Loss', 1),
        ('val_f1', 'Validation F1 Score', 2),
        ('val_acc', 'Validation Accuracy', 3),
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_fold_histories)))
    
    for key, title, ax_idx in plot_configs:
        ax = axes[ax_idx]
        
        for fold_idx, history in enumerate(all_fold_histories):
            if key in history and history[key]:
                epochs = range(1, len(history[key]) + 1)
                ax.plot(epochs, history[key], 
                       label=f'Fold {fold_idx + 1}',
                       linewidth=2, alpha=0.7,
                       color=colors[fold_idx])
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} - All Folds', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        for fmt in save_formats:
            full_path = Path(str(save_path).replace('.png', '').replace('.pdf', '') + f'.{fmt}')
            full_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(full_path, dpi=dpi, bbox_inches='tight', format=fmt)
            print(f"✅ Plot saved: {full_path}")
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
