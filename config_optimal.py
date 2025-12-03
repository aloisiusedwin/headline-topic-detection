"""
Optimal Configuration for Headline Topic Detection
===================================================
Central configuration file untuk konsistensi di seluruh pipeline
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# Artifact subdirectories
VOCAB_DIR = ARTIFACTS_DIR / "vocab"
LABELS_DIR = ARTIFACTS_DIR / "labels"
CONFIG_DIR = ARTIFACTS_DIR / "config"
DATASET_ARTIFACTS_DIR = ARTIFACTS_DIR / "dataset"
EMBEDDING_DIR = ARTIFACTS_DIR / "embedding"
MODEL_DIR = ARTIFACTS_DIR / "model_final"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
RESULTS_DIR = ARTIFACTS_DIR / "results"
PLOTS_DIR = ARTIFACTS_DIR / "plots"  # ✅ NEW: Visualization plots
LOGS_DIR = ARTIFACTS_DIR / "logs"    # ✅ NEW: Runtime logs

# Create all directories
for dir_path in [VOCAB_DIR, LABELS_DIR, CONFIG_DIR, DATASET_ARTIFACTS_DIR, 
                 EMBEDDING_DIR, MODEL_DIR, CHECKPOINT_DIR, RESULTS_DIR,
                 PLOTS_DIR, LOGS_DIR]:  # ✅ Added PLOTS_DIR and LOGS_DIR
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset files
ORIGINAL_DATASET = DATASET_DIR / "indonesian-news-title.csv"
BALANCED_DATASET = DATASET_DIR / "indonesian-news-title-balanced.csv"
AUGMENTED_DATASET = DATASET_DIR / "indonesian-news-title-augmented.csv"

# Preprocessing
MAX_LEN = 20  # Maximum sequence length (optimized for headlines)
MIN_WORD_FREQ = 1  # Minimum word frequency to include in vocab

# Data split
TRAIN_SIZE = 0.8
VAL_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# K-Fold
N_SPLITS = 5  # Number of folds for cross-validation


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Embedding
EMBEDDING_DIM = 300  # ✅ FIXED: Match actual Word2Vec dimension
W2V_MODEL_PATH = EMBEDDINGS_DIR / "idwiki_word2vec.model"
OOV_INIT_SCALE = 0.6  # Scale for random initialization of OOV words

# Architecture
MODEL_TYPE = "LSTM"  # or "GRU"
HIDDEN_SIZE = 256  # ✅ INCREASED: From 128 to 256
NUM_LAYERS = 2  # ✅ NEW: Multi-layer RNN
BIDIRECTIONAL = True  # ✅ NEW: Bidirectional RNN

# Regularization
DROPOUT_EMBEDDING = 0.3  # ✅ NEW: Dropout after embedding
DROPOUT_RNN = 0.3  # ✅ NEW: Dropout between RNN layers
DROPOUT_FC = 0.5  # ✅ NEW: Dropout before final layer
WEIGHT_DECAY = 1e-5  # ✅ NEW: L2 regularization


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # ✅ LOWERED: From 1e-3 to 1e-4 for fine-tuning
MAX_EPOCHS = 20  # ✅ INCREASED: From 5 to 20 with early stopping
GRAD_CLIP_NORM = 1.0  # ✅ NEW: Gradient clipping

# Early stopping
EARLY_STOPPING = True
PATIENCE = 5  # ✅ NEW: Stop if no improvement for 5 epochs

# Learning rate scheduler
LR_SCHEDULER = "ReduceLROnPlateau"
LR_FACTOR = 0.5  # Reduce LR by 50%
LR_PATIENCE = 3  # Wait 3 epochs before reducing
LR_MIN = 1e-6  # Minimum learning rate

# Device
DEVICE = "cuda"  # or "cpu" - will be auto-detected


# ============================================================================
# AUGMENTATION CONFIGURATION
# ============================================================================

# Augmentation methods (set to False to disable)
USE_PARAPHRASE = True  # ✅ KEEP: Most effective
USE_BACKTRANSLATION = False  # ✅ DISABLED: Too slow (60% of time)
USE_SYNONYM = True  # ✅ KEEP: Fast and effective

# Quality filtering
MIN_SIMILARITY = 0.6  # Minimum cosine similarity to accept
MAX_SIMILARITY = 0.95  # Maximum cosine similarity (avoid duplicates)

# Model paths for augmentation
PARAPHRASE_MODEL = "Wikidepia/IndoT5-base-paraphrase"
TRANSLATION_MODEL_ID_EN = "Helsinki-NLP/opus-mt-id-en"
TRANSLATION_MODEL_EN_ID = "Helsinki-NLP/opus-mt-en-id"

# Generation parameters
PARA_MAX_LENGTH = 128
PARA_TOP_K = 50
PARA_TOP_P = 0.95
PARA_DO_SAMPLE = True


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Metrics
METRICS = ["accuracy", "precision", "recall", "f1"]
AVERAGE = "macro"  # Macro-averaged metrics

# Reporting
SAVE_RESULTS = True
SAVE_PLOTS = True
VERBOSE = True

# Plotting configuration
AUTO_PLOT = True  # ✅ NEW: Automatically generate plots during training
PLOT_DPI = 300  # High quality plots
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # Plot style
SAVE_PLOT_FORMATS = ['png', 'pdf']  # Save formats

# Runtime logging
LOG_RUNTIME = True  # ✅ NEW: Log training runtime statistics
LOG_MEMORY = True  # Log GPU memory usage
LOG_INTERVAL = 1  # Log every N epochs


# ============================================================================
# OPTIMIZATION FLAGS
# ============================================================================

# Memory optimization
PIN_MEMORY = True  # For DataLoader (GPU only)
NUM_WORKERS = 0  # For DataLoader (0 for Windows compatibility)
PREFETCH_FACTOR = 2  # Prefetch batches

# Training optimization
MIXED_PRECISION = False  # FP16 training (requires GPU with Tensor Cores)
COMPILE_MODEL = False  # Torch.compile (PyTorch 2.0+)

# Caching
CACHE_EMBEDDINGS = True  # Cache embedding matrix
CACHE_DATASETS = True  # Cache preprocessed datasets


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_dict():
    """Get configuration as dictionary"""
    return {
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "bidirectional": BIDIRECTIONAL,
        "dropout_embedding": DROPOUT_EMBEDDING,
        "dropout_rnn": DROPOUT_RNN,
        "dropout_fc": DROPOUT_FC,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "grad_clip_norm": GRAD_CLIP_NORM,
    }


def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("OPTIMAL CONFIGURATION")
    print("=" * 80)
    print(f"\n📊 DATA:")
    print(f"  Max sequence length: {MAX_LEN}")
    print(f"  Train/Val/Test split: {TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE}")
    print(f"  K-Fold splits: {N_SPLITS}")
    
    print(f"\n🏗️  MODEL ARCHITECTURE:")
    print(f"  Type: {MODEL_TYPE}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Bidirectional: {BIDIRECTIONAL}")
    print(f"  Dropout (emb/rnn/fc): {DROPOUT_EMBEDDING}/{DROPOUT_RNN}/{DROPOUT_FC}")
    
    print(f"\n⚙️  TRAINING:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Max epochs: {MAX_EPOCHS}")
    print(f"  Early stopping patience: {PATIENCE}")
    print(f"  Gradient clipping: {GRAD_CLIP_NORM}")
    
    print(f"\n🎨 AUGMENTATION:")
    print(f"  Paraphrase: {USE_PARAPHRASE}")
    print(f"  Backtranslation: {USE_BACKTRANSLATION}")
    print(f"  Synonym: {USE_SYNONYM}")
    print(f"  Similarity filter: {MIN_SIMILARITY}-{MAX_SIMILARITY}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_config()
