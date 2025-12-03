"""
Optimal Data Preparation Pipeline
==================================
Consolidated script untuk preprocessing, embedding, dan dataset preparation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

# Import configurations
from config_optimal import *
from utils_optimal import *


def prepare_dataset(force_rebuild: bool = False):
    """
    Complete dataset preparation pipeline
    
    Steps:
    1. Load and balance dataset
    2. Build vocabulary
    3. Encode sequences
    4. Build embedding matrix
    5. Save all artifacts
    
    Args:
        force_rebuild: If True, rebuild even if artifacts exist
    """
    print("="*80)
    print("OPTIMAL DATA PREPARATION PIPELINE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load and Balance Dataset
    # ========================================================================
    print("\n[1/5] Loading and balancing dataset...")
    
    if not BALANCED_DATASET.exists():
        print(f"  Loading original dataset from {ORIGINAL_DATASET}")
        df_original = pd.read_csv(ORIGINAL_DATASET)
        
        print(f"  Original dataset: {len(df_original)} samples")
        print(f"  Class distribution:\n{df_original['category'].value_counts()}")
        
        # Balance dataset using undersampling
        groups = [df_original[df_original['category'] == label] 
                  for label in df_original['category'].unique()]
        min_size = min(len(g) for g in groups)
        
        print(f"  Balancing to {min_size} samples per class...")
        df_balanced = pd.concat([
            g.sample(min_size, random_state=RANDOM_STATE)
            for g in groups
        ], ignore_index=True)
        
        # Shuffle
        df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        # Save
        df_balanced.to_csv(BALANCED_DATASET, index=False)
        print(f"  ✅ Saved balanced dataset: {len(df_balanced)} samples")
    else:
        df_balanced = pd.read_csv(BALANCED_DATASET)
        print(f"  ✅ Loaded balanced dataset: {len(df_balanced)} samples")
    
    
    # ========================================================================
    # STEP 2: Build Vocabulary
    # ========================================================================
    print("\n[2/5] Building vocabulary...")
    
    vocab_path = VOCAB_DIR / "word2idx.pkl"
    
    if force_rebuild or not vocab_path.exists():
        print(f"  Processing {len(df_balanced)} texts...")
        word2idx = build_vocabulary(
            df_balanced['title'].tolist(),
            min_freq=MIN_WORD_FREQ
        )
        
        print(f"  Vocabulary size: {len(word2idx)} words")
        print(f"  Special tokens: <PAD>=0, <UNK>=1")
        
        save_pickle(word2idx, vocab_path)
    else:
        word2idx = load_pickle(vocab_path)
        print(f"  ✅ Loaded vocabulary: {len(word2idx)} words")
    
    
    # ========================================================================
    # STEP 3: Encode Sequences
    # ========================================================================
    print("\n[3/5] Encoding sequences...")
    
    X_path = DATASET_ARTIFACTS_DIR / "X.npy"
    y_path = DATASET_ARTIFACTS_DIR / "y.npy"
    label_encoder_path = LABELS_DIR / "label_encoder.pkl"
    
    if force_rebuild or not X_path.exists():
        print(f"  Encoding {len(df_balanced)} texts to sequences...")
        print(f"  Max length: {MAX_LEN}")
        
        # Encode texts
        X = encode_batch(df_balanced['title'].tolist(), word2idx, MAX_LEN)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df_balanced['category'])
        
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Number of classes: {len(label_encoder.classes_)}")
        print(f"  Classes: {label_encoder.classes_}")
        
        # Save
        np.save(X_path, X)
        np.save(y_path, y)
        save_pickle(label_encoder, label_encoder_path)
        print(f"  ✅ Saved encoded dataset")
    else:
        X = np.load(X_path)
        y = np.load(y_path)
        label_encoder = load_pickle(label_encoder_path)
        print(f"  ✅ Loaded encoded dataset: X{X.shape}, y{y.shape}")
    
    
    # ========================================================================
    # STEP 4: Build Embedding Matrix
    # ========================================================================
    print("\n[4/5] Building embedding matrix...")
    
    embedding_path = EMBEDDING_DIR / "embedding_matrix.npy"
    
    if force_rebuild or not embedding_path.exists():
        print(f"  Loading Word2Vec model from {W2V_MODEL_PATH}")
        w2v_model = Word2Vec.load(str(W2V_MODEL_PATH))
        
        actual_dim = w2v_model.vector_size
        print(f"  Word2Vec dimension: {actual_dim}")
        
        if actual_dim != EMBEDDING_DIM:
            print(f"  ⚠️  WARNING: Config has {EMBEDDING_DIM}, but Word2Vec is {actual_dim}")
            print(f"  Using actual dimension: {actual_dim}")
        
        print(f"  Building embedding matrix...")
        embedding_matrix = build_embedding_matrix(
            word2idx, w2v_model, actual_dim, OOV_INIT_SCALE
        )
        
        print(f"  Embedding matrix shape: {embedding_matrix.shape}")
        
        # Save
        np.save(embedding_path, embedding_matrix)
        print(f"  ✅ Saved embedding matrix")
    else:
        embedding_matrix = np.load(embedding_path)
        print(f"  ✅ Loaded embedding matrix: {embedding_matrix.shape}")
    
    
    # ========================================================================
    # STEP 5: Save Configuration
    # ========================================================================
    print("\n[5/5] Saving configuration...")
    
    config_path = CONFIG_DIR / "config.json"
    
    config = {
        "max_len": MAX_LEN,
        "vocab_size": len(word2idx),
        "embedding_dim": embedding_matrix.shape[1],  # Use actual dimension
        "num_classes": len(label_encoder.classes_),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "bidirectional": BIDIRECTIONAL,
        "dropout_embedding": DROPOUT_EMBEDDING,
        "dropout_rnn": DROPOUT_RNN,
        "dropout_fc": DROPOUT_FC,
        "class_names": label_encoder.classes_.tolist(),
    }
    
    save_json(config, config_path)
    
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE ✅")
    print("="*80)
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(X)}")
    print(f"  Vocabulary size: {len(word2idx)}")
    print(f"  Embedding dimension: {embedding_matrix.shape[1]}")
    print(f"  Number of classes: {len(label_encoder.classes_)}")
    print(f"  Max sequence length: {MAX_LEN}")
    
    print(f"\n📁 Saved Artifacts:")
    print(f"  ✅ {vocab_path}")
    print(f"  ✅ {X_path}")
    print(f"  ✅ {y_path}")
    print(f"  ✅ {label_encoder_path}")
    print(f"  ✅ {embedding_path}")
    print(f"  ✅ {config_path}")
    
    print("\n✨ Ready for training!")
    print("="*80)
    
    return {
        'X': X,
        'y': y,
        'word2idx': word2idx,
        'label_encoder': label_encoder,
        'embedding_matrix': embedding_matrix,
        'config': config
    }


if __name__ == "__main__":
    # Run the pipeline
    set_seed(RANDOM_STATE)
    artifacts = prepare_dataset(force_rebuild=False)
    
    print("\n🎯 Quick Stats:")
    print(f"  X shape: {artifacts['X'].shape}")
    print(f"  y shape: {artifacts['y'].shape}")
    print(f"  Unique labels: {np.unique(artifacts['y'])}")
    print(f"  Embedding shape: {artifacts['embedding_matrix'].shape}")
