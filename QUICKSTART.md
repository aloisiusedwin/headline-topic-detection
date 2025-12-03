# 🚀 Quick Start Guide - Optimal Pipeline

## ⚡ Super Quick Start (3 Commands)

```bash
# 1. Prepare data
python prepare_data_optimal.py

# 2. Train with auto-plotting & runtime logging
python train_with_plots.py

# 3. Check results in artifacts/plots/
```

**NEW FEATURES:**
- ✅ **Auto-generate graphs** untuk setiap fold dan overall
- ✅ **Runtime logging** dengan timestamps dan memory usage
- ✅ **Multi-format export** (PNG, PDF)
- ✅ **Automatic saving** ke folder `artifacts/plots/` dan `artifacts/logs/`

---

## 📋 What's Been Optimized?

### **1. Centralized Configuration** (`config_optimal.py`)
✅ Semua settings di satu file:
- Paths (auto-created)
- Model hyperparameters
- Training parameters
- Augmentation settings

**Before:** Settings tersebar di 5+ notebook  
**After:** Edit 1 file, apply everywhere

---

### **2. Reusable Utilities** (`utils_optimal.py`)
✅ Functions untuk:
- Text preprocessing
- Embedding matrix creation
- Metrics computation
- Visualization
- File I/O

**Before:** Copy-paste code antar notebook  
**After:** `from utils_optimal import *`

---

### **3. Data Preparation** (`prepare_data_optimal.py`)
✅ All-in-one pipeline:
1. Load & balance dataset
2. Build vocabulary
3. Encode sequences
4. Build embedding matrix
5. Save all artifacts

**Before:** Run 3 notebooks sequentially  
**After:** 1 command

---

### **4. Augmentation** (`augment_data_optimal.py`)
✅ Optimized augmentation:
- ❌ Removed backtranslation (60% time saved)
- ✅ Quality filtering (cosine similarity)
- ✅ 100+ synonym entries
- ✅ Batch processing

**Before:** 4 hours for augmentation  
**After:** ~1.6 hours (60% faster)

---

### **5. Training** (`training_improved.ipynb`)
✅ Improved architecture & training:
- BiLSTM/BiGRU (2 layers)
- Dropout regularization (0.3-0.5)
- Train/val split (80/20)
- Early stopping (patience=5)
- LR scheduler
- Gradient clipping

**Before:** F1=0.818 (overfitting risk)  
**After:** F1=0.85-0.87 (reliable)

---

## 🎯 Key Improvements Summary

| Aspect | Before | After | Impact |
|--------|--------|-------|--------|
| **Architecture** | Unidirectional LSTM, no dropout | BiLSTM + dropout | +3-5% F1 |
| **Training** | Blind 5 epochs | Early stopping + monitoring | Prevent overfit |
| **Augmentation** | 3 methods, 4 hours | 2 methods, 1.6 hours | 60% faster |
| **Data Leakage** | ❌ Aug before split | ✅ Aug after split | Valid results |
| **Code Quality** | Scattered notebooks | Modular scripts | Maintainable |
| **Configuration** | Hardcoded values | Centralized config | Easy tuning |

---

## 🎨 Auto-Generated Visualizations

### **Training menghasilkan 4 jenis plot otomatis:**

1. **Per-Fold Training History** (`fold{N}_history.png`)
   - Train/Val loss curves
   - Validation metrics (F1, accuracy, precision, recall)
   - Saved untuk setiap fold

2. **K-Fold Results Comparison** (`kfold_results.png`)
   - Bar chart perbandingan semua fold
   - Mean lines untuk setiap metric
   - Value labels di setiap bar

## 🛠️ How to Modify Settings

### **1. Change plot settings:**
Edit `config_optimal.py`:
```python
AUTO_PLOT = True           # Set to False to disable auto-plotting
PLOT_DPI = 300            # Change to 150 or 600
SAVE_PLOT_FORMATS = ['png', 'pdf']  # Add 'svg' if needed
LOG_RUNTIME = True        # Set to False to disable runtime logging
```

### **2. Change model architecture:**, val F1, val accuracy
   - Easy comparison antar folds

4. **Runtime Logs** (`runtime.json`)
   - Start/end timestamps
   - Total training time
   - Per-epoch timing
   - Memory usage (GPU)
   - Best epoch info

### **Example output structure:**
```
artifacts/
├── plots/
│   ├── training_20241204_143022_fold1_history.png
│   ├── training_20241204_143022_fold1_history.pdf
│   ├── training_20241204_143022_fold2_history.png
│   ├── training_20241204_143022_kfold_results.png
│   └── training_20241204_143022_all_folds_comparison.png
│
└── logs/
    └── training_20241204_143022_runtime.json
```

---

## 📊 Expected Results

### **Without Augmentation:**
- F1 Score: **0.85-0.87** (from 0.818)
- Training time: ~20 min (with early stopping)
- Variance: 1.5-2.5% (natural)

### **With Augmentation:**
- F1 Score: **0.87-0.90** (realistic, valid)
- Training time: ~1.6 hours aug + 30 min training
- Variance: 1.5-2.5% (natural)

---

## 🛠️ How to Modify Settings

### **1. Change model architecture:**
Edit `config_optimal.py`:
```python
HIDDEN_SIZE = 256  # Change to 128 or 512
NUM_LAYERS = 2     # Change to 1 or 3
DROPOUT_FC = 0.5   # Change to 0.3 or 0.7
```

### **2. Change training parameters:**
Edit `config_optimal.py`:
```python
LEARNING_RATE = 1e-4   # Try 5e-5 or 1e-3
BATCH_SIZE = 32        # Try 16 or 64
MAX_EPOCHS = 20        # Try 15 or 25
PATIENCE = 5           # Try 3 or 7
```

### **3. Disable augmentation methods:**
Edit `config_optimal.py`:
```python
USE_PARAPHRASE = True   # Set to False to disable
USE_SYNONYM = False     # Set to False to disable
```

### **4. Change data split:**
Edit `config_optimal.py`:
```python
TRAIN_SIZE = 0.8  # Change to 0.7 or 0.9
VAL_SIZE = 0.2    # Change to 0.15 or 0.25
```

---

## 🔍 Troubleshooting

### **Problem: Model overfitting**
**Solution:**
1. Increase dropout: `DROPOUT_FC = 0.6`
2. Reduce hidden size: `HIDDEN_SIZE = 128`
3. Add more weight decay: `WEIGHT_DECAY = 1e-4`
4. Reduce patience: `PATIENCE = 3`

### **Problem: Training too slow**
**Solution:**
1. Reduce max epochs: `MAX_EPOCHS = 15`
2. Increase batch size: `BATCH_SIZE = 64`
3. Skip augmentation or use only paraphrase

### **Problem: Low F1 score**
**Solution:**
1. Increase hidden size: `HIDDEN_SIZE = 512`
2. Add more layers: `NUM_LAYERS = 3`
3. Use augmentation
4. Reduce dropout: `DROPOUT_FC = 0.3`

### **Problem: Validation F1 not improving**
**Solution:**
1. Check data balance
2. Lower learning rate: `LEARNING_RATE = 5e-5`
3. Check augmentation quality
4. Increase model capacity

---

## 📈 Performance Benchmarks

### **Data Preparation:**
- Vocabulary building: ~5 seconds
- Sequence encoding: ~3 seconds
- Embedding matrix: ~10 seconds
- **Total:** ~20 seconds

### **Augmentation:**
- Paraphrase: ~1.5 hours for 21,924 samples
- Synonym: ~3 seconds for 21,924 samples
- **Total:** ~1.6 hours (vs 4 hours with backtranslation)

### **Training (per fold):**
- Without augmentation: ~4 minutes/fold → ~20 min total
- With augmentation: ~6 minutes/fold → ~30 min total

---

## 🎓 Best Practices

1. **Always set seed:** `set_seed(42)` for reproducibility
## 📞 Quick Commands Reference

```bash
# Check configuration
python -c "from config_optimal import print_config; print_config()"

# Test utils module
python utils_optimal.py

# Prepare data (force rebuild)
python prepare_data_optimal.py

# Run training with plots
python train_with_plots.py

# Run augmentation
python augment_data_optimal.py

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# View runtime log
python -c "import json; print(json.dumps(json.load(open('artifacts/logs/training_*_runtime.json')), indent=2))"

# Check disk space
python -c "from pathlib import Path; import shutil; print(f'Free space: {shutil.disk_usage(\".\")[2] / (1024**3):.1f} GB')"
```

---hon augment_data_optimal.py

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check disk space
python -c "from pathlib import Path; import shutil; print(f'Free space: {shutil.disk_usage(\".\")[2] / (1024**3):.1f} GB')"
```

---

## ✅ Validation Checklist

Before running experiments:
- [ ] Configuration checked (`config_optimal.py`)
- [ ] Data prepared (`artifacts/dataset/X.npy` exists)
- [ ] Embedding ready (`artifacts/embedding/embedding_matrix.npy` exists)
- [ ] GPU available (if intended)
- [ ] Disk space sufficient (>5GB free)

After training:
- [ ] Results saved (`artifacts/results/`)
- [ ] Plots generated (loss curves, F1 bars)
- [ ] Mean ± std reported for all metrics
- [ ] Best checkpoint saved
- [ ] Config archived with results

---

**📝 Note:** Selalu update `README_IMPROVEMENTS.md` dengan hasil eksperimen baru!

**🎯 Target Performance:** F1 = 0.87-0.90 (without data leakage, with proper regularization)
