# 📊 Auto-Plotting & Runtime Logging Guide

## 🎯 Overview

Training script sekarang **otomatis menghasilkan visualisasi** dan **mencatat runtime** untuk setiap eksperimen.

---

## 🎨 Visualizations Generated

### **1. Per-Fold Training History**

**File:** `{experiment_name}_fold{N}_history.png/pdf`

**Contents:**
- **Left plot:** Train loss vs Val loss per epoch
- **Right plot:** Validation metrics (F1, Accuracy, Precision, Recall)

**Features:**
- ✅ High-quality (300 DPI)
- ✅ Color-coded lines
- ✅ Grid for easy reading
- ✅ Clear labels and legends

**Example:**
```
training_20241204_143022_fold1_history.png
training_20241204_143022_fold2_history.png
training_20241204_143022_fold3_history.png
training_20241204_143022_fold4_history.png
training_20241204_143022_fold5_history.png
```

---

### **2. K-Fold Results Comparison**

**File:** `{experiment_name}_kfold_results.png/pdf`

**Contents:**
- Bar chart comparing all metrics across all folds
- 4 metrics: Accuracy, Precision, Recall, F1
- Value labels on each bar
- Mean lines for reference

**Use Case:**
- Quick comparison of fold performance
- Identify outlier folds
- Visual consistency check

---

### **3. All Folds Training Curves Overlay**

**File:** `{experiment_name}_all_folds_comparison.png/pdf`

**Contents:**
- 4 subplots:
  1. Train Loss - all folds overlaid
  2. Val Loss - all folds overlaid
  3. Val F1 - all folds overlaid
  4. Val Accuracy - all folds overlaid

**Use Case:**
- See training consistency across folds
- Detect if one fold is behaving differently
- Verify convergence patterns

---

## 📝 Runtime Logging

### **Log File Structure**

**File:** `{experiment_name}_runtime.json`

```json
{
    "experiment_name": "training_20241204_143022",
    "start_time": "2024-12-04 14:30:22",
    "end_time": "2024-12-04 14:52:18",
    "total_time_seconds": 1316.4,
    "total_time_formatted": "21m 56s",
    "total_epochs": 87,
    "avg_time_per_epoch": 15.13,
    "best_epoch": 12,
    "best_f1": 0.8642,
    "epoch_logs": [
        {
            "epoch": 0,
            "timestamp": 1701695422.5,
            "train_loss": 1.8234,
            "val_loss": 1.5421,
            "accuracy": 0.6234,
            "precision": 0.6189,
            "recall": 0.6312,
            "f1": 0.6247,
            "memory_gb": 1.234
        },
        ...
    ]
}
```

### **What's Logged:**

1. **Experiment metadata:**
   - Experiment name (with timestamp)
   - Start/end times
   - Total duration
   - Average time per epoch

2. **Per-epoch metrics:**
   - Epoch number
   - Timestamp
   - Train loss
   - Validation loss
   - All validation metrics (acc, prec, rec, F1)
   - GPU memory usage (if available)

3. **Best model info:**
   - Best epoch number
   - Best F1 score

---

## 🚀 Usage

### **Basic Usage (Automatic)**

```python
# Just run the training script
python train_with_plots.py
```

**Result:** Plots and logs automatically saved to:
- `artifacts/plots/`
- `artifacts/logs/`

---

### **Customizing Plot Settings**

Edit `config_optimal.py`:

```python
# Enable/disable auto-plotting
AUTO_PLOT = True  # Set to False to disable

# Plot quality
PLOT_DPI = 300  # 150 (draft), 300 (high), 600 (print)

# Save formats
SAVE_PLOT_FORMATS = ['png', 'pdf']  # Can add 'svg', 'jpg'

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # or 'default', 'ggplot'

# Runtime logging
LOG_RUNTIME = True  # Set to False to disable
LOG_MEMORY = True  # Log GPU memory usage
```

---

### **Manual Plotting (Advanced)**

```python
from utils_optimal import (
    plot_training_history, 
    plot_kfold_results,
    plot_fold_comparison,
    RuntimeLogger
)
from config_optimal import PLOTS_DIR, SAVE_PLOT_FORMATS, PLOT_DPI

# Plot single fold history
history = {
    'train_loss': [1.5, 1.2, 0.9, 0.7],
    'val_loss': [1.6, 1.3, 1.0, 0.8],
    'val_f1': [0.65, 0.75, 0.82, 0.86]
}

plot_training_history(
    history,
    save_path=PLOTS_DIR / "my_experiment_fold1",
    save_formats=SAVE_PLOT_FORMATS,
    dpi=PLOT_DPI,
    title_prefix="Fold 1"
)

# Plot K-Fold comparison
fold_metrics = [
    {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.86, 'f1': 0.85},
    {'accuracy': 0.87, 'precision': 0.86, 'recall': 0.88, 'f1': 0.87},
    # ... more folds
]

plot_kfold_results(
    fold_metrics,
    save_path=PLOTS_DIR / "kfold_comparison",
    save_formats=['png', 'pdf'],
    dpi=300
)

# Manual runtime logging
logger = RuntimeLogger("my_experiment")
logger.start_training()

# ... training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_metrics = validate()
    
    logger.log_epoch(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_metrics['loss'],
        val_metrics={'f1': val_metrics['f1'], 'acc': val_metrics['accuracy']},
        memory_gb=get_gpu_memory_gb()
    )

logger.end_training()
logger.save(LOGS_DIR / "my_experiment_runtime.json")
```

---

## 📁 Output Directory Structure

```
artifacts/
├── plots/
│   ├── training_20241204_143022_fold1_history.png
│   ├── training_20241204_143022_fold1_history.pdf
│   ├── training_20241204_143022_fold2_history.png
│   ├── training_20241204_143022_fold2_history.pdf
│   ├── training_20241204_143022_fold3_history.png
│   ├── training_20241204_143022_fold3_history.pdf
│   ├── training_20241204_143022_fold4_history.png
│   ├── training_20241204_143022_fold4_history.pdf
│   ├── training_20241204_143022_fold5_history.png
│   ├── training_20241204_143022_fold5_history.pdf
│   ├── training_20241204_143022_kfold_results.png
│   ├── training_20241204_143022_kfold_results.pdf
│   ├── training_20241204_143022_all_folds_comparison.png
│   └── training_20241204_143022_all_folds_comparison.pdf
│
├── logs/
│   └── training_20241204_143022_runtime.json
│
├── results/
│   └── training_20241204_143022_results.json
│
└── checkpoints/
    ├── training_20241204_143022_fold1_best.pth
    ├── training_20241204_143022_fold2_best.pth
    ├── training_20241204_143022_fold3_best.pth
    ├── training_20241204_143022_fold4_best.pth
    └── training_20241204_143022_fold5_best.pth
```

---

## 💡 Tips & Best Practices

### **1. Naming Conventions**

Experiments automatically named with timestamp:
```
training_YYYYMMDD_HHMMSS
```

Benefits:
- ✅ Chronological sorting
- ✅ No filename conflicts
- ✅ Easy to track experiments

---

### **2. Comparing Multiple Experiments**

Load and compare logs:

```python
import json
import pandas as pd
from pathlib import Path

# Load multiple runtime logs
logs_dir = Path("artifacts/logs")
experiments = []

for log_file in logs_dir.glob("training_*_runtime.json"):
    with open(log_file) as f:
        data = json.load(f)
        experiments.append({
            'name': data['experiment_name'],
            'time': data['total_time_formatted'],
            'best_f1': data['best_f1'],
            'best_epoch': data['best_epoch']
        })

df = pd.DataFrame(experiments)
print(df.sort_values('best_f1', ascending=False))
```

---

### **3. Plot Customization**

For publication-quality plots:

```python
# High DPI for print
PLOT_DPI = 600

# Multiple formats
SAVE_PLOT_FORMATS = ['png', 'pdf', 'svg']

# Custom style
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
```

---

### **4. Memory Tracking**

Monitor GPU memory to optimize batch size:

```python
from utils_optimal import get_memory_usage

print(get_memory_usage())
# Output: "GPU Memory - Allocated: 1.23GB, Reserved: 2.45GB"
```

If memory too high:
- Reduce `BATCH_SIZE` in `config_optimal.py`
- Reduce `HIDDEN_SIZE`
- Enable gradient checkpointing (advanced)

---

## 🔍 Analyzing Results

### **1. Check Training Curves**

Look for:
- ✅ **Smooth convergence:** Loss steadily decreasing
- ✅ **No overfitting:** Val loss tracking train loss
- ⚠️ **Early plateau:** May need more capacity or better LR
- ⚠️ **Divergence:** Val loss increasing = overfitting

---

### **2. Check K-Fold Variance**

Look for:
- ✅ **Low variance (< 2%):** Consistent model
- ⚠️ **High variance (> 5%):** Data imbalance or model instability

---

### **3. Check Runtime Efficiency**

```python
# Load runtime log
import json
with open("artifacts/logs/training_20241204_143022_runtime.json") as f:
    log = json.load(f)

print(f"Total time: {log['total_time_formatted']}")
print(f"Time per epoch: {log['avg_time_per_epoch']:.1f}s")
print(f"Best epoch: {log['best_epoch']} (saved {log['total_epochs'] - log['best_epoch']} epochs)")
```

---

## 📊 Example Workflow

```bash
# 1. Run training
python train_with_plots.py

# 2. Check plots
ls artifacts/plots/

# 3. View runtime log
cat artifacts/logs/training_*_runtime.json | jq .

# 4. Compare with previous runs
python -c "
import json
import glob
for f in glob.glob('artifacts/logs/*.json'):
    with open(f) as fp:
        d = json.load(fp)
        print(f'{d[\"experiment_name\"]}: F1={d[\"best_f1\"]:.4f}, Time={d[\"total_time_formatted\"]}')
"
```

---

## 🎯 Summary

| Feature | Benefit |
|---------|---------|
| **Auto-plotting** | No manual matplotlib code needed |
| **Multi-format** | PNG for viewing, PDF for papers, SVG for editing |
| **Runtime logging** | Track experiment efficiency |
| **Memory tracking** | Optimize batch size |
| **Timestamp naming** | No conflicts, easy sorting |
| **Per-fold plots** | Detailed analysis per fold |
| **Comparison plots** | Quick overview of all folds |

---

**🚀 Ready to train? Just run:**
```bash
python train_with_plots.py
```

**All visualizations and logs will be automatically saved! ✅**
