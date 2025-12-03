"""
Training Script dengan Auto-Plotting dan Runtime Logging
=========================================================
Example script demonstrating automatic plot generation and runtime tracking
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

# Import configurations dan utilities
from config_optimal import *
from utils_optimal import *
from prepare_data_optimal import prepare_dataset


# ============================================================================
# DATASET CLASS
# ============================================================================

class NewsDataset(Dataset):
    """Simple dataset for news headlines"""
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class ImprovedLSTMClassifier(nn.Module):
    """Improved LSTM with dropout and bidirectional"""
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 embedding_matrix=None, num_layers=2, dropout=0.5, bidirectional=True):
        super().__init__()
        
        # Embedding layer
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix), freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.emb_dropout = nn.Dropout(DROPOUT_EMBEDDING)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=DROPOUT_RNN if num_layers > 1 else 0
        )
        
        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, num_classes)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = self.emb_dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state (for bidirectional, concat forward and backward)
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Fully connected
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


# ============================================================================
# MAIN TRAINING LOOP WITH AUTO-PLOTTING
# ============================================================================

def train_with_plotting(use_augmentation: bool = False):
    """
    Complete training pipeline with automatic plotting and runtime logging
    
    Args:
        use_augmentation: Whether to use augmented dataset
    """
    print("="*80)
    print("TRAINING WITH AUTO-PLOTTING & RUNTIME LOGGING")
    print("="*80)
    
    # Set seed
    set_seed(RANDOM_STATE)
    
    # Load data
    print("\n[1/6] Loading data...")
    X = np.load(DATASET_ARTIFACTS_DIR / "X.npy")
    y = np.load(DATASET_ARTIFACTS_DIR / "y.npy")
    embedding_matrix = np.load(EMBEDDING_DIR / "embedding_matrix.npy")
    config = load_json(CONFIG_DIR / "config.json")
    
    print(f"  Dataset: {X.shape}, Classes: {config['num_classes']}")
    
    # Setup device
    device = get_device()
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  {get_memory_usage()}")
    
    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"training_{timestamp}"
    print(f"\n[2/6] Experiment: {experiment_name}")
    
    # Initialize runtime logger
    runtime_logger = RuntimeLogger(experiment_name)
    runtime_logger.start_training()
    
    # K-Fold setup
    print(f"\n[3/6] Setting up {N_SPLITS}-Fold Cross-Validation...")
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # Storage for results
    all_fold_metrics = []
    all_fold_histories = []
    
    # Training loop
    print(f"\n[4/6] Training {N_SPLITS} folds...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{N_SPLITS}")
        print(f"{'='*60}")
        
        # Split data
        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]
        
        # Further split train into train/val
        val_size = int(len(X_train_val) * VAL_SIZE)
        X_train, X_val = X_train_val[:-val_size], X_train_val[-val_size:]
        y_train, y_val = y_train_val[:-val_size], y_train_val[-val_size:]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create dataloaders
        train_dataset = NewsDataset(X_train, y_train)
        val_dataset = NewsDataset(X_val, y_val)
        test_dataset = NewsDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = ImprovedLSTMClassifier(
            vocab_size=config['vocab_size'],
            embed_dim=config['embedding_dim'],
            hidden_size=HIDDEN_SIZE,
            num_classes=config['num_classes'],
            embedding_matrix=embedding_matrix,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_FC,
            bidirectional=BIDIRECTIONAL
        ).to(device)
        
        print(f"Model parameters: {count_parameters(model):,}")
        
        # Optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=LR_FACTOR, 
            patience=LR_PATIENCE, min_lr=LR_MIN, verbose=True
        )
        
        # Training history for this fold
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        best_val_f1 = 0
        patience_counter = 0
        
        # Epoch loop
        for epoch in range(MAX_EPOCHS):
            # Train
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1'])
            
            # Log to runtime logger
            memory_gb = None
            if device.type == 'cuda':
                memory_gb = torch.cuda.memory_allocated() / (1024**3)
            
            # Print progress
            print(f"Epoch {epoch+1:2d}/{MAX_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Save best checkpoint
                checkpoint_path = CHECKPOINT_DIR / f"{experiment_name}_fold{fold_idx}_best.pth"
                save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
            else:
                patience_counter += 1
            
            # Scheduler step
            scheduler.step(val_metrics['f1'])
            
            # Early stopping check
            if EARLY_STOPPING and patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model for testing
        checkpoint_path = CHECKPOINT_DIR / f"{experiment_name}_fold{fold_idx}_best.pth"
        load_checkpoint(model, None, checkpoint_path, device)
        
        # Test
        test_metrics = validate(model, test_loader, criterion, device)
        print(f"\nFold {fold_idx} Test Results:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        
        # Store results
        all_fold_metrics.append(test_metrics)
        all_fold_histories.append(history)
        
        # 🎨 PLOT THIS FOLD'S TRAINING HISTORY
        print(f"\n📊 Generating plot for Fold {fold_idx}...")
        fold_plot_path = PLOTS_DIR / f"{experiment_name}_fold{fold_idx}_history"
        plot_training_history(
            history, 
            save_path=fold_plot_path,
            save_formats=SAVE_PLOT_FORMATS,
            dpi=PLOT_DPI,
            title_prefix=f"Fold {fold_idx}"
        )
        
        # Clear memory
        del model, optimizer, scheduler
        clear_memory()
    
    # End runtime logging
    runtime_logger.end_training()
    
    # 🎨 PLOT ALL FOLDS COMPARISON
    print(f"\n[5/6] Generating comparison plots...")
    
    # K-Fold results bar chart
    kfold_plot_path = PLOTS_DIR / f"{experiment_name}_kfold_results"
    plot_kfold_results(
        all_fold_metrics,
        save_path=kfold_plot_path,
        save_formats=SAVE_PLOT_FORMATS,
        dpi=PLOT_DPI
    )
    
    # All folds training curves
    comparison_plot_path = PLOTS_DIR / f"{experiment_name}_all_folds_comparison"
    plot_fold_comparison(
        all_fold_histories,
        save_path=comparison_plot_path,
        save_formats=SAVE_PLOT_FORMATS,
        dpi=PLOT_DPI
    )
    
    # 📝 SAVE RUNTIME LOGS
    print(f"\n[6/6] Saving runtime logs...")
    log_path = LOGS_DIR / f"{experiment_name}_runtime.json"
    runtime_logger.save(log_path)
    
    # Aggregate and save final results
    aggregated = aggregate_metrics(all_fold_metrics)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE ✅")
    print("="*80)
    print(f"\n📊 Final Results (Mean ± Std):")
    for metric, (mean, std) in aggregated.items():
        if metric != 'loss':
            print(f"  {metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")
    
    # Save final results
    final_results = {
        'experiment_name': experiment_name,
        'config': get_config_dict(),
        'aggregated_metrics': {k: {'mean': v[0], 'std': v[1]} 
                              for k, v in aggregated.items()},
        'fold_metrics': all_fold_metrics,
        'timestamp': timestamp
    }
    
    results_path = RESULTS_DIR / f"{experiment_name}_results.json"
    save_json(final_results, results_path)
    
    print(f"\n📁 Saved Artifacts:")
    print(f"  ✅ Plots: {PLOTS_DIR}")
    print(f"  ✅ Logs: {log_path}")
    print(f"  ✅ Results: {results_path}")
    print(f"  ✅ Checkpoints: {CHECKPOINT_DIR}")
    
    print("\n" + "="*80)
    
    return final_results


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    # Ensure data is prepared
    print("Checking if data is prepared...")
    if not (DATASET_ARTIFACTS_DIR / "X.npy").exists():
        print("Data not found. Running data preparation...")
        prepare_dataset()
    
    # Run training with plotting
    results = train_with_plotting(use_augmentation=False)
    
    print("\n🎉 All done! Check artifacts/plots/ for visualizations!")
