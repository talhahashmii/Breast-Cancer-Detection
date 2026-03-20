"""
Phase 5: Model Training Script
Breast Cancer Detection - Dual-View CNN with ResNet50

This script trains the dual-view CNN model on the preprocessed dataset.
Supports both local training and Google Colab with GPU acceleration.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_architecture import DualViewCNN, DualViewMammographyDataset, create_dual_view_model


# ==================== CONFIGURATION ====================

class Config:
    """Training configuration"""
    # Model parameters
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.5
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    # Data
    TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% val from training set
    
    # Checkpointing
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_NAME = "best_model.pt"
    LAST_MODEL_NAME = "last_model.pt"
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== UTILITY FUNCTIONS ====================

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_subheader(title):
    """Print formatted subsection header"""
    print(f"\n{title}")
    print("-"*80)


def load_preprocessed_data(data_path):
    """Load preprocessed data from metadata.json"""
    metadata_path = os.path.join(data_path, "metadata.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def calculate_class_weights(labels):
    """
    Calculate class weights to handle imbalanced dataset
    
    Args:
        labels: Array of class labels
    
    Returns:
        weights: Tensor of class weights
    """
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float32)


def setup_training(data_path, config):
    """Setup training components"""
    print_header("SETTING UP TRAINING")
    
    # Load metadata
    print_subheader("Loading Metadata")
    metadata = load_preprocessed_data(data_path)
    
    train_images = metadata['train']['images']
    train_labels = np.array(metadata['train']['labels'])
    val_images = metadata['val']['images']
    val_labels = np.array(metadata['val']['labels'])
    test_images = metadata['test']['images']
    test_labels = np.array(metadata['test']['labels'])
    
    print(f"  [OK] Training samples: {len(train_labels)}")
    print(f"  [OK] Validation samples: {len(val_labels)}")
    print(f"  [OK] Test samples: {len(test_labels)}")
    
    # Create datasets
    print_subheader("Creating Datasets")
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')
    
    train_dataset = DualViewMammographyDataset(train_images, train_labels, train_dir)
    val_dataset = DualViewMammographyDataset(val_images, val_labels, val_dir)
    test_dataset = DualViewMammographyDataset(test_images, test_labels, test_dir)
    
    print(f"  [OK] Datasets created")
    
    # Create data loaders
    print_subheader("Creating Data Loaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Colab compatibility
        pin_memory=(config.DEVICE == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.DEVICE == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.DEVICE == 'cuda')
    )
    
    print(f"  [OK] Data loaders created with batch size: {config.BATCH_SIZE}")
    print(f"      Training batches: {len(train_loader)}")
    print(f"      Validation batches: {len(val_loader)}")
    print(f"      Test batches: {len(test_loader)}")
    
    # Calculate class weights
    print_subheader("Calculating Class Weights")
    class_weights = calculate_class_weights(train_labels)
    print(f"  [OK] Class weights: {class_weights.tolist()}")
    print(f"      Class 0 (Benign): {class_weights[0]:.4f}")
    print(f"      Class 1 (Malignant): {class_weights[1]:.4f}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'class_weights': class_weights,
        'metadata': metadata,
        'num_train': len(train_labels),
        'num_val': len(val_labels),
        'num_test': len(test_labels)
    }


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        avg_acc = 100 * correct / total
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.2f}%'})
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, config, output_dir):
    """Train the model with early stopping"""
    print_header("TRAINING MODEL")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )
    
    # Create checkpoint directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print_subheader("Training Progress")
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE, config)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, config.DEVICE)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            best_model_path = os.path.join(output_dir, config.BEST_MODEL_NAME)
            torch.save(model.state_dict(), best_model_path)
            print(f"  [SAVE] Best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n  [EARLY STOP] No improvement for {config.EARLY_STOPPING_PATIENCE} epochs")
                print(f"  Best model was at epoch {best_epoch+1}")
                break
        
        # Save last model
        last_model_path = os.path.join(output_dir, config.LAST_MODEL_NAME)
        torch.save(model.state_dict(), last_model_path)
    
    total_time = time.time() - start_time
    
    print_subheader("Training Complete")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Best epoch: {best_epoch+1}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    
    return history


def evaluate_model(model, test_loader, config):
    """Evaluate model on test set"""
    print_header("EVALUATING MODEL")
    
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    print_subheader("Test Results")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {correct}/{total}")
    
    # Calculate per-class metrics
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    
    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print_subheader("Confusion Matrix")
    print(f"  {cm}")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': all_preds,
        'labels': all_labels,
        'confusion_matrix': cm.tolist(),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_training_report(history, eval_results, output_dir, config):
    """Save training report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': config.BATCH_SIZE,
            'num_epochs': config.NUM_EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'dropout_rate': config.DROPOUT_RATE
        },
        'history': history,
        'evaluation': eval_results
    }
    
    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[OK] Training report saved to {report_path}")
    
    return report


def plot_training_history(history, output_dir):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Training', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Validation', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Training', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Validation', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rates'], marker='o', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Empty subplot for balance
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5, 'Training Complete', 
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Training history plot saved to {plot_path}")
    plt.close()


# ==================== MAIN TRAINING FUNCTION ====================

def main(data_path, output_dir=None, config=None):
    """
    Main training function
    
    Args:
        data_path: Path to preprocessed data directory
        output_dir: Directory to save outputs (default: 'training_output')
        config: Training configuration (default: Config class)
    """
    if config is None:
        config = Config()
    
    if output_dir is None:
        output_dir = 'training_output'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print_header("BREAST CANCER DETECTION - MODEL TRAINING")
    print(f"  Device: {config.DEVICE}")
    print(f"  Data path: {data_path}")
    print(f"  Output directory: {output_dir}")
    
    # Setup training
    training_data = setup_training(data_path, config)
    
    # Create model
    print_header("CREATING MODEL")
    model = create_dual_view_model(
        device=config.DEVICE,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    
    # Train model
    history = train_model(
        model,
        training_data['train_loader'],
        training_data['val_loader'],
        config,
        output_dir
    )
    
    # Load best model
    best_model_path = os.path.join(output_dir, config.BEST_MODEL_NAME)
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on test set
    eval_results = evaluate_model(model, training_data['test_loader'], config)
    
    # Save reports
    report = save_training_report(history, eval_results, output_dir, config)
    plot_training_history(history, output_dir)
    
    print_header("TRAINING FINISHED")
    print(f"  Final test accuracy: {eval_results['accuracy']:.2f}%")
    print(f"  Output directory: {output_dir}")
    
    return model, history, eval_results


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train dual-view CNN model')
    
    # Set default paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(os.path.dirname(script_dir), 'Data', 'Preprocessed Data')
    default_output_dir = os.path.join(script_dir, 'training_output')
    
    parser.add_argument('--data_path', type=str, default=default_data_path,
                       help='Path to preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                       help='Directory to save training outputs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Normalize paths
    args.data_path = os.path.normpath(args.data_path)
    args.output_dir = os.path.normpath(args.output_dir)
    
    print(f"Data path: {args.data_path}")
    print(f"Data exists: {os.path.exists(args.data_path)}")
    print()
    
    # Create config
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    
    # Run training
    model, history, eval_results = main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        config=config
    )
