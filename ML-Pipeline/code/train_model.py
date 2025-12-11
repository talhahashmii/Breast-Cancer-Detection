"""
Training script for the single-view CNN model
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from model import create_single_view_model
import matplotlib.pyplot as plt


def load_preprocessed_data(data_dir=r"..\Data\Preprocessed Data\ROI"):
    """
    Load preprocessed images and labels from disk
    
    Args:
        data_dir: directory containing roi_preprocessed_images.npy and roi_preprocessed_labels.npy
    
    Returns:
        images: numpy array of preprocessed images
        labels: numpy array of labels (0=benign, 1=malignant)
    """
    print("Loading preprocessed data...")
    print("=" * 60)
    
    try:
        images = np.load(os.path.join(data_dir, 'roi_preprocessed_images.npy'))
        labels = np.load(os.path.join(data_dir, 'roi_preprocessed_labels.npy'))
        
        print(f"✓ Images loaded: {images.shape}")
        print(f"✓ Labels loaded: {labels.shape}")
        print(f"✓ Benign: {sum(labels == 0)}")
        print(f"✓ Malignant: {sum(labels == 1)}")
        
        return images, labels
    except FileNotFoundError:
        print(f"✗ Error: Could not find preprocessed data in {data_dir}")
        print("Please run batch_roi_preprocessing.py first")
        return None, None


def prepare_data(images, labels, test_size=0.2, val_size=0.1):
    """
    Prepare data for training: split into train/val/test and convert labels
    
    Args:
        images: numpy array of images
        labels: numpy array of labels
        test_size: fraction for test set
        val_size: fraction for validation set (from training data)
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\nPreparing data...")
    print("=" * 60)
    
    # Add channel dimension if needed
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)
        print(f"✓ Added channel dimension: {images.shape}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    
    print(f"✓ Train set: {X_train.shape} with labels {y_train.shape}")
    print(f"✓ Validation set: {X_val.shape} with labels {y_val.shape}")
    print(f"✓ Test set: {X_test.shape} with labels {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """
    Train the model
    
    Args:
        model: compiled Keras model
        X_train: training images
        y_train: training labels (one-hot encoded)
        X_val: validation images
        y_val: validation labels (one-hot encoded)
        epochs: number of training epochs
        batch_size: batch size for training
    
    Returns:
        history: training history object
    """
    print("\nTraining model...")
    print("=" * 60)
    
    # Calculate class weights to handle imbalance
    y_train_orig = np.argmax(y_train, axis=1)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_orig),
        y=y_train_orig
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"✓ Class weights: Benign={class_weight_dict[0]:.4f}, Malignant={class_weight_dict[1]:.4f}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set
    
    Args:
        model: trained Keras model
        X_test: test images
        y_test: test labels (one-hot encoded)
    
    Returns:
        loss, accuracy
    """
    print("\nEvaluating model on test set...")
    print("=" * 60)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"✓ Test Loss: {loss:.4f}")
    print(f"✓ Test Accuracy: {accuracy:.4f}")
    
    return loss, accuracy


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: training history object
        save_path: where to save the plot
    """
    print("\nPlotting training history...")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Training history plot saved to {save_path}")


def save_model(model, save_path=r"D:\Hashmi FYP\BREAST CANCER DETECTION\ML-Pipeline\Model\breast_cancer_model.h5"):
    """
    Save trained model to disk
    
    Args:
        model: trained Keras model
        save_path: where to save the model
    """
    print("\nSaving model...")
    print("=" * 60)
    
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as .keras format instead of .h5 to avoid Lambda layer issues
    keras_path = save_path.replace('.h5', '.keras')
    model.save(keras_path)
    print(f"✓ Model saved to {keras_path}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TRAINING SINGLE-VIEW CNN MODEL FOR BREAST CANCER DETECTION")
    print("=" * 80)
    
    # Step 1: Load preprocessed data
    images, labels = load_preprocessed_data()
    
    if images is None:
        print("\n✗ Cannot proceed without preprocessed data")
        exit(1)
    
    # Step 2: Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(images, labels)
    
    # Step 3: Create model
    print("\nCreating model...")
    print("=" * 60)
    model = create_single_view_model()
    
    # Step 4: Train model
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=20,
        batch_size=32
    )
    
    # Step 5: Evaluate model
    test_loss, test_accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 6: Plot training history
    plot_training_history(history)
    
    # Step 7: Save model
    save_model(model)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print("\n" + "=" * 80 + "\n")
