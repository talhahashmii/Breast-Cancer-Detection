"""
Fine-tune the Swin Transformer model on preprocessed breast cancer dataset
Uses the pre-trained Koushim/breast-cancer-swin-classifier model
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, SwinForImageClassification
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import json as json_lib

# Configuration
MODEL_PATH = "Koushim/breast-cancer-swin-classifier"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["benign", "malignant"]

print(f"Using device: {DEVICE}")


class NumpyBreastCancerDataset(Dataset):
    """Dataset for numpy-based breast cancer images"""
    
    def __init__(self, data_dir, processor, label_mapping=None, transform=None):
        """
        Args:
            data_dir: Path to folder containing .npy files
            processor: Hugging Face image processor
            label_mapping: Dict mapping class names to labels
            transform: Optional torchvision transforms
        """
        self.processor = processor
        self.transform = transform
        self.images = []
        self.labels = []
        
        if label_mapping is None:
            label_mapping = {"benign": 0, "malignant": 1}
        
        data_dir = Path(data_dir)
        
        # Load all .npy files
        for npy_file in sorted(data_dir.glob("*.npy")):
            self.images.append(str(npy_file))
            # Infer label from filename or directory
            if "benign" in str(npy_file).lower() or "normal" in str(npy_file).lower():
                self.labels.append(0)
            elif "malignant" in str(npy_file).lower() or "cancer" in str(npy_file).lower():
                self.labels.append(1)
            else:
                # Default to 0 if unclear
                self.labels.append(0)
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
        if len(self.labels) > 0:
            print(f"  - Benign: {sum(1 for l in self.labels if l == 0)}")
            print(f"  - Malignant: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        npy_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Load numpy array
            img_array = np.load(npy_path)
            
            # Normalize if needed
            if img_array.dtype == np.uint8:
                img_array = img_array.astype(np.float32) / 255.0
            elif img_array.max() > 1.0:
                img_array = img_array.astype(np.float32) / 255.0
            
            # Convert to PIL Image for processing
            if img_array.ndim == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Ensure 3-channel
            if img_array.shape[-1] != 3:
                img_array = img_array[:, :, :3]
            
            # Convert to uint8 for PIL
            img_array = (img_array * 255).astype(np.uint8)
            image = Image.fromarray(img_array).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            # Process image with Hugging Face processor
            inputs = self.processor(images=image, return_tensors="pt")
            
            return {
                "pixel_values": inputs["pixel_values"].squeeze(),
                "labels": torch.tensor(label)
            }
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            # Return dummy data on error
            return {
                "pixel_values": torch.zeros(3, 224, 224),
                "labels": torch.tensor(0)
            }


def get_data_augmentation():
    """Define data augmentation transforms"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            "loss": loss.item(),
            "acc": correct / total
        })
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            preds = outputs.logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({
                "loss": loss.item(),
                "acc": correct / total
            })
    
    return total_loss / len(val_loader), correct / total, predictions, true_labels


def main(args):
    """Main fine-tuning function"""
    
    print("\n" + "="*70)
    print("BREAST CANCER DETECTION - FINE-TUNING PRE-TRAINED MODEL")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and processor
    print(f"\nLoading base model: {MODEL_PATH}")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = SwinForImageClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model = model.to(DEVICE)
    print("✓ Model loaded successfully!")
    
    # Prepare datasets
    print(f"\nPreparing datasets from {args.data_dir}")
    
    augmentation = get_data_augmentation() if args.augmentation else None
    
    train_dataset = NumpyBreastCancerDataset(
        args.data_dir,
        processor,
        transform=augmentation
    )
    
    if len(train_dataset) == 0:
        print("ERROR: No training data found!")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Optional: Load validation dataset
    val_loader = None
    if args.val_dir and os.path.exists(args.val_dir):
        val_dataset = NumpyBreastCancerDataset(args.val_dir, processor)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
            print(f"✓ Validation set loaded with {len(val_dataset)} samples")
        else:
            print("⚠ Validation set is empty")
    
    # Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*70)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "timestamp": datetime.now().isoformat()
    }
    
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validate
        if val_loader:
            val_loss, val_acc, _, _ = evaluate(model, val_loader, DEVICE)
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(output_dir / "best_model")
                processor.save_pretrained(output_dir / "best_model")
                print(f"  ✓ Best model saved! (Acc: {val_acc:.4f})")
    
    # Save final model
    model.save_pretrained(output_dir / "final_model")
    processor.save_pretrained(output_dir / "final_model")
    
    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json_lib.dump(history, f, indent=4)
    
    print("\n" + "="*70)
    print("✓ Fine-tuning complete!")
    print(f"✓ Models saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Swin Transformer for breast cancer detection")
    parser.add_argument("--data_dir", type=str, default="../Data/Preprocessed Data/train",
                        help="Path to training data (numpy files)")
    parser.add_argument("--val_dir", type=str, default="../Data/Preprocessed Data/val",
                        help="Path to validation data (numpy files)")
    parser.add_argument("--output_dir", type=str, default="../Model/finetuned_swin",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--augmentation", action="store_true", default=True,
                        help="Enable data augmentation")
    
    args = parser.parse_args()
    
    main(args)
