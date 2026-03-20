"""
Inference script for fine-tuned Swin Transformer model
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, SwinForImageClassification
import argparse
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["benign", "malignant"]

print(f"Using device: {DEVICE}")


def load_finetuned_model(model_path):
    """Load fine-tuned model and processor"""
    model_path = Path(model_path)
    
    print(f"Loading model from {model_path}...")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = SwinForImageClassification.from_pretrained(model_path)
    model = model.to(DEVICE)
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model, processor


def predict_numpy_array(img_array, model, processor):
    """Predict on numpy array"""
    try:
        # Normalize if needed
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        elif img_array.max() > 1.0:
            img_array = img_array.astype(np.float32) / 255.0
        
        # Convert to PIL Image
        if img_array.ndim == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        
        if img_array.shape[-1] != 3:
            img_array = img_array[:, :, :3]
        
        img_array = (img_array * 255).astype(np.uint8)
        image = Image.fromarray(img_array).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()
        
        return LABELS[predicted_class], confidence
    except Exception as e:
        print(f"Error processing array: {e}")
        return None, None


def predict_image_file(image_path, model, processor):
    """Predict on image file"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()
        
        return LABELS[predicted_class], confidence
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def batch_predict_npy_folder(folder_path, model, processor):
    """Batch predict on numpy files"""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return {}
    
    results = {
        "benign": [],
        "malignant": [],
        "error": []
    }
    
    npy_files = sorted(folder_path.glob("*.npy"))
    
    print(f"Found {len(npy_files)} numpy files in {folder_path}")
    
    for i, npy_file in enumerate(npy_files, 1):
        try:
            img_array = np.load(npy_file)
            prediction, confidence = predict_numpy_array(img_array, model, processor)
            
            if prediction:
                results[prediction].append({
                    "file": npy_file.name,
                    "confidence": confidence
                })
                print(f"[{i}/{len(npy_files)}] {npy_file.name} → {prediction.upper()} ({confidence:.2%})")
            else:
                results["error"].append(npy_file.name)
        except Exception as e:
            results["error"].append(npy_file.name)
            print(f"[{i}/{len(npy_files)}] {npy_file.name} → ERROR: {e}")
    
    return results


def batch_predict_image_folder(folder_path, model, processor, file_extensions=(".jpg", ".jpeg", ".png")):
    """Batch predict on image files"""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return {}
    
    results = {
        "benign": [],
        "malignant": [],
        "error": []
    }
    
    image_files = [
        f for f in folder_path.iterdir()
        if f.suffix.lower() in file_extensions
    ]
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    for i, image_file in enumerate(image_files, 1):
        prediction, confidence = predict_image_file(str(image_file), model, processor)
        
        if prediction:
            results[prediction].append({
                "file": image_file.name,
                "confidence": confidence
            })
            print(f"[{i}/{len(image_files)}] {image_file.name} → {prediction.upper()} ({confidence:.2%})")
        else:
            results["error"].append(image_file.name)
    
    return results


def print_summary(results):
    """Print prediction summary"""
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"✓ Benign: {len(results['benign'])} images")
    print(f"✗ Malignant: {len(results['malignant'])} images")
    print(f"⚠ Errors: {len(results['error'])} images")
    print("="*60)
    
    if results['benign']:
        print("\nBENIGN (Normal):")
        for item in results['benign']:
            print(f"  - {item['file']}: {item['confidence']:.2%}")
    
    if results['malignant']:
        print("\nMALIGNANT (Cancer):")
        for item in results['malignant']:
            print(f"  - {item['file']}: {item['confidence']:.2%}")


def main(args):
    model, processor = load_finetuned_model(args.model_path)
    
    print("\n" + "="*60)
    print("FINE-TUNED MODEL INFERENCE")
    print("="*60)
    
    if args.image:
        print(f"\nInferring on single image: {args.image}")
        prediction, confidence = predict_image_file(args.image, model, processor)
        if prediction:
            print(f"Prediction: {prediction.upper()}")
            print(f"Confidence: {confidence:.2%}")
    
    elif args.npy_folder:
        print(f"\nInferring on numpy files in: {args.npy_folder}")
        results = batch_predict_npy_folder(args.npy_folder, model, processor)
        print_summary(results)
    
    elif args.image_folder:
        print(f"\nInferring on images in: {args.image_folder}")
        results = batch_predict_image_folder(args.image_folder, model, processor)
        print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image file")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="Path to folder with image files")
    parser.add_argument("--npy_folder", type=str, default=None,
                        help="Path to folder with numpy files")
    
    args = parser.parse_args()
    
    if not any([args.image, args.image_folder, args.npy_folder]):
        print("ERROR: Provide --image, --image_folder, or --npy_folder")
        exit(1)
    
    main(args)
