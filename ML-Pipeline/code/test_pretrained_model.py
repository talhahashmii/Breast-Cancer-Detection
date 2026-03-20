"""
Test the pre-trained Swin Transformer model from Hugging Face
for breast cancer detection (Benign vs Malignant classification)
"""

import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, SwinForImageClassification
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = "Koushim/breast-cancer-swin-classifier"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["benign", "malignant"]

print(f"Using device: {DEVICE}")


def load_model_and_processor():
    """Load the pre-trained model and image processor"""
    print(f"Loading model from {MODEL_PATH}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = SwinForImageClassification.from_pretrained(MODEL_PATH)
    model = model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully!")
    return model, processor


def predict_single_image(image_path, model, processor):
    """Predict on a single image"""
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


def batch_predict_folder(folder_path, model, processor, file_extensions=(".jpg", ".jpeg", ".png")):
    """
    Predict on all images in a folder
    
    Args:
        folder_path: Path to folder containing images
        model: Loaded model
        processor: Image processor
        file_extensions: Tuple of allowed file extensions
    
    Returns:
        Dictionary with results
    """
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
    
    print(f"\nFound {len(image_files)} images in {folder_path}")
    
    for i, image_file in enumerate(image_files, 1):
        prediction, confidence = predict_single_image(str(image_file), model, processor)
        
        if prediction:
            results[prediction].append({
                "file": image_file.name,
                "confidence": confidence
            })
            print(f"[{i}/{len(image_files)}] {image_file.name} → {prediction.upper()} ({confidence:.2%})")
        else:
            results["error"].append(image_file.name)
            print(f"[{i}/{len(image_files)}] {image_file.name} → ERROR")
    
    return results



def print_results_summary(results):
    """Print summary of batch predictions"""
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
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


def main():
    """Main testing function"""
    model, processor = load_model_and_processor()
    
    print("\n" + "="*60)
    print("BREAST CANCER DETECTION - PRE-TRAINED MODEL TEST")
    print("="*60)
    
    # Test 1: Single image test (if exists)
    single_image_path = "test_image.png"
    if os.path.exists(single_image_path):
        print(f"\n[TEST 1] Single image prediction:")
        prediction, confidence = predict_single_image(single_image_path, model, processor)
        if prediction:
            print(f"  Image: {single_image_path}")
            print(f"  Prediction: {prediction.upper()}")
            print(f"  Confidence: {confidence:.2%}")
        else:
            print(f"  Could not process {single_image_path}")
    
    # Test 2: Batch prediction from folder (if exists)
    test_folder = "test_images"
    if os.path.exists(test_folder) and os.path.isdir(test_folder):
        print(f"\n[TEST 2] Batch prediction from folder:")
        results = batch_predict_folder(test_folder, model, processor)
        print_results_summary(results)
    
    print("\n" + "="*60)
    print("✓ Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
