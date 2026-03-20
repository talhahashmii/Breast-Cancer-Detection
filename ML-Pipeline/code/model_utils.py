"""
Utility functions for breast cancer detection model
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path


def load_model_and_processor(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load model and processor from pretrained path or Hub
    
    Args:
        model_path: Model path or Hugging Face model ID
        device: Device to load model to ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, processor, device)
    """
    from transformers import AutoImageProcessor, SwinForImageClassification
    
    print(f"Loading model from {model_path}...")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = SwinForImageClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model, processor, device


def normalize_image(img_array):
    """
    Normalize image array to [0, 1] range
    
    Args:
        img_array: Numpy array of image
    
    Returns:
        Normalized array
    """
    if img_array.dtype == np.uint8:
        return img_array.astype(np.float32) / 255.0
    elif img_array.max() > 1.0:
        return img_array.astype(np.float32) / 255.0
    return img_array.astype(np.float32)


def convert_to_rgb(img_array):
    """
    Convert image array to RGB format
    
    Args:
        img_array: Numpy array (can be grayscale or RGB)
    
    Returns:
        RGB numpy array
    """
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Ensure 3 channels
    if img_array.shape[-1] != 3:
        img_array = img_array[:, :, :3]
    
    return img_array


def array_to_pil_image(img_array):
    """
    Convert numpy array to PIL Image
    
    Args:
        img_array: Normalized numpy array [0, 1] or [0, 255]
    
    Returns:
        PIL Image
    """
    # Normalize if needed
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)
    
    # Convert to RGB
    img_array = convert_to_rgb(img_array)
    
    return Image.fromarray(img_array).convert("RGB")


def load_npy_image(npy_path):
    """
    Load and prepare numpy image for inference
    
    Args:
        npy_path: Path to .npy file
    
    Returns:
        PIL Image ready for processing
    """
    img_array = np.load(npy_path)
    img_array = normalize_image(img_array)
    img_array = convert_to_rgb(img_array)
    return array_to_pil_image(img_array)


def predict_single(image_path, model, processor, device, labels=["benign", "malignant"]):
    """
    Predict on a single image
    
    Args:
        image_path: Path to image file
        model: Loaded model
        processor: Image processor
        device: Device to run on
        labels: Class labels
    
    Returns:
        Tuple of (prediction, confidence)
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()
        
        return labels[predicted_class], confidence
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return None, None


def get_class_distribution(labels_list):
    """
    Get distribution of classes
    
    Args:
        labels_list: List of class indices
    
    Returns:
        Dictionary with class counts
    """
    unique, counts = np.unique(labels_list, return_counts=True)
    return {i: count for i, count in zip(unique, counts)}


def print_section(title, width=70):
    """Print formatted section header"""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)
