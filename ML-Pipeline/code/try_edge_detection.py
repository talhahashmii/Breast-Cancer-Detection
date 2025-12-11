"""
Try edge detection preprocessing - maybe edges are more discriminative than raw pixels
"""

import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = r"..\data\Dataset"

print("=" * 80)
print("EDGE DETECTION PREPROCESSING")
print("=" * 80)

def get_label_from_annotation(ann_path):
    """Extract benign/malignant label"""
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        if 'tags' in data:
            for tag in data['tags']:
                tag_name = tag.get('name', '').lower()
                if tag_name in ['benign', 'benign_without_callback']:
                    return 0
                elif tag_name == 'malignant':
                    return 1
        return None
    except:
        return None

def resize_image(image, target_size=512):
    """Resize to square"""
    height, width = image.shape
    scale = target_size / max(height, width)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

def preprocess_with_edges(img_path):
    """Use Canny edge detection instead of raw pixels"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize
    resized = resize_image(img, target_size=512)
    
    # Apply Canny edge detection
    edges = cv2.Canny(resized, 50, 150)
    
    return edges

# Process all images
all_images = []
all_labels = []
processed_count = 0

for split in ['train', 'test']:
    print(f"\nProcessing {split.upper()}...")
    
    img_dir = os.path.join(dataset_path, split, 'img')
    ann_dir = os.path.join(dataset_path, split, 'ann')
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, img_file + '.json')
        
        if not os.path.exists(ann_path):
            continue
        
        label = get_label_from_annotation(ann_path)
        if label is None:
            continue
        
        processed = preprocess_with_edges(img_path)
        
        if processed is not None:
            all_images.append(processed)
            all_labels.append(label)
            processed_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(img_files)}...")

all_images = np.array(all_images, dtype=np.uint8)
all_labels = np.array(all_labels)

print(f"\n✓ Total processed: {len(all_images)}")
print(f"✓ Benign: {sum(all_labels == 0)}")
print(f"✓ Malignant: {sum(all_labels == 1)}")

# Check class separation with edges
benign_mean = all_images[all_labels == 0].mean()
malignant_mean = all_images[all_labels == 1].mean()
difference = abs(benign_mean - malignant_mean)
percent_diff = (difference / (benign_mean + 0.001)) * 100

print(f"\nEdge detection class separation:")
print(f"  Benign mean: {benign_mean:.2f}")
print(f"  Malignant mean: {malignant_mean:.2f}")
print(f"  Difference: {difference:.2f} ({percent_diff:.1f}%)")

if percent_diff > 5:
    print(f"  ✅ GOOD! Edges are more discriminative!")
else:
    print(f"  ❌ Still not enough separation")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Find examples
benign_idx = np.where(all_labels == 0)[0][0]
malignant_idx = np.where(all_labels == 1)[0][0]

# Raw images
img_dir = os.path.join(dataset_path, 'train', 'img')
img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

# Show raw
for i, img_file in enumerate(img_files[:2]):
    img_path = os.path.join(img_dir, img_file)
    ann_path = os.path.join(img_dir.replace('img', 'ann'), img_file + '.json')
    
    if not os.path.exists(ann_path):
        continue
    
    label = get_label_from_annotation(ann_path)
    if label is None:
        continue
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    
    resized = resize_image(img, 512)
    edges = cv2.Canny(resized, 50, 150)
    
    row = i
    
    # Raw
    axes[row, 0].imshow(resized, cmap='gray')
    axes[row, 0].set_title(f"{'Benign' if label == 0 else 'Malignant'} - Raw")
    axes[row, 0].axis('off')
    
    # Edges
    axes[row, 1].imshow(edges, cmap='gray')
    axes[row, 1].set_title(f"{'Benign' if label == 0 else 'Malignant'} - Edges")
    axes[row, 1].axis('off')
    
    # Histogram
    axes[row, 2].hist(edges.flatten(), bins=50, alpha=0.7)
    axes[row, 2].set_title(f"Edge histogram")
    axes[row, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('edge_detection_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved edge_detection_analysis.png")

print("\n" + "=" * 80)
if percent_diff > 5:
    print("RECOMMENDATION: Use edge detection preprocessing!")
    print("The edges show better class separation than raw pixels.")
else:
    print("CONCLUSION: Even edge detection doesn't help much.")
    print("The dataset may be fundamentally too difficult.")
print("=" * 80)
