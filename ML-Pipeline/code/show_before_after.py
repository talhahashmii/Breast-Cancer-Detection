import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os
import random

# Set up paths
dataset_path = r"..\data\Dataset"
train_img_dir = os.path.join(dataset_path, 'train', 'img')
train_ann_dir = os.path.join(dataset_path, 'train', 'ann')

# Find all images
image_files = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
print(f"Found {len(image_files)} images")

# Helper functions
def get_label_from_annotation(ann_path):
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        if 'tags' in data:
            for tag in data['tags']:
                tag_name = tag.get('name', '').lower()
                if tag_name == 'benign':
                    return 'Benign'
                elif tag_name == 'malignant':
                    return 'Malignant'
                elif tag_name == 'benign_without_callback':
                    return 'Benign'
        return 'Unknown'
    except:
        return 'Unknown'

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def resize_image(image, target_size=512):
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

def normalize_image(image):
    image_float = image.astype(np.float32)
    normalized = image_float / 255.0
    return normalized

def preprocess_image(image_path):
    image = load_image(image_path)
    if image is None:
        return None
    resized = resize_image(image, 512)
    normalized = normalize_image(resized)
    return normalized

# Pick 2 random images
print("\n" + "="*80)
print("PICKING 2 RANDOM IMAGES...")
print("="*80)

for img_num in range(1, 3):
    random_file = random.choice(image_files)
    image_path = os.path.join(train_img_dir, random_file)
    ann_path = os.path.join(train_ann_dir, random_file + '.json')
    
    label_text = get_label_from_annotation(ann_path)
    
    print(f"\nImage {img_num}: {random_file}")
    print(f"Label: {label_text}")
    
    # Load raw image
    raw_image = load_image(image_path)
    print(f"Raw image shape: {raw_image.shape}")
    print(f"Raw image pixel range: [{raw_image.min()}, {raw_image.max()}]")
    
    # Preprocess image
    preprocessed_image = preprocess_image(image_path)
    print(f"Preprocessed image shape: {preprocessed_image.shape}")
    print(f"Preprocessed image pixel range: [{preprocessed_image.min():.4f}, {preprocessed_image.max():.4f}]")
    
    # Display side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(raw_image, cmap='gray')
    axes[0].set_title(f'BEFORE Preprocessing\nShape: {raw_image.shape}\nPixel Range: [{raw_image.min()}, {raw_image.max()}]\nLabel: {label_text}', 
                      fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(preprocessed_image, cmap='gray')
    axes[1].set_title(f'AFTER Preprocessing\nShape: {preprocessed_image.shape}\nPixel Range: [{preprocessed_image.min():.4f}, {preprocessed_image.max():.4f}]\nLabel: {label_text}', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'before_after_image_{img_num}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved: before_after_image_{img_num}.png")

print("\n" + "="*80)
print("DONE! Check the PNG files in the code directory")
print("="*80)
