"""
Ultra-minimal preprocessing - preserve raw image differences
Just: Load → Resize → Save (no cropping, no enhancement)
"""

import os
import cv2
import json
import numpy as np

dataset_path = r"..\data\Dataset"
output_dir = r"..\Data\Preprocessed Data\Raw"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("ULTRA-MINIMAL PREPROCESSING - PRESERVE RAW DIFFERENCES")
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
    """Resize to square with aspect ratio preserved"""
    height, width = image.shape
    scale = target_size / max(height, width)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create black canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Center the image
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

def preprocess_raw(img_path):
    """Ultra-minimal preprocessing"""
    # Load
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Just resize - NO cropping, NO enhancement
    resized = resize_image(img, target_size=512)
    
    return resized  # Return as uint8 [0-255]

# Process all images
all_images = []
all_labels = []
processed_count = 0
error_count = 0

for split in ['train', 'test']:
    print(f"\n{'='*80}")
    print(f"Processing {split.upper()} split")
    print(f"{'='*80}")
    
    img_dir = os.path.join(dataset_path, split, 'img')
    ann_dir = os.path.join(dataset_path, split, 'ann')
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    
    print(f"Found {len(img_files)} images")
    
    for i, img_file in enumerate(img_files):
        # Get paths
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, img_file + '.json')
        
        # Get label
        label = get_label_from_annotation(ann_path)
        
        if label is None:
            error_count += 1
            continue
        
        # Preprocess
        processed = preprocess_raw(img_path)
        
        if processed is not None:
            all_images.append(processed)
            all_labels.append(label)
            processed_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(img_files)} images...")
        else:
            error_count += 1
    
    print(f"✓ {split} complete: {processed_count} images processed")

# Convert to arrays
all_images = np.array(all_images, dtype=np.uint8)
all_labels = np.array(all_labels)

print(f"\n{'='*80}")
print("FINAL STATISTICS")
print(f"{'='*80}")
print(f"Total processed: {len(all_images)}")
print(f"Errors: {error_count}")
print(f"Benign: {sum(all_labels == 0)}")
print(f"Malignant: {sum(all_labels == 1)}")
print()
print(f"Image stats:")
print(f"  Shape: {all_images.shape}")
print(f"  Dtype: {all_images.dtype}")
print(f"  Range: [{all_images.min()}, {all_images.max()}]")
print(f"  Mean: {all_images.mean():.2f}")
print(f"  Std: {all_images.std():.2f}")
print()

# Check class separation
benign_mean = all_images[all_labels == 0].mean()
malignant_mean = all_images[all_labels == 1].mean()
difference = abs(benign_mean - malignant_mean)
percent_diff = (difference / benign_mean) * 100

print(f"Class separation:")
print(f"  Benign mean: {benign_mean:.2f}")
print(f"  Malignant mean: {malignant_mean:.2f}")
print(f"  Difference: {difference:.2f} ({percent_diff:.1f}%)")
print()

if percent_diff < 1:
    print("❌ Classes still too similar")
elif percent_diff < 2:
    print("⚠️  Classes somewhat similar - model may struggle")
else:
    print("✅ Good class separation!")

# Save
np.save(os.path.join(output_dir, 'raw_images.npy'), all_images)
np.save(os.path.join(output_dir, 'raw_labels.npy'), all_labels)

print(f"\n{'='*80}")
print(f"✓ Saved to {output_dir}/")
print(f"  - raw_images.npy: {all_images.shape}")
print(f"  - raw_labels.npy: {all_labels.shape}")
print(f"{'='*80}\n")
