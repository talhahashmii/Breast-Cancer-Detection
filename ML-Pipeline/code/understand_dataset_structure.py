"""
Understand the actual dataset structure
Why are there 3103 annotations for 237 images?
"""

import os
import json
import cv2
import numpy as np
from collections import defaultdict

dataset_path = r"..\data\Dataset"

print("=" * 80)
print("UNDERSTANDING DATASET STRUCTURE")
print("=" * 80)

# Map images to their annotations
image_to_labels = defaultdict(list)

for split in ['train', 'test']:
    ann_dir = os.path.join(dataset_path, split, 'ann')
    
    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith('.json'):
            continue
        
        # Extract image filename (remove .json)
        img_file = ann_file.replace('.json', '')
        
        ann_path = os.path.join(ann_dir, ann_file)
        
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # Get label
            label = None
            if 'tags' in data:
                for tag in data['tags']:
                    tag_name = tag.get('name', '').lower()
                    if tag_name in ['benign', 'benign_without_callback']:
                        label = 0
                        break
                    elif tag_name == 'malignant':
                        label = 1
                        break
            
            if label is not None:
                image_to_labels[img_file].append(label)
        
        except:
            pass

print(f"\nTotal unique images: {len(image_to_labels)}")
print(f"Total annotations: {sum(len(labels) for labels in image_to_labels.values())}")

# Check for images with multiple labels
multi_label_images = {img: labels for img, labels in image_to_labels.items() if len(labels) > 1}

print(f"\nImages with MULTIPLE annotations: {len(multi_label_images)}")

if multi_label_images:
    print("\nExamples of images with multiple annotations:")
    for img, labels in list(multi_label_images.items())[:5]:
        print(f"  {img}: {labels}")
        # Check if labels are consistent
        if len(set(labels)) > 1:
            print(f"    ⚠️  CONFLICTING LABELS! {set(labels)}")
        else:
            print(f"    ✓ All labels are the same: {labels[0]}")

# Check distribution
print("\n" + "=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)

benign_images = sum(1 for labels in image_to_labels.values() if 0 in labels)
malignant_images = sum(1 for labels in image_to_labels.values() if 1 in labels)
conflicting_images = sum(1 for labels in image_to_labels.values() if len(set(labels)) > 1)

print(f"\nImages labeled as BENIGN: {benign_images}")
print(f"Images labeled as MALIGNANT: {malignant_images}")
print(f"Images with CONFLICTING labels: {conflicting_images}")
print(f"Total unique images: {len(image_to_labels)}")

# The KEY QUESTION
print("\n" + "=" * 80)
print("KEY FINDING")
print("=" * 80)

if conflicting_images > 0:
    print(f"\n❌ PROBLEM: {conflicting_images} images have CONFLICTING labels!")
    print("   Same image labeled as both benign AND malignant")
    print("   This is why the model can't learn!")
else:
    print(f"\n✓ Good: All images have consistent labels")
    print(f"   But we have {len(multi_label_images)} images with multiple annotations")
    print(f"   (probably multiple ROIs marked in same image)")

# Show some examples
print("\n" + "=" * 80)
print("SAMPLE IMAGES")
print("=" * 80)

train_img_dir = os.path.join(dataset_path, 'train', 'img')
benign_sample = None
malignant_sample = None

for img_file, labels in image_to_labels.items():
    if benign_sample is None and 0 in labels:
        img_path = os.path.join(train_img_dir, img_file)
        if os.path.exists(img_path):
            benign_sample = (img_file, img_path)
    
    if malignant_sample is None and 1 in labels:
        img_path = os.path.join(train_img_dir, img_file)
        if os.path.exists(img_path):
            malignant_sample = (img_file, img_path)
    
    if benign_sample and malignant_sample:
        break

if benign_sample:
    print(f"\nBenign example: {benign_sample[0]}")
    img = cv2.imread(benign_sample[1], cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print(f"  Shape: {img.shape}, Mean: {img.mean():.1f}, Std: {img.std():.1f}")

if malignant_sample:
    print(f"\nMalignant example: {malignant_sample[0]}")
    img = cv2.imread(malignant_sample[1], cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print(f"  Shape: {img.shape}, Mean: {img.mean():.1f}, Std: {img.std():.1f}")

print("\n" + "=" * 80)
