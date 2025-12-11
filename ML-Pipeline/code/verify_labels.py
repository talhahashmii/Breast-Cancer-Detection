"""
Verify if labels are correct by checking annotation files
"""

import os
import json
import cv2
import numpy as np

dataset_path = r"..\data\Dataset"

print("=" * 80)
print("VERIFYING LABELS - CHECKING ANNOTATION FILES")
print("=" * 80)

# Count labels from annotations
benign_count = 0
malignant_count = 0
label_issues = []

for split in ['train', 'test']:
    ann_dir = os.path.join(dataset_path, split, 'ann')
    
    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith('.json'):
            continue
        
        ann_path = os.path.join(ann_dir, ann_file)
        
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # Check for label
            label = None
            if 'tags' in data:
                for tag in data['tags']:
                    tag_name = tag.get('name', '').lower()
                    if tag_name in ['benign', 'benign_without_callback']:
                        label = 'benign'
                        benign_count += 1
                        break
                    elif tag_name == 'malignant':
                        label = 'malignant'
                        malignant_count += 1
                        break
            
            if label is None:
                label_issues.append(ann_file)
        
        except Exception as e:
            label_issues.append(f"{ann_file}: {str(e)}")

print(f"\nLabel Distribution from Annotations:")
print(f"  Benign: {benign_count}")
print(f"  Malignant: {malignant_count}")
print(f"  Total: {benign_count + malignant_count}")
print(f"  Ratio: {benign_count/(benign_count+malignant_count):.1%} benign / {malignant_count/(benign_count+malignant_count):.1%} malignant")

if label_issues:
    print(f"\n⚠️  Issues found: {len(label_issues)}")
    for issue in label_issues[:5]:
        print(f"  - {issue}")

# Now check if there's a pattern in image names
print("\n" + "=" * 80)
print("CHECKING IMAGE NAME PATTERNS")
print("=" * 80)

train_img_dir = os.path.join(dataset_path, 'train', 'img')
img_files = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]

print(f"\nSample image names:")
for img_file in img_files[:10]:
    print(f"  {img_file}")

# Check if names contain clues
print("\n" + "=" * 80)
print("CHECKING FOR PATTERNS IN FILENAMES")
print("=" * 80)

# Look for keywords in filenames
calc_count = 0
mass_count = 0

for img_file in img_files:
    if 'calc' in img_file.lower():
        calc_count += 1
    if 'mass' in img_file.lower():
        mass_count += 1

print(f"\nFilename patterns:")
print(f"  'Calc' in filename: {calc_count}")
print(f"  'Mass' in filename: {mass_count}")
print(f"  Other: {len(img_files) - calc_count - mass_count}")

print("\n" + "=" * 80)
print("IMPORTANT QUESTIONS:")
print("=" * 80)
print("""
1. Are the labels definitely correct in the JSON files?
2. Could there be a data labeling error?
3. Are benign and malignant images actually visually different?
4. Should we try a different approach (e.g., use only Calcification vs Mass)?

Recommendation:
- Manually check 5-10 benign and 5-10 malignant images
- Verify they look different to human eye
- If they don't look different, the dataset may be too difficult
- Consider using only one type (e.g., only Calcifications)
""")
