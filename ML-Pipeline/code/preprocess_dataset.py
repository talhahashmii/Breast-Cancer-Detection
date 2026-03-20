"""
Phase 4: Preprocessing and Dual-View Pairing
Breast Cancer Detection - Dual-View CNN with ResNet50

This script:
- Resizes all images to 512x512
- Normalizes pixel values to 0-1
- Pairs CC and MLO views per patient
- Creates train/validation/test splits
- Saves preprocessed data for model training
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import json
from pathlib import Path
from collections import defaultdict
import traceback

# ==================== CONFIGURATION ====================
DATASET_BASE_PATH = r"c:\Users\Digi Opia\Desktop\BREAST CANCER DETECTION\ML-Pipeline\Data\Dataset"
CSV_PATH = os.path.join(DATASET_BASE_PATH, "csv")
ROI_PATH = r"c:\Users\Digi Opia\Desktop\BREAST CANCER DETECTION\ML-Pipeline\Data\Roi Extreacted Data"
PREPROCESSED_OUTPUT = r"c:\Users\Digi Opia\Desktop\BREAST CANCER DETECTION\ML-Pipeline\Data\Preprocessed Data"

# Preprocessing parameters
TARGET_SIZE = (512, 512)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Create output directory
os.makedirs(PREPROCESSED_OUTPUT, exist_ok=True)

# ==================== UTILITY FUNCTIONS ====================

def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)

def print_subheader(title):
    """Print formatted subsection header"""
    print(f"\n{title}")
    print("-" * 90)

def load_roi_metadata():
    """Load ROI extraction results and rebuild file paths based on actual directory contents"""
    print_subheader("LOADING ROI METADATA")
    
    results_path = os.path.join(ROI_PATH, "extraction_results.csv")
    results_df = pd.read_csv(results_path)
    
    print(f"  [OK] Loaded {len(results_df)} ROI extraction records")
    
    # Rebuild file paths based on actual directory contents
    # The extracted files have format: {DICOM_ID}_{original_filename}_roi.jpg
    train_dir = os.path.join(ROI_PATH, "train")
    test_dir = os.path.join(ROI_PATH, "test")
    
    # Build a mapping from original filename to actual file path
    actual_file_map = {}
    
    for file in os.listdir(train_dir):
        if file.endswith('.jpg'):
            # Extract the original filename from format: {DICOM_ID}_{original}_roi.jpg
            # e.g., 1.3.6.1.4.1.9590.100.1.2.1234_1-263_roi.jpg -> 1-263.jpg
            if '_roi.jpg' in file:
                parts = file.split('_')
                # Last part before _roi is the original filename pattern
                if len(parts) >= 2:
                    # Reconstruct original name from parts
                    for i, part in enumerate(parts):
                        if part.startswith('1-') or part.startswith('2-'):
                            original_name = part + '.jpg'
                            actual_file_map[original_name] = file
                            break
    
    for file in os.listdir(test_dir):
        if file.endswith('.jpg'):
            if '_roi.jpg' in file:
                parts = file.split('_')
                if len(parts) >= 2:
                    for i, part in enumerate(parts):
                        if part.startswith('1-') or part.startswith('2-'):
                            original_name = part + '.jpg'
                            actual_file_map[original_name] = file
                            break
    
    print(f"  [OK] Found {len(actual_file_map)} actual ROI files")
    
    # Update results_df with actual paths
    results_df['actual_filename'] = results_df['image_file'].map(actual_file_map)
    results_df['file_found'] = results_df['actual_filename'].notna()
    
    return results_df

def load_original_labels():
    """Load original labels from CSV files and create DICOM ID mapping"""
    print_subheader("LOADING ORIGINAL LABELS")
    
    import re
    
    calc_train = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_train_set.csv"))
    calc_test = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_test_set.csv"))
    mass_train = pd.read_csv(os.path.join(CSV_PATH, "mass_case_description_train_set.csv"))
    mass_test = pd.read_csv(os.path.join(CSV_PATH, "mass_case_description_test_set.csv"))
    
    # Add metadata
    calc_train['split_original'] = 'train'
    calc_test['split_original'] = 'test'
    mass_train['split_original'] = 'train'
    mass_test['split_original'] = 'test'
    
    all_labels = pd.concat([calc_train, calc_test, mass_train, mass_test], ignore_index=True)
    
    print(f"  [OK] Loaded {len(all_labels)} label records")
    
    # Create a mapping from patient ID to label
    patient_id_to_label = {}
    for _, row in all_labels.iterrows():
        patient_id = row['patient_id']
        pathology = row['pathology']
        label = 1 if pathology == 'MALIGNANT' else 0
        # Store by patient ID (only if not already mapped)
        if patient_id not in patient_id_to_label:
            patient_id_to_label[patient_id] = label
    
    # Now load DICOM info to create DICOM ID to label mapping
    dicom_to_label = {}
    dicom_info_path = os.path.join(CSV_PATH, "dicom_info.csv")
    if os.path.exists(dicom_info_path):
        dicom_info = pd.read_csv(dicom_info_path)
        
        # Extract DICOM ID from image_path
        def extract_dicom_id(path):
            if pd.isna(path):
                return None
            parts = str(path).split('/')
            if len(parts) >= 3:
                return parts[2]
            return None
        
        # Extract patient ID from PatientID field
        def extract_patient_id(patient_str):
            if pd.isna(patient_str):
                return None
            match = re.search(r'(P_\d+)', str(patient_str))
            if match:
                return match.group(1)
            return None
        
        dicom_info['dicom_id'] = dicom_info['image_path'].apply(extract_dicom_id)
        dicom_info['patient_id'] = dicom_info['PatientID'].apply(extract_patient_id)
        
        # Create DICOM ID to label mapping
        for _, row in dicom_info.iterrows():
            dicom_id = row['dicom_id']
            patient_id = row['patient_id']
            if dicom_id and patient_id and patient_id in patient_id_to_label:
                dicom_to_label[dicom_id] = patient_id_to_label[patient_id]
        
        print(f"  [OK] Created DICOM-to-label mapping for {len(dicom_to_label)} DICOM IDs")
    else:
        print(f"  [WARN] dicom_info.csv not found - using patient_id mapping only")
    
    return all_labels, dicom_to_label

def create_label_index(labels_df):
    """Create index to quickly find labels"""
    print_subheader("CREATING LABEL INDEX")
    
    # Create a multi-level index by patient_id and view
    label_index = {}
    for _, row in labels_df.iterrows():
        patient_id = row['patient_id']
        view = row['image view']
        left_right = row['left or right breast']
        
        # Create key
        key = f"{patient_id}_{left_right}_{view}"
        
        # Convert pathology to binary label
        label = 1 if row['pathology'] == 'MALIGNANT' else 0
        
        label_index[key] = {
            'label': label,
            'pathology': row['pathology'],
            'view': view,
            'breast': left_right,
            'split_original': row['split_original']
        }
    
    print(f"  [OK] Created index for {len(label_index)} records")
    return label_index

def parse_roi_filename(filename):
    """Parse ROI filename to extract patient info"""
    # Format: {DICOM_ID}_{image_name}_roi.jpg
    # Extract DICOM ID (first ~20 chars) and image name
    parts = filename.replace('_roi.jpg', '').split('_')
    
    if len(parts) >= 2:
        dicom_id_part = parts[0]
        image_name = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
        return (dicom_id_part, image_name)
    
    return (None, None)

def extract_view_from_filename(filename):
    """Extract view type from filename"""
    # Pattern: 1-XXX.jpg = CC, 2-XXX.jpg = MLO
    if filename.startswith('1-'):
        return 'CC'
    elif filename.startswith('2-'):
        return 'MLO'
    
    return None

def load_and_preprocess_image(image_path, target_size=TARGET_SIZE):
    """Load image, resize, and normalize"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Resize using PIL
        img_pil = Image.fromarray(img_array.astype(np.uint8), mode='L')
        img_resized_pil = img_pil.resize(target_size, Image.Resampling.BILINEAR)
        img_resized = np.array(img_resized_pil)
        
        # Normalize to 0-1
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized, True
    
    except Exception as e:
        return None, False

def group_images_by_patient(roi_results_df, roi_path, labels_df=None, dicom_to_label=None):
    """Group images by patient and view, and merge with labels"""
    print_header("GROUPING IMAGES BY PATIENT AND VIEW")
    
    if dicom_to_label is None:
        dicom_to_label = {}
    
    patient_views = defaultdict(lambda: {'CC': [], 'MLO': []})
    failed_images = []
    successful_count = 0
    labeled_count = 0
    
    for _, row in roi_results_df.iterrows():
        if row['status'] != 'success':
            failed_images.append(row)
            continue
        
        # Check if actual file was found
        if not row.get('file_found', False):
            failed_images.append(row)
            continue
        
        image_file = row['image_file']
        actual_filename = row.get('actual_filename', image_file)
        dicom_id = row.get('dicom_id', '')
        split = row.get('split', 'train')
        
        # Extract view from filename (1- = CC, 2- = MLO)
        view = extract_view_from_filename(image_file)
        
        if view is None:
            failed_images.append(row)
            continue
        
        # Get label from DICOM-to-label mapping, default to -1 (unknown)
        label = dicom_to_label.get(dicom_id, -1)
        if label != -1:
            labeled_count += 1
        
        # Use DICOM ID as patient key
        patient_key = dicom_id
        
        # Build full path to ROI image using ACTUAL filename
        split_folder = 'train' if split == 'train' else 'test'
        roi_full_path = os.path.join(roi_path, split_folder, actual_filename)
        
        patient_views[patient_key][view].append({
            'image_file': image_file,
            'actual_filename': actual_filename,
            'roi_path': roi_full_path,
            'dicom_id': dicom_id,
            'label': label,
            'split': split
        })
        
        successful_count += 1
    
    # Check label distribution
    total_images = sum(1 for p in patient_views.values() for v in p.values() for _ in v)
    
    print(f"  [OK] Grouped images from {len(patient_views)} patients")
    print(f"  [OK] Total images: {total_images}")
    print(f"  [OK] Successfully grouped: {successful_count}")
    print(f"  [OK] Labeled images: {labeled_count}")
    if len(failed_images) > 0:
        print(f"  [WARN] Failed images: {len(failed_images)}")
    
    return patient_views, failed_images

def create_dual_view_pairs(patient_views, label_index):
    """Create dual-view pairs from CC and MLO"""
    print_header("CREATING DUAL-VIEW PAIRS")
    
    pairs = []
    unpaired_patients = 0
    
    for patient_key, views in patient_views.items():
        cc_images = views['CC']
        mlo_images = views['MLO']
        
        # Try to pair CC and MLO views
        if cc_images and mlo_images:
            # Pair first CC with first MLO
            cc_img = cc_images[0]
            mlo_img = mlo_images[0]
            
            # Get label (they should have the same label for the same patient)
            label = cc_img.get('label', -1)
            
            pairs.append({
                'patient_key': patient_key,
                'cc_image': cc_img,
                'mlo_image': mlo_img,
                'label': label,
                'pair_type': 'both'
            })
        elif cc_images:
            unpaired_patients += 1
        elif mlo_images:
            unpaired_patients += 1
    
    print(f"  [OK] Created {len(pairs)} dual-view pairs")
    print(f"  [INFO] {unpaired_patients} patients have only one view")
    
    return pairs

def load_dual_view_pair(pair, roi_path):
    """Load and preprocess a dual-view pair"""
    try:
        # Load CC view
        cc_path = pair['cc_image']['roi_path']
        cc_img, cc_success = load_and_preprocess_image(cc_path)
        
        if not cc_success:
            return None, False
        
        # Load MLO view
        mlo_path = pair['mlo_image']['roi_path']
        mlo_img, mlo_success = load_and_preprocess_image(mlo_path)
        
        if not mlo_success:
            return None, False
        
        # Stack into 2-channel input (CC, MLO)
        dual_view = np.stack([cc_img, mlo_img], axis=0)
        
        return dual_view, True
    
    except Exception as e:
        return None, False

def save_preprocessed_data(train_pairs, val_pairs, test_pairs, output_path):
    """Save preprocessed data"""
    print_header("SAVING PREPROCESSED DATA")
    
    # Create subdirectories
    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'val')
    test_dir = os.path.join(output_path, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Save data info
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []
    
    # Process training data
    print("\n  Processing training data...")
    for idx, pair in enumerate(train_pairs):
        dual_view, success = load_dual_view_pair(pair, ROI_PATH)
        if success:
            # Save as numpy file
            img_path = os.path.join(train_dir, f"train_{idx:06d}.npy")
            np.save(img_path, dual_view)
            train_images.append(f"train_{idx:06d}.npy")
            train_labels.append(pair['label'])
        
        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx+1}/{len(train_pairs)} training samples")
    
    # Process validation data
    print("  Processing validation data...")
    for idx, pair in enumerate(val_pairs):
        dual_view, success = load_dual_view_pair(pair, ROI_PATH)
        if success:
            img_path = os.path.join(val_dir, f"val_{idx:06d}.npy")
            np.save(img_path, dual_view)
            val_images.append(f"val_{idx:06d}.npy")
            val_labels.append(pair['label'])
        
        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx+1}/{len(val_pairs)} validation samples")
    
    # Process test data
    print("  Processing test data...")
    for idx, pair in enumerate(test_pairs):
        dual_view, success = load_dual_view_pair(pair, ROI_PATH)
        if success:
            img_path = os.path.join(test_dir, f"test_{idx:06d}.npy")
            np.save(img_path, dual_view)
            test_images.append(f"test_{idx:06d}.npy")
            test_labels.append(pair['label'])
        
        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx+1}/{len(test_pairs)} test samples")
    
    # Save metadata
    metadata = {
        'train': {
            'images': train_images,
            'labels': train_labels,
            'size': len(train_images)
        },
        'val': {
            'images': val_images,
            'labels': val_labels,
            'size': len(val_images)
        },
        'test': {
            'images': test_images,
            'labels': test_labels,
            'size': len(test_images)
        },
        'image_size': TARGET_SIZE,
        'num_channels': 2,
        'data_format': 'numpy (.npy)',
        'label_format': 'binary (0=benign, 1=malignant)'
    }
    
    # Save metadata JSON
    metadata_path = os.path.join(output_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        # Convert numpy lists to regular lists for JSON serialization
        metadata_json = {
            'train': {
                'images': train_images,
                'labels': [int(x) for x in train_labels],
                'size': len(train_images)
            },
            'val': {
                'images': val_images,
                'labels': [int(x) for x in val_labels],
                'size': len(val_images)
            },
            'test': {
                'images': test_images,
                'labels': [int(x) for x in test_labels],
                'size': len(test_images)
            },
            'image_size': list(TARGET_SIZE),
            'num_channels': 2,
            'data_format': 'numpy (.npy)',
            'label_format': 'binary (0=benign, 1=malignant)'
        }
        json.dump(metadata_json, f, indent=2)
    
    print(f"  [OK] Saved training data: {len(train_images)} samples")
    print(f"  [OK] Saved validation data: {len(val_images)} samples")
    print(f"  [OK] Saved test data: {len(test_images)} samples")
    print(f"  [OK] Saved metadata to: {metadata_path}")
    
    return metadata

def generate_preprocessing_report(train_pairs, val_pairs, test_pairs, output_path):
    """Generate preprocessing report"""
    print_header("PREPROCESSING REPORT")
    
    # Calculate statistics
    all_pairs = train_pairs + val_pairs + test_pairs
    
    if len(all_pairs) == 0:
        print("  [WARN] No pairs to report!")
        return
    
    all_labels = [p['label'] for p in all_pairs]
    malignant = sum(1 for l in all_labels if l == 1)
    benign = sum(1 for l in all_labels if l == 0)
    unknown = sum(1 for l in all_labels if l == -1)
    
    train_labels = [p['label'] for p in train_pairs] if train_pairs else []
    train_mal = sum(1 for l in train_labels if l == 1)
    train_ben = sum(1 for l in train_labels if l == 0)
    
    val_labels = [p['label'] for p in val_pairs] if val_pairs else []
    val_mal = sum(1 for l in val_labels if l == 1)
    val_ben = sum(1 for l in val_labels if l == 0)
    
    test_labels = [p['label'] for p in test_pairs] if test_pairs else []
    test_mal = sum(1 for l in test_labels if l == 1)
    test_ben = sum(1 for l in test_labels if l == 0)
    
    # Print report
    report = f"""
PREPROCESSING SUMMARY
{'='*80}

DATA STATISTICS:
  Total dual-view pairs: {len(all_pairs)}
  Training pairs: {len(train_pairs)}
  Validation pairs: {len(val_pairs)}
  Test pairs: {len(test_pairs)}

CLASS DISTRIBUTION:
  Total Benign: {benign} ({benign/len(all_pairs)*100:.1f}% if len(all_pairs) > 0 else 0)
  Total Malignant: {malignant} ({malignant/len(all_pairs)*100:.1f}% if len(all_pairs) > 0 else 0)
  Unknown: {unknown}
  
  Training:
    Benign: {train_ben} ({train_ben/len(train_labels)*100:.1f}% if len(train_labels) > 0 else 0)
    Malignant: {train_mal} ({train_mal/len(train_labels)*100:.1f}% if len(train_labels) > 0 else 0)
  
  Validation:
    Benign: {val_ben} ({val_ben/len(val_labels)*100:.1f}% if len(val_labels) > 0 else 0)
    Malignant: {val_mal} ({val_mal/len(val_labels)*100:.1f}% if len(val_labels) > 0 else 0)
  
  Test:
    Benign: {test_ben} ({test_ben/len(test_labels)*100:.1f}% if len(test_labels) > 0 else 0)
    Malignant: {test_mal} ({test_mal/len(test_labels)*100:.1f}% if len(test_labels) > 0 else 0)

IMAGE SPECIFICATIONS:
  Input size: Variable (ROI extracted)
  Output size: {TARGET_SIZE[0]} x {TARGET_SIZE[1]} pixels
  Format: Grayscale (single channel per view)
  Channels: 2 (CC view + MLO view)
  Data type: float32
  Normalization: 0-1 (pixel values / 255)

SPLIT RATIOS:
  Training: {TRAIN_RATIO*100:.0f}%
  Validation: {VAL_RATIO*100:.0f}%
  Test: {TEST_RATIO*100:.0f}%

OUTPUT LOCATION:
  {output_path}

FILES CREATED:
  - train/: Training data (.npy files)
  - val/: Validation data (.npy files)
  - test/: Test data (.npy files)
  - metadata.json: Data information and statistics
  - preprocessing_report.txt: This file
"""
    
    # Save report
    report_path = os.path.join(output_path, 'preprocessing_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n  [OK] Saved report to: {report_path}")

def main():
    """Main execution"""
    print("\n")
    print("[" + "=" * 88 + "]")
    print("|" + " " * 20 + "BREAST CANCER DETECTION SYSTEM" + " " * 38 + "|")
    print("|" + " " * 25 + "Phase 4: Preprocessing & Pairing" + " " * 33 + "|")
    print("[" + "=" * 88 + "]")
    
    print_header("CONFIGURATION")
    print(f"  Target Image Size: {TARGET_SIZE}")
    print(f"  Input: ROI Extracted Images from {ROI_PATH}")
    print(f"  Output: {PREPROCESSED_OUTPUT}")
    print(f"  Train/Val/Test Ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    
    try:
        # Load data
        roi_results = load_roi_metadata()
        labels_df, dicom_to_label = load_original_labels()
        label_index = create_label_index(labels_df)
        
        # Group images - NOW PASSING DICOM_TO_LABEL MAPPING
        patient_views, failed = group_images_by_patient(roi_results, ROI_PATH, labels_df, dicom_to_label)
        
        # Create pairs
        pairs = create_dual_view_pairs(patient_views, label_index)
        
        # Split data
        print_header("SPLITTING DATA")
        np.random.seed(RANDOM_SEED)
        indices = np.arange(len(pairs))
        np.random.shuffle(indices)
        
        train_size = int(len(pairs) * TRAIN_RATIO)
        val_size = int(len(pairs) * VAL_RATIO)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]
        test_pairs = [pairs[i] for i in test_idx]
        
        print(f"  [OK] Split into {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
        
        # Save preprocessed data
        metadata = save_preprocessed_data(train_pairs, val_pairs, test_pairs, PREPROCESSED_OUTPUT)
        
        # Generate report
        generate_preprocessing_report(train_pairs, val_pairs, test_pairs, PREPROCESSED_OUTPUT)
        
        return True
    
    except Exception as e:
        print_header("FATAL ERROR")
        print(f"  {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    print_header("PREPROCESSING COMPLETE")
    if success:
        print("  Status: [PASSED]")
        print(f"  Preprocessed data saved to: {PREPROCESSED_OUTPUT}")
        print("  Dataset is ready for Phase 5: Model Training")
    else:
        print("  Status: [FAILED]")
    
    print("\n")
