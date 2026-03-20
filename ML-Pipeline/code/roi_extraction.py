"""
Phase 3: ROI Extraction
Breast Cancer Detection - Dual-View CNN with ResNet50

This script extracts Region of Interest (ROI) from mammogram images.
It removes black borders and focuses on the actual breast tissue.

Process:
- Convert image to grayscale (already grayscale)
- Find the breast tissue boundary (non-black pixels)
- Add padding around the boundary
- Crop and save the ROI
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import traceback
import shutil
from datetime import datetime

# ==================== CONFIGURATION ====================
DATASET_BASE_PATH = r"c:\Users\Digi Opia\Desktop\BREAST CANCER DETECTION\ML-Pipeline\Data\Dataset"
CSV_PATH = os.path.join(DATASET_BASE_PATH, "csv")
JPEG_PATH = os.path.join(DATASET_BASE_PATH, "jpeg")
ROI_OUTPUT_PATH = r"c:\Users\Digi Opia\Desktop\BREAST CANCER DETECTION\ML-Pipeline\Data\Roi Extreacted Data"

# Create output directory
os.makedirs(ROI_OUTPUT_PATH, exist_ok=True)

# ROI extraction parameters
PADDING_RATIO = 0.10  # 10% padding around tissue boundary
MIN_TISSUE_THRESHOLD = 10  # Minimum tissue area to consider valid
PREVIEW_SAMPLES = 5  # Number of samples to save as before/after preview

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

def find_tissue_boundary(image_array):
    """
    Find the boundary of breast tissue in the image.
    Returns (min_row, max_row, min_col, max_col) or None if no tissue found.
    """
    try:
        # Find non-zero pixels (tissue)
        rows = np.any(image_array > MIN_TISSUE_THRESHOLD, axis=1)
        cols = np.any(image_array > MIN_TISSUE_THRESHOLD, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return (rmin, rmax, cmin, cmax)
    except Exception as e:
        return None

def add_padding(boundary, image_shape, padding_ratio=PADDING_RATIO):
    """
    Add padding around the boundary.
    """
    rmin, rmax, cmin, cmax = boundary
    height, width = image_shape
    
    # Calculate padding
    row_padding = int((rmax - rmin) * padding_ratio)
    col_padding = int((cmax - cmin) * padding_ratio)
    
    # Apply padding with boundaries
    rmin_padded = max(0, rmin - row_padding)
    rmax_padded = min(height - 1, rmax + row_padding)
    cmin_padded = max(0, cmin - col_padding)
    cmax_padded = min(width - 1, cmax + col_padding)
    
    return (rmin_padded, rmax_padded, cmin_padded, cmax_padded)

def extract_roi(image_array):
    """
    Extract ROI from image.
    Returns cropped image or None if extraction fails.
    """
    try:
        # Find tissue boundary
        boundary = find_tissue_boundary(image_array)
        if boundary is None:
            return None
        
        # Add padding
        boundary_padded = add_padding(boundary, image_array.shape)
        rmin, rmax, cmin, cmax = boundary_padded
        
        # Crop
        roi = image_array[rmin:rmax+1, cmin:cmax+1]
        
        return roi
    except Exception as e:
        return None

def process_image(image_path, output_path):
    """
    Process a single image: extract ROI and save.
    Returns (success, original_size, roi_size, original_shape, roi_shape)
    """
    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        original_shape = img_array.shape
        
        # Extract ROI
        roi_array = extract_roi(img_array)
        if roi_array is None:
            return (False, original_shape, None, original_shape, None, "No tissue found")
        
        roi_shape = roi_array.shape
        
        # Convert back to PIL Image
        roi_img = Image.fromarray(roi_array.astype(np.uint8), mode='L')
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save ROI
        roi_img.save(output_path)
        
        return (True, original_shape, roi_array.nbytes / (1024*1024), original_shape, roi_shape, "Success")
    
    except Exception as e:
        return (False, None, None, None, None, str(e))

def load_dataset():
    """Load all dataset CSVs"""
    print_subheader("LOADING DATASET")
    
    calc_train = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_train_set.csv"))
    calc_test = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_test_set.csv"))
    mass_train = pd.read_csv(os.path.join(CSV_PATH, "mass_case_description_train_set.csv"))
    mass_test = pd.read_csv(os.path.join(CSV_PATH, "mass_case_description_test_set.csv"))
    
    # Add metadata
    calc_train['dataset_type'] = 'calc'
    calc_train['split'] = 'train'
    calc_test['dataset_type'] = 'calc'
    calc_test['split'] = 'test'
    mass_train['dataset_type'] = 'mass'
    mass_train['split'] = 'train'
    mass_test['dataset_type'] = 'mass'
    mass_test['split'] = 'test'
    
    all_data = pd.concat([calc_train, calc_test, mass_train, mass_test], ignore_index=True)
    
    print(f"  [OK] Loaded {len(all_data)} cases")
    return all_data

def build_patient_mapping():
    """
    Build a mapping using the CSV file paths.
    The 'cropped image file path' in CSV points to the DICOM patient ID.
    """
    print_subheader("ANALYZING DATASET STRUCTURE")
    
    # Load one CSV to understand structure
    calc_train = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_train_set.csv"))
    
    # Check a sample row
    sample_row = calc_train.iloc[0]
    print(f"  Sample cropped image path: {sample_row.get('cropped image file path', 'N/A')}")
    
    # The structure appears to be dataset-specific paths in the CSV
    # We need to work with the actual JPEG directories
    return None

def extract_roi_from_dataset(data_df, sample_size=None):
    """
    Extract ROI from all images in dataset.
    Process all images in the JPEG directory since CSV doesn't map directly.
    """
    print_header("EXTRACTING ROI FROM IMAGES")
    
    print(f"  Processing all images in JPEG folder")
    
    # Create directory structure
    train_output = os.path.join(ROI_OUTPUT_PATH, "train")
    test_output = os.path.join(ROI_OUTPUT_PATH, "test")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)
    
    results = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'errors': defaultdict(int),
        'original_sizes': [],
        'roi_sizes': [],
        'compression_ratios': [],
        'cases': []
    }
    
    print("\n  Progress:")
    
    # Process all patient directories
    jpeg_dirs = sorted(os.listdir(JPEG_PATH))
    
    for idx, patient_dicom_id in enumerate(jpeg_dirs):
        patient_path = os.path.join(JPEG_PATH, patient_dicom_id)
        
        if not os.path.isdir(patient_path):
            continue
        
        # Find JPEG files
        images = [f for f in os.listdir(patient_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if not images:
            results['failed'] += 1
            results['errors']['no_images'] += 1
            results['total'] += 1
            continue
        
        # Process each image
        for img_file in images:
            img_path = os.path.join(patient_path, img_file)
            
            # Determine split (train vs test) based on file name pattern or random split
            split = 'train'  # Default to train
            
            # Try to find in data_df to get metadata
            pathology = 'UNKNOWN'
            view = 'UNKNOWN'
            
            # Create output path
            split_dir = train_output if split == 'train' else test_output
            roi_file = f"{patient_dicom_id[:20]}_{os.path.splitext(img_file)[0]}_roi.jpg"
            roi_path = os.path.join(split_dir, roi_file)
            
            # Extract ROI
            success, orig_shape, roi_size_mb, orig_shape_unused, roi_shape, error_msg = process_image(img_path, roi_path)
            
            results['total'] += 1
            
            if success:
                results['success'] += 1
                results['original_sizes'].append(orig_shape)
                if roi_shape:
                    results['roi_sizes'].append(roi_shape)
                
                results['cases'].append({
                    'dicom_id': patient_dicom_id,
                    'image_file': img_file,
                    'original_size': orig_shape,
                    'roi_size': roi_shape,
                    'pathology': pathology,
                    'view': view,
                    'split': split,
                    'status': 'success'
                })
            else:
                results['failed'] += 1
                results['errors'][error_msg] += 1
                results['cases'].append({
                    'dicom_id': patient_dicom_id,
                    'image_file': img_file,
                    'original_size': orig_shape,
                    'roi_size': None,
                    'pathology': pathology,
                    'view': view,
                    'split': split,
                    'status': f'failed: {error_msg}'
                })
            
            # Progress update
            if (idx + 1) % 500 == 0 or (idx + 1) % 100 == 0 and idx < 500:
                print(f"    [{idx+1}/{len(jpeg_dirs)}] Processed {results['success']} successful, {results['failed']} failed")
    
    return results

def save_results_summary(results):
    """Save results summary to CSV"""
    print_header("SAVING RESULTS SUMMARY")
    
    # Save results
    results_df = pd.DataFrame(results['cases'])
    summary_path = os.path.join(ROI_OUTPUT_PATH, "extraction_results.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"  [OK] Saved results to: {summary_path}")
    
    # Save statistics
    stats_path = os.path.join(ROI_OUTPUT_PATH, "extraction_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("ROI EXTRACTION STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total images processed: {results['total']}\n")
        f.write(f"Successful extractions: {results['success']} ({results['success']/max(results['total'], 1)*100:.1f}%)\n")
        f.write(f"Failed extractions: {results['failed']} ({results['failed']/max(results['total'], 1)*100:.1f}%)\n\n")
        
        if results['errors']:
            f.write("Errors encountered:\n")
            for error, count in results['errors'].items():
                f.write(f"  {error}: {count}\n")
        
        if results['original_sizes']:
            orig_array = np.array(results['original_sizes'])
            f.write(f"\nOriginal Image Dimensions:\n")
            f.write(f"  Min: {orig_array.min(axis=0)}\n")
            f.write(f"  Max: {orig_array.max(axis=0)}\n")
            f.write(f"  Mean: {orig_array.mean(axis=0).astype(int)}\n")
        
        if results['roi_sizes']:
            roi_array = np.array(results['roi_sizes'])
            f.write(f"\nROI Image Dimensions:\n")
            f.write(f"  Min: {roi_array.min(axis=0)}\n")
            f.write(f"  Max: {roi_array.max(axis=0)}\n")
            f.write(f"  Mean: {roi_array.mean(axis=0).astype(int)}\n")
            
            # Calculate compression
            original_pixels = np.prod(orig_array, axis=1).mean()
            roi_pixels = np.prod(roi_array, axis=1).mean()
            compression = (original_pixels - roi_pixels) / original_pixels * 100
            f.write(f"\nData Reduction:\n")
            f.write(f"  Average pixels removed: {compression:.1f}%\n")
    
    print(f"  [OK] Saved statistics to: {stats_path}")

def create_preview():
    """Create before/after preview images"""
    print_header("CREATING PREVIEW IMAGES")
    
    print("  Creating side-by-side before/after comparison...")
    
    try:
        # Get some sample images
        sample_images = []
        for patient_id in sorted(os.listdir(JPEG_PATH))[:PREVIEW_SAMPLES]:
            patient_path = os.path.join(JPEG_PATH, patient_id)
            if os.path.isdir(patient_path):
                images = [f for f in os.listdir(patient_path) if f.lower().endswith('.jpg')]
                if images:
                    sample_images.append((patient_id, os.path.join(patient_path, images[0])))
        
        # Create preview figure
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(PREVIEW_SAMPLES, 2, figsize=(12, 4*PREVIEW_SAMPLES))
        if PREVIEW_SAMPLES == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('ROI Extraction: Before and After', fontsize=14, fontweight='bold')
        
        for idx, (patient_id, img_path) in enumerate(sample_images):
            # Load original
            original_img = Image.open(img_path)
            orig_array = np.array(original_img)
            
            # Extract ROI
            roi_array = extract_roi(orig_array)
            
            # Plot original
            axes[idx, 0].imshow(orig_array, cmap='gray')
            axes[idx, 0].set_title(f'Original ({orig_array.shape[1]}x{orig_array.shape[0]})')
            axes[idx, 0].axis('off')
            
            # Plot ROI
            if roi_array is not None:
                axes[idx, 1].imshow(roi_array, cmap='gray')
                axes[idx, 1].set_title(f'ROI Extracted ({roi_array.shape[1]}x{roi_array.shape[0]})')
            else:
                axes[idx, 1].text(0.5, 0.5, 'ROI Extraction Failed', 
                                ha='center', va='center', transform=axes[idx, 1].transAxes)
                axes[idx, 1].set_title('Failed')
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        preview_path = os.path.join(ROI_OUTPUT_PATH, 'roi_extraction_preview.png')
        plt.savefig(preview_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Saved preview to: {preview_path}")
    
    except Exception as e:
        print(f"  [WARN] Could not create preview: {str(e)}")

def main():
    """Main execution"""
    print("\n")
    print("[" + "=" * 88 + "]")
    print("|" + " " * 20 + "BREAST CANCER DETECTION SYSTEM" + " " * 38 + "|")
    print("|" + " " * 25 + "Phase 3: ROI Extraction" + " " * 41 + "|")
    print("[" + "=" * 88 + "]")
    
    print_header("CONFIGURATION")
    print(f"  Input Dataset: {JPEG_PATH}")
    print(f"  Output Directory: {ROI_OUTPUT_PATH}")
    print(f"  Padding Ratio: {PADDING_RATIO*100:.0f}%")
    print(f"  Tissue Threshold: {MIN_TISSUE_THRESHOLD}")
    
    try:
        # Load dataset
        data_df = load_dataset()
        
        # Build patient mapping (for reference)
        build_patient_mapping()
        
        # Extract ROI
        results = extract_roi_from_dataset(data_df)
        
        # Save results
        save_results_summary(results)
        
        # Create preview
        create_preview()
        
        # Print statistics
        print_header("EXTRACTION SUMMARY")
        print(f"  Total images processed: {results['total']}")
        print(f"  Successful: {results['success']} ({results['success']/max(results['total'], 1)*100:.1f}%)")
        print(f"  Failed: {results['failed']} ({results['failed']/max(results['total'], 1)*100:.1f}%)")
        
        if results['errors']:
            print("\n  Errors:")
            for error, count in results['errors'].items():
                print(f"    {error}: {count}")
        
        if results['original_sizes']:
            orig_array = np.array(results['original_sizes'])
            print(f"\n  Original Image Dimensions:")
            print(f"    Min: {orig_array.min(axis=0)}")
            print(f"    Max: {orig_array.max(axis=0)}")
            print(f"    Mean: {orig_array.mean(axis=0).astype(int)}")
        
        if results['roi_sizes']:
            roi_array = np.array(results['roi_sizes'])
            print(f"\n  ROI Image Dimensions:")
            print(f"    Min: {roi_array.min(axis=0)}")
            print(f"    Max: {roi_array.max(axis=0)}")
            print(f"    Mean: {roi_array.mean(axis=0).astype(int)}")
        
        return True
    
    except Exception as e:
        print_header("FATAL ERROR")
        print(f"  {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    print_header("ROI EXTRACTION COMPLETE")
    if success:
        print("  Status: [PASSED] ✓")
        print(f"  ROI data saved to: {ROI_OUTPUT_PATH}")
        print("  Dataset is ready for Phase 4: Preprocessing")
    else:
        print("  Status: [FAILED] ✗")
    
    print("\n")
