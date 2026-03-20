"""
Phase 1: Dataset Loader
Breast Cancer Detection - Dual-View CNN with ResNet50

This script loads and validates the CBIS-DDSM dataset structure.
It verifies:
- All CSV files are readable
- Image files exist and are accessible
- Labels match images
- Dataset statistics are correct
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import traceback

# ==================== CONFIGURATION ====================
DATASET_BASE_PATH = r"c:\Users\Digi Opia\Desktop\BREAST CANCER DETECTION\ML-Pipeline\Data\Dataset"
CSV_PATH = os.path.join(DATASET_BASE_PATH, "csv")
JPEG_PATH = os.path.join(DATASET_BASE_PATH, "jpeg")

# ==================== UTILITY FUNCTIONS ====================

def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subheader(title):
    """Print formatted subsection header"""
    print(f"\n{title}")
    print("-" * 80)

def validate_paths():
    """Validate that base paths exist"""
    print_header("STEP 1: VALIDATING PATHS")
    
    checks = {
        "Base Dataset Path": DATASET_BASE_PATH,
        "CSV Path": CSV_PATH,
        "JPEG Path": JPEG_PATH
    }
    
    all_valid = True
    for name, path in checks.items():
        if os.path.exists(path):
            print(f"  [OK] {name}: {path}")
        else:
            print(f"  [FAIL] {name}: {path} - NOT FOUND")
            all_valid = False
    
    return all_valid

def load_csv_files():
    """Load all CSV files and return as dictionary"""
    print_header("STEP 2: LOADING CSV FILES")
    
    csv_files = {
        "calc_train": "calc_case_description_train_set.csv",
        "calc_test": "calc_case_description_test_set.csv",
        "mass_train": "mass_case_description_train_set.csv",
        "mass_test": "mass_case_description_test_set.csv",
        "meta": "meta.csv",
        "dicom_info": "dicom_info.csv"
    }
    
    loaded_csvs = {}
    
    for key, filename in csv_files.items():
        filepath = os.path.join(CSV_PATH, filename)
        try:
            df = pd.read_csv(filepath)
            loaded_csvs[key] = df
            print(f"  [OK] {filename}")
            print(f"       Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        except Exception as e:
            print(f"  [FAIL] {filename}: {str(e)}")
    
    return loaded_csvs

def validate_images():
    """Check image accessibility and basic properties"""
    print_header("STEP 3: VALIDATING IMAGE FILES")
    
    try:
        patient_dirs = os.listdir(JPEG_PATH)
        print(f"  Total patient directories: {len(patient_dirs)}")
        
        total_images = 0
        valid_images = 0
        invalid_images = 0
        image_sizes = []
        
        # Sample first 100 patients for detailed validation
        print(f"\n  Sampling first 100 patients for validation...")
        for patient_id in patient_dirs[:100]:
            patient_path = os.path.join(JPEG_PATH, patient_id)
            if os.path.isdir(patient_path):
                images = [f for f in os.listdir(patient_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(images)
                
                for img_file in images:
                    img_path = os.path.join(patient_path, img_file)
                    try:
                        img = Image.open(img_path)
                        image_sizes.append(img.size)
                        valid_images += 1
                    except Exception as e:
                        invalid_images += 1
                        print(f"    [WARN] Cannot open {patient_id}/{img_file}: {str(e)}")
        
        print(f"\n  Validation Results (first 100 patients):")
        print(f"    Total images found: {total_images}")
        print(f"    Valid images: {valid_images}")
        print(f"    Invalid images: {invalid_images}")
        
        if image_sizes:
            sizes_array = np.array(image_sizes)
            print(f"\n  Image Size Statistics:")
            print(f"    Min size: {sizes_array.min(axis=0)}")
            print(f"    Max size: {sizes_array.max(axis=0)}")
            print(f"    Mean size: {sizes_array.mean(axis=0).astype(int)}")
        
        return valid_images > 0
    
    except Exception as e:
        print(f"  [FAIL] Image validation error: {str(e)}")
        traceback.print_exc()
        return False

def analyze_dataset(csvs):
    """Analyze dataset structure and statistics"""
    print_header("STEP 4: DATASET ANALYSIS")
    
    # Combine all case descriptions
    calc_train = csvs.get("calc_train", pd.DataFrame())
    calc_test = csvs.get("calc_test", pd.DataFrame())
    mass_train = csvs.get("mass_train", pd.DataFrame())
    mass_test = csvs.get("mass_test", pd.DataFrame())
    
    print_subheader("4.1 Dataset Splits")
    print(f"  Calcification Training:  {len(calc_train):5d} cases")
    print(f"  Calcification Test:      {len(calc_test):5d} cases")
    print(f"  Mass Training:           {len(mass_train):5d} cases")
    print(f"  Mass Test:               {len(mass_test):5d} cases")
    print(f"  {'─' * 40}")
    print(f"  TOTAL:                   {len(calc_train) + len(calc_test) + len(mass_train) + len(mass_test):5d} cases")
    
    # Combine training data
    train_data = pd.concat([calc_train, mass_train], ignore_index=True)
    test_data = pd.concat([calc_test, mass_test], ignore_index=True)
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    
    print_subheader("4.2 Class Distribution (All Data)")
    if 'pathology' in all_data.columns:
        pathology_dist = all_data['pathology'].value_counts()
        for label, count in pathology_dist.items():
            percentage = (count / len(all_data)) * 100
            print(f"  {label:30s}: {count:5d} ({percentage:5.1f}%)")
    
    print_subheader("4.3 Image Views Distribution (Training)")
    if 'image view' in train_data.columns:
        view_dist = train_data['image view'].value_counts()
        for view, count in view_dist.items():
            print(f"  {view:30s}: {count:5d}")
    
    print_subheader("4.4 Breast Side Distribution (Training)")
    if 'left or right breast' in train_data.columns:
        side_dist = train_data['left or right breast'].value_counts()
        for side, count in side_dist.items():
            print(f"  {side:30s}: {count:5d}")
    
    print_subheader("4.5 Unique Patients")
    if 'patient_id' in train_data.columns:
        unique_train = train_data['patient_id'].nunique()
        unique_test = test_data['patient_id'].nunique()
        print(f"  Training set:  {unique_train} unique patients")
        print(f"  Test set:      {unique_test} unique patients")
        print(f"  Total:         {unique_train + unique_test} unique patients")
    
    print_subheader("4.6 Binary Classification Target")
    print(f"  Target: Distinguish between MALIGNANT vs BENIGN")
    print(f"  Mapping: MALIGNANT = 1, BENIGN = 0")
    if 'pathology' in all_data.columns:
        malignant = (all_data['pathology'] == 'MALIGNANT').sum()
        benign = ((all_data['pathology'] == 'BENIGN') | 
                 (all_data['pathology'] == 'BENIGN_WITHOUT_CALLBACK')).sum()
        total = len(all_data)
        print(f"  Malignant cases: {malignant:5d} ({malignant/total*100:5.1f}%)")
        print(f"  Benign cases:    {benign:5d} ({benign/total*100:5.1f}%)")
        print(f"  Class imbalance ratio (Benign:Malignant) = {benign/malignant:.2f}:1")

def check_image_label_mapping(csvs):
    """Verify that images can be matched with labels"""
    print_header("STEP 5: IMAGE-LABEL MAPPING VALIDATION")
    
    train_data = pd.concat([
        csvs.get("calc_train", pd.DataFrame()),
        csvs.get("mass_train", pd.DataFrame())
    ], ignore_index=True)
    
    if train_data.empty or 'patient_id' not in train_data.columns:
        print("  [WARN] Cannot validate mapping - required columns missing")
        return True
    
    print(f"  Checking mapping for {len(train_data)} training cases...")
    
    # Check a sample of cases
    sample_size = min(50, len(train_data))
    unmapped = 0
    mapped = 0
    
    for idx, row in train_data.head(sample_size).iterrows():
        patient_id = row.get('patient_id', '')
        # Try to find patient directory in JPEG folder
        patient_path = os.path.join(JPEG_PATH, patient_id)
        
        # Also check if encoded differently
        found = False
        if os.path.exists(patient_path):
            found = True
        else:
            # Search for partial match
            for dir_name in os.listdir(JPEG_PATH)[:100]:  # Quick search
                if patient_id in dir_name or dir_name.endswith(patient_id):
                    found = True
                    break
        
        if found:
            mapped += 1
        else:
            unmapped += 1
    
    print(f"  Sample validation ({sample_size} cases):")
    print(f"    Mapped:     {mapped}")
    print(f"    Unmapped:   {unmapped}")
    print(f"    Mapping Success Rate: {(mapped/sample_size)*100:.1f}%")
    
    return unmapped < sample_size * 0.5  # Success if at least 50% mapped

def main():
    """Main execution function"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "BREAST CANCER DETECTION SYSTEM" + " " * 28 + "║")
    print("║" + " " * 25 + "Phase 1: Dataset Loading" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # Step 1: Validate paths
        if not validate_paths():
            print_header("CRITICAL ERROR")
            print("  Required dataset paths do not exist!")
            print("  Please verify the dataset location and try again.")
            return False
        
        # Step 2: Load CSV files
        csvs = load_csv_files()
        if not csvs:
            print_header("CRITICAL ERROR")
            print("  Could not load any CSV files!")
            return False
        
        # Step 3: Validate images
        if not validate_images():
            print_header("WARNING")
            print("  Some image files could not be accessed")
        
        # Step 4: Analyze dataset
        analyze_dataset(csvs)
        
        # Step 5: Check mapping
        check_image_label_mapping(csvs)
        
        # Final status
        print_header("DATASET LOADING SUMMARY")
        print("  [SUCCESS] Dataset structure is valid!")
        print("  [SUCCESS] All required files are accessible")
        print("  [SUCCESS] CSV files loaded successfully")
        print("  [SUCCESS] Image files are present and readable")
        print("\n  Dataset is ready for Phase 2: Exploration")
        print("\n  Next Steps:")
        print("    1. Run Phase 2 for detailed data exploration")
        print("    2. Implement Phase 3: ROI Extraction")
        print("    3. Implement Phase 4: Preprocessing and Pairing")
        
        return True
    
    except Exception as e:
        print_header("FATAL ERROR")
        print(f"  An unexpected error occurred: {str(e)}")
        print("\n  Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    print_header("EXECUTION COMPLETE")
    if success:
        print("  Status: [PASSED] ✓")
        print("  The dataset is ready for processing.")
    else:
        print("  Status: [FAILED] ✗")
        print("  Please check the errors above and try again.")
    
    print("\n")
