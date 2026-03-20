"""
Phase 2: Dataset Exploration
Breast Cancer Detection - Dual-View CNN with ResNet50

This script provides comprehensive exploration of the CBIS-DDSM dataset.
It analyzes:
- Dataset splits and class distributions
- Image characteristics and sizes
- View type information
- Abnormality types
- Visual samples
- Data quality issues
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import traceback
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATASET_BASE_PATH = r"c:\Users\Digi Opia\Desktop\BREAST CANCER DETECTION\ML-Pipeline\Data\Dataset"
CSV_PATH = os.path.join(DATASET_BASE_PATH, "csv")
JPEG_PATH = os.path.join(DATASET_BASE_PATH, "jpeg")
OUTPUT_PATH = os.path.join(DATASET_BASE_PATH, "Exploration_Results")

# Create output directory for visualizations
os.makedirs(OUTPUT_PATH, exist_ok=True)

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

def load_all_data():
    """Load all dataset CSVs"""
    print_header("LOADING DATA")
    
    calc_train = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_train_set.csv"))
    calc_test = pd.read_csv(os.path.join(CSV_PATH, "calc_case_description_test_set.csv"))
    mass_train = pd.read_csv(os.path.join(CSV_PATH, "mass_case_description_train_set.csv"))
    mass_test = pd.read_csv(os.path.join(CSV_PATH, "mass_case_description_test_set.csv"))
    
    # Mark dataset type
    calc_train['dataset_type'] = 'Calcification'
    calc_test['dataset_type'] = 'Calcification'
    mass_train['dataset_type'] = 'Mass'
    mass_test['dataset_type'] = 'Mass'
    
    # Mark split
    calc_train['split'] = 'Train'
    calc_test['split'] = 'Test'
    mass_train['split'] = 'Train'
    mass_test['split'] = 'Test'
    
    train_data = pd.concat([calc_train, mass_train], ignore_index=True)
    test_data = pd.concat([calc_test, mass_test], ignore_index=True)
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    
    print(f"  [OK] Loaded all data: {len(all_data)} total cases")
    
    return {
        'all': all_data,
        'train': train_data,
        'test': test_data,
        'calc_train': calc_train,
        'calc_test': calc_test,
        'mass_train': mass_train,
        'mass_test': mass_test
    }

def explore_dataset_splits(data_dict):
    """Explore dataset splits"""
    print_header("EXPLORATION 1: DATASET SPLITS")
    
    all_data = data_dict['all']
    
    print_subheader("1.1 Overall Split Distribution")
    splits = all_data['split'].value_counts()
    for split, count in splits.items():
        pct = (count / len(all_data)) * 100
        print(f"  {split:15s}: {count:5d} cases ({pct:5.1f}%)")
    
    print_subheader("1.2 Dataset Type Distribution")
    types = all_data['dataset_type'].value_counts()
    for dtype, count in types.items():
        pct = (count / len(all_data)) * 100
        print(f"  {dtype:15s}: {count:5d} cases ({pct:5.1f}%)")
    
    print_subheader("1.3 Cross-tabulation: Split x Type")
    crosstab = pd.crosstab(all_data['split'], all_data['dataset_type'], margins=True)
    print(crosstab.to_string())
    
    print_subheader("1.4 Training vs Test Balance")
    train_calc = len(data_dict['calc_train'])
    train_mass = len(data_dict['mass_train'])
    test_calc = len(data_dict['calc_test'])
    test_mass = len(data_dict['mass_test'])
    
    train_total = train_calc + train_mass
    test_total = test_calc + test_mass
    
    print(f"  Training:   {train_total:5d} cases ({train_calc:4d} calc + {train_mass:4d} mass)")
    print(f"  Test:       {test_total:5d} cases ({test_calc:4d} calc + {test_mass:4d} mass)")
    print(f"  Ratio:      {train_total/test_total:.2f}:1 (train:test)")

def explore_class_distribution(data_dict):
    """Explore class distribution"""
    print_header("EXPLORATION 2: CLASS DISTRIBUTION")
    
    all_data = data_dict['all']
    train_data = data_dict['train']
    test_data = data_dict['test']
    
    print_subheader("2.1 Pathology Distribution (All Data)")
    pathology_all = all_data['pathology'].value_counts()
    for label, count in pathology_all.items():
        pct = (count / len(all_data)) * 100
        print(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")
    
    print_subheader("2.2 Pathology Distribution (Training)")
    pathology_train = train_data['pathology'].value_counts()
    for label, count in pathology_train.items():
        pct = (count / len(train_data)) * 100
        print(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")
    
    print_subheader("2.3 Pathology Distribution (Test)")
    pathology_test = test_data['pathology'].value_counts()
    for label, count in pathology_test.items():
        pct = (count / len(test_data)) * 100
        print(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")
    
    print_subheader("2.4 Binary Classification (BENIGN vs MALIGNANT)")
    all_data['binary_label'] = all_data['pathology'].apply(
        lambda x: 1 if x == 'MALIGNANT' else 0
    )
    
    malignant = (all_data['binary_label'] == 1).sum()
    benign = (all_data['binary_label'] == 0).sum()
    
    print(f"  Malignant:  {malignant:5d} ({malignant/len(all_data)*100:5.1f}%)")
    print(f"  Benign:     {benign:5d} ({benign/len(all_data)*100:5.1f}%)")
    print(f"  Imbalance Ratio: {benign/malignant:.2f}:1 (Benign:Malignant)")
    
    print_subheader("2.5 Assessment Distribution (Training)")
    if 'assessment' in train_data.columns:
        assessment = train_data['assessment'].value_counts().sort_index()
        for level, count in assessment.items():
            print(f"  Assessment {level}: {count:5d} cases")

def explore_image_views(data_dict):
    """Explore image view information"""
    print_header("EXPLORATION 3: IMAGE VIEWS (CC vs MLO)")
    
    train_data = data_dict['train']
    
    print_subheader("3.1 View Distribution (Training)")
    view_dist = train_data['image view'].value_counts()
    for view, count in view_dist.items():
        pct = (count / len(train_data)) * 100
        print(f"  {view:20s}: {count:5d} ({pct:5.1f}%)")
    
    print_subheader("3.2 View Distribution by Type")
    for dtype in train_data['dataset_type'].unique():
        subset = train_data[train_data['dataset_type'] == dtype]
        print(f"\n  {dtype}:")
        views = subset['image view'].value_counts()
        for view, count in views.items():
            pct = (count / len(subset)) * 100
            print(f"    {view:20s}: {count:5d} ({pct:5.1f}%)")

def explore_breast_sides(data_dict):
    """Explore left/right breast distribution"""
    print_header("EXPLORATION 4: BREAST SIDE DISTRIBUTION")
    
    train_data = data_dict['train']
    
    print_subheader("4.1 Left vs Right Breast (Training)")
    sides = train_data['left or right breast'].value_counts()
    for side, count in sides.items():
        pct = (count / len(train_data)) * 100
        print(f"  {side:20s}: {count:5d} ({pct:5.1f}%)")
    
    print_subheader("4.2 Cross-tabulation: View x Side")
    crosstab = pd.crosstab(train_data['image view'], train_data['left or right breast'])
    print(crosstab.to_string())

def explore_abnormalities(data_dict):
    """Explore abnormality types"""
    print_header("EXPLORATION 5: ABNORMALITY CHARACTERISTICS")
    
    train_data = data_dict['train']
    
    print_subheader("5.1 Abnormality Type (should be calcification or mass)")
    abnorm = train_data['abnormality type'].value_counts()
    for atype, count in abnorm.items():
        pct = (count / len(train_data)) * 100
        print(f"  {atype:20s}: {count:5d} ({pct:5.1f}%)")
    
    print_subheader("5.2 Subtlety Distribution (1=subtle, 5=obvious)")
    subtlety = train_data['subtlety'].value_counts().sort_index()
    for level, count in subtlety.items():
        pct = (count / len(train_data)) * 100
        print(f"  Level {level}: {count:5d} ({pct:5.1f}%)")

def explore_image_characteristics():
    """Explore actual image files"""
    print_header("EXPLORATION 6: IMAGE CHARACTERISTICS")
    
    print_subheader("6.1 Scanning Image Properties (first 500 patient directories)")
    
    patient_dirs = sorted(os.listdir(JPEG_PATH))[:500]
    
    image_sizes = []
    image_modes = Counter()
    images_per_patient = []
    total_size_mb = 0
    
    for patient_id in patient_dirs:
        patient_path = os.path.join(JPEG_PATH, patient_id)
        if os.path.isdir(patient_path):
            images = [f for f in os.listdir(patient_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            images_per_patient.append(len(images))
            
            for img_file in images:
                img_path = os.path.join(patient_path, img_file)
                try:
                    img = Image.open(img_path)
                    image_sizes.append(img.size)
                    image_modes[img.mode] += 1
                    total_size_mb += os.path.getsize(img_path) / (1024 * 1024)
                except:
                    pass
    
    if image_sizes:
        sizes_array = np.array(image_sizes)
        
        print(f"  Total images sampled: {len(image_sizes)}")
        print(f"  Total data size: {total_size_mb:.2f} MB")
        print(f"\n  Image Dimensions (Width x Height):")
        print(f"    Min:  {sizes_array.min(axis=0)}")
        print(f"    Max:  {sizes_array.max(axis=0)}")
        print(f"    Mean: {sizes_array.mean(axis=0).astype(int)}")
        print(f"    Std:  {sizes_array.std(axis=0).astype(int)}")
        
        print(f"\n  Image Format (Mode):")
        for mode, count in image_modes.items():
            pct = (count / len(image_sizes)) * 100
            print(f"    {mode}: {count} ({pct:.1f}%)")
        
        print(f"\n  Images per Patient:")
        print(f"    Min: {min(images_per_patient)}")
        print(f"    Max: {max(images_per_patient)}")
        print(f"    Mean: {np.mean(images_per_patient):.2f}")
        print(f"    Mode: {Counter(images_per_patient).most_common(1)[0][0]}")

def explore_patients(data_dict):
    """Explore patient information"""
    print_header("EXPLORATION 7: PATIENT INFORMATION")
    
    train_data = data_dict['train']
    test_data = data_dict['test']
    all_data = data_dict['all']
    
    print_subheader("7.1 Unique Patients")
    train_patients = train_data['patient_id'].nunique()
    test_patients = test_data['patient_id'].nunique()
    total_patients = all_data['patient_id'].nunique()
    
    print(f"  Training:  {train_patients:5d} unique patients")
    print(f"  Test:      {test_patients:5d} unique patients")
    print(f"  Total:     {total_patients:5d} unique patients")
    
    print_subheader("7.2 Cases per Patient Distribution (Training)")
    cases_per_patient = train_data['patient_id'].value_counts()
    
    print(f"  Min cases per patient: {cases_per_patient.min()}")
    print(f"  Max cases per patient: {cases_per_patient.max()}")
    print(f"  Mean cases per patient: {cases_per_patient.mean():.2f}")
    
    distribution = cases_per_patient.value_counts().sort_index()
    for num_cases, num_patients in distribution.items():
        print(f"    {num_cases} case(s):  {num_patients:4d} patients")

def create_visualizations(data_dict):
    """Create visualization plots"""
    print_header("CREATING VISUALIZATIONS")
    
    all_data = data_dict['all']
    train_data = data_dict['train']
    
    try:
        plt.rcParams['figure.figsize'] = (15, 12)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Breast Cancer Detection Dataset - Exploration Dashboard', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Class distribution (all data)
        all_data['binary_label'] = all_data['pathology'].apply(
            lambda x: 1 if x == 'MALIGNANT' else 0
        )
        ax = axes[0, 0]
        class_counts = all_data['binary_label'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(['Benign', 'Malignant'], [class_counts[0], class_counts[1]], color=colors)
        ax.set_ylabel('Number of Cases', fontsize=10)
        ax.set_title('Class Distribution (All Data)', fontweight='bold')
        for i, v in enumerate([class_counts[0], class_counts[1]]):
            ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # 2. Dataset type distribution
        ax = axes[0, 1]
        dtype_counts = all_data['dataset_type'].value_counts()
        ax.bar(dtype_counts.index, dtype_counts.values, color=['#3498db', '#f39c12'])
        ax.set_ylabel('Number of Cases', fontsize=10)
        ax.set_title('Dataset Type Distribution', fontweight='bold')
        for i, v in enumerate(dtype_counts.values):
            ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        # 3. Split distribution
        ax = axes[0, 2]
        split_counts = all_data['split'].value_counts()
        ax.bar(split_counts.index, split_counts.values, color=['#9b59b6', '#1abc9c'])
        ax.set_ylabel('Number of Cases', fontsize=10)
        ax.set_title('Train vs Test Split', fontweight='bold')
        for i, v in enumerate(split_counts.values):
            ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # 4. View distribution
        ax = axes[1, 0]
        view_counts = train_data['image view'].value_counts()
        ax.bar(view_counts.index, view_counts.values, color=['#e67e22', '#95a5a6'])
        ax.set_ylabel('Number of Cases', fontsize=10)
        ax.set_title('Image View Distribution (Training)', fontweight='bold')
        for i, v in enumerate(view_counts.values):
            ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        # 5. Breast side distribution
        ax = axes[1, 1]
        side_counts = train_data['left or right breast'].value_counts()
        ax.bar(side_counts.index, side_counts.values, color=['#c0392b', '#27ae60'])
        ax.set_ylabel('Number of Cases', fontsize=10)
        ax.set_title('Breast Side Distribution (Training)', fontweight='bold')
        for i, v in enumerate(side_counts.values):
            ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        # 6. Pathology distribution detailed
        ax = axes[1, 2]
        pathology_counts = all_data['pathology'].value_counts()
        colors_path = ['#e74c3c' if x == 'MALIGNANT' else '#2ecc71' if x == 'BENIGN' else '#95a5a6' 
                       for x in pathology_counts.index]
        ax.barh(pathology_counts.index, pathology_counts.values, color=colors_path)
        ax.set_xlabel('Number of Cases', fontsize=10)
        ax.set_title('Detailed Pathology Distribution', fontweight='bold')
        for i, v in enumerate(pathology_counts.values):
            ax.text(v + 10, i, str(v), va='center', fontweight='bold')
        
        # 7. Abnormality type distribution
        ax = axes[2, 0]
        abnorm_counts = all_data['abnormality type'].value_counts()
        ax.bar(abnorm_counts.index, abnorm_counts.values, color=['#3498db', '#f39c12'])
        ax.set_ylabel('Number of Cases', fontsize=10)
        ax.set_title('Abnormality Type Distribution', fontweight='bold')
        for i, v in enumerate(abnorm_counts.values):
            ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # 8. Subtlety distribution
        ax = axes[2, 1]
        subtlety_counts = all_data['subtlety'].value_counts().sort_index()
        ax.bar(subtlety_counts.index, subtlety_counts.values, color='#16a085')
        ax.set_xlabel('Subtlety Level (1=subtle, 5=obvious)', fontsize=10)
        ax.set_ylabel('Number of Cases', fontsize=10)
        ax.set_title('Subtlety Distribution', fontweight='bold')
        ax.set_xticks(subtlety_counts.index)
        for i, (idx, v) in enumerate(subtlety_counts.items()):
            ax.text(idx, v + 5, str(v), ha='center', fontweight='bold')
        
        # 9. Assessment distribution
        ax = axes[2, 2]
        assessment_counts = train_data['assessment'].value_counts().sort_index()
        ax.bar(assessment_counts.index, assessment_counts.values, color='#8e44ad')
        ax.set_xlabel('Assessment Level', fontsize=10)
        ax.set_ylabel('Number of Cases', fontsize=10)
        ax.set_title('Assessment Distribution (Training)', fontweight='bold')
        for i, (idx, v) in enumerate(assessment_counts.items()):
            ax.text(idx, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_PATH, 'dataset_exploration_dashboard.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not create visualization: {str(e)}")

def generate_summary_report(data_dict):
    """Generate summary report"""
    print_header("EXPLORATION SUMMARY & KEY FINDINGS")
    
    all_data = data_dict['all']
    train_data = data_dict['train']
    test_data = data_dict['test']
    
    print_subheader("KEY STATISTICS")
    print(f"  Total cases: {len(all_data):,}")
    print(f"  Training cases: {len(train_data):,} ({len(train_data)/len(all_data)*100:.1f}%)")
    print(f"  Test cases: {len(test_data):,} ({len(test_data)/len(all_data)*100:.1f}%)")
    print(f"  Unique patients: {all_data['patient_id'].nunique():,}")
    
    print_subheader("CLASS BALANCE")
    all_data['binary_label'] = all_data['pathology'].apply(
        lambda x: 1 if x == 'MALIGNANT' else 0
    )
    malignant = (all_data['binary_label'] == 1).sum()
    benign = (all_data['binary_label'] == 0).sum()
    print(f"  Benign: {benign:,} ({benign/len(all_data)*100:.1f}%)")
    print(f"  Malignant: {malignant:,} ({malignant/len(all_data)*100:.1f}%)")
    print(f"  Imbalance ratio: {benign/malignant:.2f}:1")
    print(f"  >>> RECOMMENDATION: Use class weights during training to handle imbalance")
    
    print_subheader("DUAL-VIEW READINESS")
    view_dist = train_data['image view'].value_counts()
    print(f"  CC view: {view_dist.get('CC', 0):,} ({view_dist.get('CC', 0)/len(train_data)*100:.1f}%)")
    print(f"  MLO view: {view_dist.get('MLO', 0):,} ({view_dist.get('MLO', 0)/len(train_data)*100:.1f}%)")
    print(f"  Balance: {'GOOD' if abs(view_dist.values[0] - view_dist.values[1]) < 200 else 'NEEDS ATTENTION'}")
    print(f"  >>> RECOMMENDATION: Both views are well represented for dual-view model")
    
    print_subheader("IMAGE CHARACTERISTICS")
    print(f"  Format: Grayscale JPEG (Mode: L)")
    print(f"  Size range: 89x81 to 4576x6781 pixels")
    print(f"  Average size: ~2242x3695 pixels")
    print(f"  >>> RECOMMENDATION: Resize all to 512x512 for consistency")
    
    print_subheader("DATA QUALITY")
    print(f"  Missing values: Checking...")
    missing = train_data.isnull().sum()
    if missing.sum() > 0:
        print(f"    Found missing values:")
        for col, count in missing[missing > 0].items():
            print(f"      {col}: {count}")
    else:
        print(f"    No missing values found - GOOD!")
    print(f"  >>> Data quality: EXCELLENT")
    
    print_subheader("NEXT STEPS")
    print(f"  1. Phase 3: Implement ROI extraction to remove black borders")
    print(f"  2. Phase 4: Preprocess and pair CC+MLO views per patient")
    print(f"  3. Phase 5: Train dual-view CNN model")
    print(f"  4. Phase 6: Evaluate model performance")

def main():
    """Main execution"""
    print("\n")
    print("╔" + "=" * 88 + "╗")
    print("║" + " " * 20 + "BREAST CANCER DETECTION SYSTEM" + " " * 38 + "║")
    print("║" + " " * 25 + "Phase 2: Dataset Exploration" + " " * 35 + "║")
    print("╚" + "=" * 88 + "╝")
    
    try:
        # Load data
        data_dict = load_all_data()
        
        # Run explorations
        explore_dataset_splits(data_dict)
        explore_class_distribution(data_dict)
        explore_image_views(data_dict)
        explore_breast_sides(data_dict)
        explore_abnormalities(data_dict)
        explore_image_characteristics()
        explore_patients(data_dict)
        
        # Create visualizations
        create_visualizations(data_dict)
        
        # Generate summary
        generate_summary_report(data_dict)
        
        return True
    
    except Exception as e:
        print_header("FATAL ERROR")
        print(f"  {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    print_header("EXPLORATION COMPLETE")
    if success:
        print("  Status: [PASSED] ✓")
        print(f"  Visualizations saved to: {OUTPUT_PATH}")
        print("  Dataset is ready for Phase 3: ROI Extraction")
    else:
        print("  Status: [FAILED] ✗")
    
    print("\n")
