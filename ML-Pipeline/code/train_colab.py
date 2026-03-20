"""
Google Colab Training Wrapper
Breast Cancer Detection - Dual-View CNN Training on Google Colab

This script sets up the environment and runs training on Google Colab with GPU support.
"""

import os
import sys
from pathlib import Path

# ==================== GOOGLE COLAB SETUP ====================

def setup_colab_environment():
    """Setup Google Colab environment"""
    print("="*80)
    print("  SETTING UP GOOGLE COLAB ENVIRONMENT")
    print("="*80)
    
    # Check if running on Colab
    try:
        from google.colab import drive
        IN_COLAB = True
        print("[OK] Running on Google Colab")
    except ImportError:
        IN_COLAB = False
        print("[INFO] Not running on Google Colab")
    
    # Mount Google Drive if on Colab
    if IN_COLAB:
        from google.colab import drive
        print("\n[ACTION] Mounting Google Drive...")
        drive.mount('/content/drive')
        print("[OK] Google Drive mounted at /content/drive")
        
        # Set working directory
        os.chdir('/content/drive/MyDrive')
        print(f"[OK] Working directory: {os.getcwd()}")
    
    return IN_COLAB


def install_requirements():
    """Install required packages"""
    print("\n" + "="*80)
    print("  INSTALLING REQUIREMENTS")
    print("="*80)
    
    requirements = [
        'torch',
        'torchvision',
        'torchaudio',
        'numpy',
        'pandas',
        'pillow',
        'matplotlib',
        'scikit-learn',
        'tqdm'
    ]
    
    print("[INFO] Installing required packages...")
    for package in requirements:
        try:
            __import__(package)
            print(f"  [OK] {package} already installed")
        except ImportError:
            print(f"  [INSTALL] {package}...")
            os.system(f"pip install -q {package}")
            print(f"  [OK] {package} installed")


def verify_data_structure(base_path):
    """Verify preprocessed data structure"""
    print("\n" + "="*80)
    print("  VERIFYING DATA STRUCTURE")
    print("="*80)
    
    data_path = Path(base_path) / "ML-Pipeline" / "Data" / "Preprocessed Data"
    
    required_dirs = ['train', 'val', 'test']
    required_files = ['metadata.json', 'preprocessing_report.txt']
    
    print(f"[CHECK] Data path: {data_path}")
    print(f"[CHECK] Data path exists: {data_path.exists()}")
    
    if data_path.exists():
        for dir_name in required_dirs:
            dir_path = data_path / dir_name
            if dir_path.exists():
                num_files = len(list(dir_path.glob('*.npy')))
                print(f"  [OK] {dir_name}/ directory exists with {num_files} files")
            else:
                print(f"  [ERROR] {dir_name}/ directory not found")
                return False
        
        for file_name in required_files:
            file_path = data_path / file_name
            if file_path.exists():
                print(f"  [OK] {file_name} found")
            else:
                print(f"  [ERROR] {file_name} not found")
                return False
        
        return True
    else:
        print(f"  [ERROR] Data path not found: {data_path}")
        return False


def print_gpu_info():
    """Print GPU information"""
    print("\n" + "="*80)
    print("  GPU INFORMATION")
    print("="*80)
    
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OK] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("[WARNING] CUDA not available - training will use CPU (very slow)")
    except Exception as e:
        print(f"[ERROR] Could not get GPU info: {e}")


# ==================== MAIN COLAB EXECUTION ====================

def run_colab_training(project_path, data_path, output_path, batch_size=32, epochs=100, learning_rate=5e-5):
    """
    Run training on Google Colab
    
    Args:
        project_path: Path to BREAST CANCER DETECTION project
        data_path: Path to preprocessed data directory
        output_path: Path to save outputs
        batch_size: Batch size for training
        epochs: Number of epochs
        learning_rate: Learning rate
    """
    print("\n" + "="*80)
    print("  STARTING MODEL TRAINING")
    print("="*80)
    
    # Add project to Python path
    sys.path.insert(0, str(Path(project_path) / "ML-Pipeline" / "code"))
    
    # Import training functions
    from train_model import main, Config
    
    # Create config
    config = Config()
    config.BATCH_SIZE = batch_size
    config.NUM_EPOCHS = epochs
    config.LEARNING_RATE = learning_rate
    
    print(f"\n[CONFIG] Batch size: {config.BATCH_SIZE}")
    print(f"[CONFIG] Epochs: {config.NUM_EPOCHS}")
    print(f"[CONFIG] Learning rate: {config.LEARNING_RATE}")
    print(f"[CONFIG] Device: {config.DEVICE}")
    
    # Run training
    model, history, eval_results = main(
        data_path=data_path,
        output_dir=output_path,
        config=config
    )
    
    print("\n" + "="*80)
    print("  TRAINING COMPLETE")
    print("="*80)
    print(f"[OK] Output saved to: {output_path}")
    print(f"[OK] Final test accuracy: {eval_results['accuracy']:.2f}%")
    
    return model, history, eval_results


# ==================== COLAB NOTEBOOK MAIN ====================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "BREAST CANCER DETECTION - MODEL TRAINING" + " "*23 + "║")
    print("║" + " "*20 + "Dual-View CNN with ResNet50" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    
    # Setup environment
    in_colab = setup_colab_environment()
    
    # Install requirements (skip if already installed)
    print("\n[INFO] Verifying packages...")
    try:
        import torch
        import numpy
        from sklearn import metrics
        print("[OK] All required packages available")
    except ImportError:
        install_requirements()
    
    # Print GPU info
    print_gpu_info()
    
    # Set paths
    if in_colab:
        project_path = "/content/drive/MyDrive/BREAST CANCER DETECTION"
    else:
        project_path = "."  # Use current directory if running locally
    
    data_path = os.path.join(project_path, "ML-Pipeline/Data/Preprocessed Data")
    output_path = os.path.join(project_path, "ML-Pipeline/training_output")
    
    # Verify data structure
    if not verify_data_structure(project_path if in_colab else project_path.split('BREAST')[0]):
        print("\n[ERROR] Data structure verification failed!")
        print("Please make sure preprocessed data is available")
        sys.exit(1)
    
    # Run training
    print("\n[ACTION] Starting training process...")
    model, history, eval_results = run_colab_training(
        project_path=project_path,
        data_path=data_path,
        output_path=output_path,
        batch_size=32,
        epochs=100,
        learning_rate=5e-5
    )
    
    print("\n[SUCCESS] Training completed successfully!")
