"""
Local Testing Script - Verify Model Architecture
This script tests that the model architecture works correctly before Colab training
Can be run locally to verify everything is set up properly
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / 'code'))

from model_architecture import DualViewCNN, create_dual_view_model, count_parameters


def test_model_initialization():
    """Test model can be initialized"""
    print("\n" + "="*80)
    print("TEST 1: MODEL INITIALIZATION")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    try:
        model = create_dual_view_model(device=device, num_classes=2, dropout_rate=0.5)
        print("[✓] Model initialized successfully")
        return model
    except Exception as e:
        print(f"[✗] Failed to initialize model: {e}")
        return None


def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("\n" + "="*80)
    print("TEST 2: FORWARD PASS")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create dummy input: batch of 4 samples
        # Shape: (batch_size, channels, height, width) = (4, 2, 512, 512)
        dummy_input = torch.randn(4, 2, 512, 512).to(device)
        
        print(f"Input shape: {dummy_input.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: torch.Size([4, 2])")
        
        if output.shape == torch.Size([4, 2]):
            print("[✓] Forward pass successful")
            
            # Show sample outputs
            print(f"\nSample logits:\n{output.cpu().numpy()[:2]}")
            
            probs = torch.softmax(output, dim=1)
            print(f"\nSample probabilities:\n{probs.cpu().numpy()[:2]}")
            
            return True
        else:
            print("[✗] Output shape mismatch")
            return False
    
    except Exception as e:
        print(f"[✗] Forward pass failed: {e}")
        return False


def test_backward_pass(model):
    """Test backward pass (gradient computation)"""
    print("\n" + "="*80)
    print("TEST 3: BACKWARD PASS")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create dummy data
        dummy_input = torch.randn(2, 2, 512, 512, requires_grad=True).to(device)
        dummy_labels = torch.tensor([0, 1]).to(device)
        
        # Forward pass
        output = model(dummy_input)
        loss = nn.CrossEntropyLoss()(output, dummy_labels)
        
        print(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        if has_gradients:
            print("[✓] Backward pass successful")
            print("[✓] Gradients computed")
            return True
        else:
            print("[✗] No gradients computed")
            return False
    
    except Exception as e:
        print(f"[✗] Backward pass failed: {e}")
        return False


def test_parameter_counts(model):
    """Test parameter counting"""
    print("\n" + "="*80)
    print("TEST 4: PARAMETER COUNTS")
    print("="*80)
    
    try:
        total, trainable = count_parameters(model)
        frozen = total - trainable
        
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters:    {frozen:,}")
        print(f"Trainable percentage: {trainable/total*100:.1f}%")
        
        if trainable > 0 and frozen > 0:
            print("[✓] Parameter configuration valid")
            return True
        else:
            print("[✗] Parameter configuration invalid")
            return False
    
    except Exception as e:
        print(f"[✗] Parameter counting failed: {e}")
        return False


def test_grayscale_conversion():
    """Test grayscale to RGB conversion in first layer"""
    print("\n" + "="*80)
    print("TEST 5: GRAYSCALE CONVERSION")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DualViewCNN().to(device)
        
        # Check conv1 layers
        print(f"CC branch conv1 input channels: {model.conv1_cc.in_channels}")
        print(f"MLO branch conv1 input channels: {model.conv1_mlo.in_channels}")
        
        if model.conv1_cc.in_channels == 1 and model.conv1_mlo.in_channels == 1:
            print("[✓] Grayscale input channels correct (1)")
            return True
        else:
            print("[✗] Grayscale input channels incorrect")
            return False
    
    except Exception as e:
        print(f"[✗] Grayscale conversion test failed: {e}")
        return False


def test_model_save_load():
    """Test saving and loading model"""
    print("\n" + "="*80)
    print("TEST 6: MODEL SAVE/LOAD")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        model1 = create_dual_view_model(device=device)
        
        # Get initial weights
        initial_weights = next(model1.parameters()).data.clone()
        
        # Save model
        save_path = 'test_model.pt'
        torch.save(model1.state_dict(), save_path)
        print(f"[✓] Model saved to {save_path}")
        
        # Create new model and load weights
        model2 = DualViewCNN().to(device)
        model2.load_state_dict(torch.load(save_path, map_location=device))
        print(f"[✓] Model loaded successfully")
        
        # Check weights match
        loaded_weights = next(model2.parameters()).data.clone()
        
        if torch.allclose(initial_weights, loaded_weights):
            print("[✓] Weights match after save/load")
            
            # Cleanup
            import os
            os.remove(save_path)
            
            return True
        else:
            print("[✗] Weights don't match after save/load")
            return False
    
    except Exception as e:
        print(f"[✗] Save/load test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "MODEL ARCHITECTURE - LOCAL VERIFICATION TESTS" + " "*19 + "║")
    print("╚" + "="*78 + "╝")
    
    # Check GPU availability
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Running on CPU (tests will be slower)")
    
    # Run tests
    results = {}
    
    # Test 1: Initialization
    model = test_model_initialization()
    results['initialization'] = model is not None
    
    if model is None:
        print("\n[✗] Cannot continue - model initialization failed")
        return results
    
    # Test 2: Forward pass
    results['forward'] = test_forward_pass(model)
    
    # Test 3: Backward pass
    results['backward'] = test_backward_pass(model)
    
    # Test 4: Parameter counts
    results['parameters'] = test_parameter_counts(model)
    
    # Test 5: Grayscale conversion
    results['grayscale'] = test_grayscale_conversion()
    
    # Test 6: Save/load
    results['save_load'] = test_model_save_load()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n[✓] ALL TESTS PASSED - Model is ready for training!")
        return True
    else:
        print(f"\n[✗] {total_tests - passed_tests} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    print("\n" + "="*80)
    if success:
        print("✓ Ready to train on Google Colab!")
        print("  1. Upload model_architecture.py to Colab")
        print("  2. Upload train_model.py to Colab")
        print("  3. Upload preprocessed data to Google Drive")
        print("  4. Run training in Colab notebook")
    else:
        print("✗ Fix the errors above before proceeding to Colab")
    print("="*80)
    
    sys.exit(0 if success else 1)
