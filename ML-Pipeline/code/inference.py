"""
Model Inference Script
Breast Cancer Detection - Using trained model for prediction

This script loads a trained model and performs inference on new data.
Can be used locally or on Colab after training.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from PIL import Image
import argparse

from model_architecture import DualViewCNN


# ==================== INFERENCE CLASS ====================

class BreastCancerDetector:
    """Inference class for breast cancer detection"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize detector with trained model
        
        Args:
            model_path: Path to saved model weights
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = DualViewCNN(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.class_names = ['Benign', 'Malignant']
        print(f"[OK] Model loaded from {model_path}")
        print(f"[OK] Using device: {self.device}")
    
    def predict_batch(self, images):
        """
        Predict on a batch of images
        
        Args:
            images: Tensor of shape (batch_size, 2, 512, 512)
        
        Returns:
            predictions: Dictionary with predictions and confidence
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
        
        return {
            'predicted_classes': predicted_classes.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'logits': outputs.cpu().numpy()
        }
    
    def predict_single(self, dual_view_image):
        """
        Predict on a single dual-view image
        
        Args:
            dual_view_image: Array of shape (2, 512, 512) or (2, H, W)
        
        Returns:
            prediction: Dictionary with prediction details
        """
        # Ensure correct shape
        if dual_view_image.shape != (2, 512, 512):
            # Resize if needed
            from torchvision.transforms import Resize
            resize = Resize((512, 512))
            dual_view_image = resize(torch.from_numpy(dual_view_image).unsqueeze(0)).squeeze(0).numpy()
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(dual_view_image).float().unsqueeze(0)
        
        # Get prediction
        result = self.predict_batch(image_tensor)
        
        predicted_class = result['predicted_classes'][0]
        confidence = result['probabilities'][0][predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                'benign': float(result['probabilities'][0][0]),
                'malignant': float(result['probabilities'][0][1])
            },
            'logits': result['logits'][0].tolist()
        }
    
    def predict_from_file(self, npy_file_path):
        """
        Predict from a .npy file
        
        Args:
            npy_file_path: Path to .npy file
        
        Returns:
            prediction: Dictionary with prediction details
        """
        image = np.load(npy_file_path)
        return self.predict_single(image)


# ==================== BATCH INFERENCE ====================

def inference_on_test_set(model_path, test_data_path, output_path=None):
    """
    Run inference on entire test set
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data directory
        output_path: Path to save predictions (optional)
    """
    print("\n" + "="*80)
    print("  BATCH INFERENCE ON TEST SET")
    print("="*80)
    
    # Load detector
    detector = BreastCancerDetector(model_path)
    
    # Load test metadata
    metadata_path = Path(test_data_path).parent / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    test_images = metadata['test']['images']
    test_labels = metadata['test']['labels']
    test_dir = Path(test_data_path)
    
    print(f"\n[INFO] Processing {len(test_images)} test samples...")
    
    predictions = []
    correct = 0
    
    for idx, (image_file, true_label) in enumerate(zip(test_images, test_labels)):
        image_path = test_dir / image_file
        
        # Make prediction
        pred = detector.predict_single(np.load(image_path))
        
        is_correct = pred['predicted_class'] == true_label
        if is_correct:
            correct += 1
        
        predictions.append({
            'image_file': image_file,
            'true_label': true_label,
            'predicted_class': pred['predicted_class'],
            'class_name': pred['class_name'],
            'confidence': pred['confidence'],
            'probabilities': pred['probabilities'],
            'correct': is_correct
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(test_images)}")
    
    accuracy = 100 * correct / len(test_images)
    print(f"\n[OK] Inference complete")
    print(f"[OK] Accuracy: {accuracy:.2f}%")
    
    # Save predictions if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        predictions_file = output_path / 'test_predictions.json'
        with open(predictions_file, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'total_samples': len(test_images),
                'correct_predictions': correct,
                'predictions': predictions
            }, f, indent=2)
        
        print(f"[OK] Predictions saved to {predictions_file}")
    
    return predictions, accuracy


# ==================== VISUALIZATION ====================

def visualize_predictions(predictions, num_samples=10, output_path=None):
    """
    Visualize predictions with confidence
    
    Args:
        predictions: List of prediction results
        num_samples: Number of samples to visualize
        output_path: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 5, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx in range(min(num_samples, len(predictions))):
            pred = predictions[idx]
            ax = axes[idx]
            
            # Create a simple visualization showing prediction
            benign_prob = pred['probabilities']['benign']
            malignant_prob = pred['probabilities']['malignant']
            
            # Bar chart
            classes = ['Benign', 'Malignant']
            probs = [benign_prob, malignant_prob]
            colors = ['green' if pred['predicted_class'] == 0 else 'red']
            
            ax.bar(classes, probs, color=['green', 'red'], alpha=0.7)
            ax.set_ylabel('Probability')
            ax.set_title(f"Pred: {pred['class_name']} ({pred['confidence']:.2%})")
            ax.set_ylim([0, 1])
            
            # Add true label
            true_class = 'Benign' if pred['true_label'] == 0 else 'Malignant'
            ax.text(0.5, -0.25, f"True: {true_class}", 
                   ha='center', transform=ax.transAxes, fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Visualization saved to {output_path}")
        
        plt.show()
    
    except ImportError:
        print("[WARNING] Matplotlib not available for visualization")


# ==================== MAIN ====================

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Run inference on breast cancer detection model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (best_model.pt)')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_path', type=str, default='inference_output',
                       help='Path to save predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    # Run inference
    predictions, accuracy = inference_on_test_set(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        output_path=args.output_path
    )
    
    # Visualize if requested
    if args.visualize:
        viz_path = Path(args.output_path) / 'predictions_visualization.png'
        visualize_predictions(predictions, num_samples=10, output_path=str(viz_path))
    
    print("\n" + "="*80)
    print("  INFERENCE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
