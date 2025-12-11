import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Enable unsafe deserialization for Lambda layers
keras.config.enable_unsafe_deserialization()

# Load trained model with custom objects
from model import grayscale_to_rgb

custom_objects = {'grayscale_to_rgb': grayscale_to_rgb}

# Try to load the model
model = None
for model_file in ['../Model/breast_cancer_model.keras', '../Model/breast_cancer_model.h5']:
    try:
        model = keras.models.load_model(model_file, custom_objects=custom_objects)
        print(f"✓ Model loaded successfully from {model_file}")
        break
    except Exception as e:
        print(f"Could not load {model_file}: {str(e)[:80]}...")

if model is None:
    print("✗ Error: Could not load trained model")
    exit(1)

# Load your test data
test_images = np.load('../Data/Preprocessed Data/ROI/roi_preprocessed_images.npy')
test_labels = np.load('../Data/Preprocessed Data/ROI/roi_preprocessed_labels.npy')

# Make predictions
predictions = model.predict(test_images)

# Convert probabilities to class (0=benign, 1=malignant)
predicted_classes = np.argmax(predictions, axis=1)

# Handle labels - they might be 1D or 2D
if len(test_labels.shape) == 1:
    actual_classes = test_labels
else:
    actual_classes = np.argmax(test_labels, axis=1)

# Debug: Check predictions
print(f"\nDebug Info:")
print(f"Predictions shape: {predictions.shape}")
print(f"Predicted classes unique values: {np.unique(predicted_classes)}")
print(f"Actual classes unique values: {np.unique(actual_classes)}")
print(f"Sample predictions (first 5):")
for i in range(min(5, len(predictions))):
    print(f"  Image {i}: Benign={predictions[i][0]:.4f}, Malignant={predictions[i][1]:.4f} -> Predicted: {predicted_classes[i]}, Actual: {actual_classes[i]}")

# Calculate metrics
accuracy = accuracy_score(actual_classes, predicted_classes)
precision = precision_score(actual_classes, predicted_classes, zero_division=0)
recall = recall_score(actual_classes, predicted_classes, zero_division=0)
f1 = f1_score(actual_classes, predicted_classes, zero_division=0)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
