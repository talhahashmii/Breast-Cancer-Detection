import numpy as np
import os

# Load the preprocessed data
output_dir = r"..\Data\Preprocessed Data"

images = np.load(os.path.join(output_dir, 'preprocessed_images.npy'))
labels = np.load(os.path.join(output_dir, 'preprocessed_labels.npy'))

print("Data Verification Report:")
print("=" * 50)
print(f"Total images processed: {len(images)}")
print(f"Image shape: {images[0].shape}")
print(f"Image dtype: {images.dtype}")
print(f"Pixel value range: [{images.min():.4f}, {images.max():.4f}]")
print(f"Mean pixel value: {images.mean():.4f}")
print(f"Std pixel value: {images.std():.4f}")
print()
print(f"Total labels: {len(labels)}")
print(f"Benign (0): {sum(labels == 0)}")
print(f"Malignant (1): {sum(labels == 1)}")
print()
print("Pixel value range should be [0.0, 1.0] - VERIFIED" if images.min() >= 0 and images.max() <= 1 else "ERROR: Pixel values out of range!")
print("All images should be 512x512 - VERIFIED" if images[0].shape == (512, 512) else "ERROR: Wrong image size!")
