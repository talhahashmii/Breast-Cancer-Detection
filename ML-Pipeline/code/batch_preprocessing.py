import numpy as np
import os
from data_loader import DataLoader

# Path to your dataset
dataset_path = r"..\data\Dataset"

# Create output directory
output_dir = r"..\Data\Preprocessed Data"
os.makedirs(output_dir, exist_ok=True)

# Create a data loader
loader = DataLoader(dataset_path)

# Find all images
image_paths, labels = loader.find_all_images(split='train')

# Process all images
print(f"\nProcessing {len(image_paths)} images...")

preprocessed_images = []
preprocessed_labels = []
errors = 0

for i in range(len(image_paths)):
    image_path = image_paths[i]
    label = labels[i]
    
    # Preprocess the image
    preprocessed = loader.preprocess_single_image(image_path)
    
    if preprocessed is not None:
        preprocessed_images.append(preprocessed)
        preprocessed_labels.append(label)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1} images...")
    else:
        errors += 1
        print(f"Error processing image {i + 1}")

print(f"\nSuccessfully processed: {len(preprocessed_images)} images")
print(f"Failed: {errors} images")
print(f"Benign: {sum(1 for l in preprocessed_labels if l == 0)}")
print(f"Malignant: {sum(1 for l in preprocessed_labels if l == 1)}")

# Save the preprocessed data
preprocessed_images = np.array(preprocessed_images)
preprocessed_labels = np.array(preprocessed_labels)

np.save(os.path.join(output_dir, 'preprocessed_images.npy'), preprocessed_images)
np.save(os.path.join(output_dir, 'preprocessed_labels.npy'), preprocessed_labels)

print(f"\nData saved to {output_dir}/")
print(f"Images shape: {preprocessed_images.shape}")
print(f"Labels shape: {preprocessed_labels.shape}")
