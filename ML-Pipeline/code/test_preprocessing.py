from data_loader import DataLoader

# Path to your dataset
dataset_path = r"..\data\Dataset"

# Create a data loader
loader = DataLoader(dataset_path)

# Find all images in the training set
image_paths, labels = loader.find_all_images(split='train')

# Test on first 10 images
print("\nTesting preprocessing on first 10 images...")

for i in range(min(10, len(image_paths))):
    image_path = image_paths[i]
    label = labels[i]
    
    # Preprocess the image
    preprocessed = loader.preprocess_single_image(image_path)
    
    if preprocessed is not None:
        label_text = "Benign" if label == 0 else "Malignant"
        print(f"Image {i+1}: {label_text} - Shape: {preprocessed.shape}")
