import matplotlib.pyplot as plt
from data_loader import DataLoader

# Path to your dataset
dataset_path = r"..\data\Dataset"

# Create a data loader
loader = DataLoader(dataset_path)

# Find all images
image_paths, labels = loader.find_all_images(split='train')

# Show 4 sample images - 2 benign, 2 malignant
benign_indices = [i for i, l in enumerate(labels) if l == 0][:2]
malignant_indices = [i for i, l in enumerate(labels) if l == 1][:2]

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Show benign images
for idx, plot_idx in enumerate(benign_indices):
    image_path = image_paths[plot_idx]
    preprocessed = loader.preprocess_single_image(image_path)
    
    axes[0, idx].imshow(preprocessed, cmap='gray')
    axes[0, idx].set_title('Benign Lesion')
    axes[0, idx].axis('off')

# Show malignant images
for idx, plot_idx in enumerate(malignant_indices):
    image_path = image_paths[plot_idx]
    preprocessed = loader.preprocess_single_image(image_path)
    
    axes[1, idx].imshow(preprocessed, cmap='gray')
    axes[1, idx].set_title('Malignant Lesion')
    axes[1, idx].axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=150)
plt.show()

print("Sample images saved as 'sample_images.png'")
