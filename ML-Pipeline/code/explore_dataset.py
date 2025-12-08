import os

# Path to your dataset
dataset_path = r"..\data\Dataset"

# List all top-level folders
print("Top-level folders in dataset:")
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        print(f"  - {folder}")

# Count total images
total_images = 0
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.png'):
            total_images += 1

print(f"\nTotal image files found: {total_images}")

# Count images by split
for split in ['train', 'test']:
    split_path = os.path.join(dataset_path, split, 'img')
    if os.path.exists(split_path):
        count = len([f for f in os.listdir(split_path) if f.endswith('.png')])
        print(f"  {split}: {count} images")
