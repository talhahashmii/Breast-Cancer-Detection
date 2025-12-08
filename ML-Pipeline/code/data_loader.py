import os
import cv2
import json
import numpy as np
from pathlib import Path


class DataLoader:
    def __init__(self, dataset_path):
        """
        Initialize the data loader with the path to your dataset
        
        Args:
            dataset_path: String path to the root of your dataset folder (contains train/ and test/)
        """
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        self.image_paths = []
    
    def find_all_images(self, split='train'):
        """
        Walk through the dataset and find all image files with their labels
        
        Args:
            split: 'train' or 'test' - which split to load
        
        Returns:
            Lists of image paths and their labels (benign=0, malignant=1)
        """
        print(f"Searching for images in {split} dataset...")
        
        img_dir = os.path.join(self.dataset_path, split, 'img')
        ann_dir = os.path.join(self.dataset_path, split, 'ann')
        
        if not os.path.exists(img_dir):
            print(f"Error: {img_dir} does not exist")
            return [], []
        
        # Walk through all image files
        for file in os.listdir(img_dir):
            if file.endswith('.png'):
                image_path = os.path.join(img_dir, file)
                
                # Find corresponding annotation file
                ann_file = file + '.json'
                ann_path = os.path.join(ann_dir, ann_file)
                
                if not os.path.exists(ann_path):
                    print(f"Warning: No annotation found for {file}")
                    continue
                
                # Load annotation and extract label
                label = self._get_label_from_annotation(ann_path)
                
                if label is not None:
                    self.image_paths.append(image_path)
                    self.labels.append(label)
        
        print(f"Found {len(self.image_paths)} images")
        print(f"  Benign images: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Malignant images: {sum(1 for l in self.labels if l == 1)}")
        
        return self.image_paths, self.labels
    
    def _get_label_from_annotation(self, ann_path):
        """
        Extract label from JSON annotation file
        
        Args:
            ann_path: Path to the JSON annotation file
        
        Returns:
            0 for benign, 1 for malignant, None if label not found
        """
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # Check tags for benign/malignant label
            if 'tags' in data:
                for tag in data['tags']:
                    tag_name = tag.get('name', '').lower()
                    if tag_name == 'benign':
                        return 0
                    elif tag_name == 'malignant':
                        return 1
                    elif tag_name == 'benign_without_callback':
                        return 0
            
            return None
        except Exception as e:
            print(f"Error reading annotation {ann_path}: {e}")
            return None
    
    def load_image(self, image_path):
        """
        Load a single image from disk
        
        Args:
            image_path: Path to the image file
        
        Returns:
            numpy array of the image in grayscale
        """
        # Read image in grayscale (because mammograms are black and white)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        
        return image
    
    def resize_image(self, image, target_size=512):
        """
        Resize image to target size while maintaining aspect ratio
        
        Args:
            image: numpy array of the image
            target_size: desired width and height (default 512)
        
        Returns:
            resized image as numpy array
        """
        # Get current dimensions
        height, width = image.shape
        
        # Calculate scale factor
        scale = target_size / max(height, width)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize the image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create a canvas of target size filled with zeros (black background)
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # Calculate where to place the resized image on the canvas
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        
        # Place the resized image in the center of the canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    def normalize_image(self, image):
        """
        Normalize image pixel values to range [0, 1]
        
        This helps the neural network learn better because pixel values
        are scaled to a standard range
        
        Args:
            image: numpy array of the image
        
        Returns:
            normalized image as numpy array with float values
        """
        # Convert to float type
        image_float = image.astype(np.float32)
        
        # Normalize to range [0, 1]
        normalized = image_float / 255.0
        
        return normalized
    
    def preprocess_single_image(self, image_path, target_size=512):
        """
        Complete preprocessing pipeline for a single image
        
        Steps:
        1. Load the image
        2. Resize to target size
        3. Normalize pixel values
        
        Args:
            image_path: Path to the image file
            target_size: Target size (default 512)
        
        Returns:
            Preprocessed image as numpy array
        """
        # Step 1: Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Step 2: Resize
        resized = self.resize_image(image, target_size)
        
        # Step 3: Normalize
        normalized = self.normalize_image(resized)
        
        return normalized
