import os
import cv2
import json
import numpy as np
from pathlib import Path


class ROIExtractor:
    def __init__(self, dataset_path):
        """
        Initialize ROI extractor for your dataset
        
        Args:
            dataset_path: String path to the root of your dataset folder (contains train/ and test/)
        """
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        self.image_paths = []
    
    def get_label_from_annotation(self, ann_path):
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
                label = self.get_label_from_annotation(ann_path)
                
                if label is not None:
                    self.image_paths.append(image_path)
                    self.labels.append(label)
        
        print(f"Found {len(self.image_paths)} images")
        print(f"  Benign images: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Malignant images: {sum(1 for l in self.labels if l == 1)}")
        
        return self.image_paths, self.labels
    
    def load_image(self, image_path):
        """
        Load a single image from disk
        
        Args:
            image_path: Path to the image file
        
        Returns:
            numpy array of the image in grayscale
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        
        return image
    
    def extract_roi_with_mask(self, image, ann_path):
        """
        IMPROVED: Extract ROI using the actual annotation mask if available
        
        Args:
            image: numpy array of the image
            ann_path: path to annotation file
        
        Returns:
            roi_image: cropped image containing the lesion area
        """
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            # If we have bitmap data, use it to find the lesion
            if 'objects' in data and len(data['objects']) > 0:
                obj = data['objects'][0]
                
                if 'bitmap' in obj:
                    bitmap_data = obj['bitmap']
                    origin = bitmap_data.get('origin', [0, 0])
                    
                    # Get approximate lesion center
                    x_center = origin[0]
                    y_center = origin[1]
                    
                    # Create ROI around lesion center (e.g., 800x800 window)
                    roi_size = min(image.shape[0], image.shape[1], 800)
                    
                    y_min = max(0, y_center - roi_size // 2)
                    y_max = min(image.shape[0], y_center + roi_size // 2)
                    x_min = max(0, x_center - roi_size // 2)
                    x_max = min(image.shape[1], x_center + roi_size // 2)
                    
                    roi = image[y_min:y_max, x_min:x_max]
                    
                    if roi.size > 0:
                        return roi
        except:
            pass
        
        # Fallback to old method
        return self.extract_roi(image)
    
    def extract_roi(self, image, padding_percent=0.05):
        """
        Extract Region of Interest (lesion area) from the image
        IMPROVED: Less aggressive cropping, better contrast preservation
        
        Args:
            image: numpy array of the image
            padding_percent: percentage of image to add as padding
        
        Returns:
            roi_image: cropped image containing the lesion area
        """
        # Find non-black pixels (breast tissue)
        threshold = 10
        non_black = np.where(image > threshold)
        
        if len(non_black[0]) == 0:
            return image
        
        # Get bounding box of non-black region
        y_min = non_black[0].min()
        y_max = non_black[0].max()
        x_min = non_black[1].min()
        x_max = non_black[1].max()
        
        # LESS aggressive padding
        height = y_max - y_min
        width = x_max - x_min
        padding_y = int(height * padding_percent)
        padding_x = int(width * padding_percent)
        
        y_min = max(0, y_min - padding_y)
        y_max = min(image.shape[0], y_max + padding_y)
        x_min = max(0, x_min - padding_x)
        x_max = min(image.shape[1], x_max + padding_x)
        
        # Crop the image
        roi_image = image[y_min:y_max, x_min:x_max]
        
        return roi_image
    
    def resize_image(self, image, target_size=512):
        """
        Resize image to target size while maintaining aspect ratio
        
        Args:
            image: numpy array of the image
            target_size: desired width and height (default 512)
        
        Returns:
            resized image as numpy array
        """
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
    
    def enhance_contrast(self, image):
        """
        CRITICAL FIX: Apply CLAHE to enhance local contrast
        This will make lesions more visible and increase differences between classes
        
        Args:
            image: numpy array of the image (uint8, range 0-255)
        
        Returns:
            contrast-enhanced image
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def preprocess_with_roi(self, image_path, ann_path=None, target_size=512, keep_uint8=False):
        """
        IMPROVED preprocessing pipeline with contrast enhancement
        
        Steps:
        1. Load the image
        2. Extract ROI (lesion area)
        3. Enhance contrast with CLAHE
        4. Resize to target size
        5. Keep as uint8 [0-255] OR normalize to float [0-1]
        
        Args:
            image_path: Path to the image file
            ann_path: Path to annotation (for better ROI extraction)
            target_size: Target size (default 512)
            keep_uint8: If True, keep range [0-255]. If False, normalize to [0-1]
        
        Returns:
            Preprocessed image as numpy array
        """
        # Step 1: Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Step 2: Extract ROI (using annotation if available)
        if ann_path and os.path.exists(ann_path):
            roi_image = self.extract_roi_with_mask(image, ann_path)
        else:
            roi_image = self.extract_roi(image)
        
        # Step 3: CRITICAL - Enhance contrast BEFORE resizing
        enhanced = self.enhance_contrast(roi_image)
        
        # Step 4: Resize
        resized = self.resize_image(enhanced, target_size)
        
        # Step 5: Normalize or keep as uint8
        if keep_uint8:
            return resized  # Range [0, 255]
        else:
            return resized.astype(np.float32) / 255.0  # Range [0, 1]