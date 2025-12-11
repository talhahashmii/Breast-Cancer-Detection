# Single-View CNN Model for Breast Cancer Detection

## Overview

This directory contains the implementation of a single-view CNN (Convolutional Neural Network) for classifying mammogram images as benign or malignant.

## Architecture

### Model: ResNet50 + Custom Classification Head

```
Input (512×512×1)
    ↓
ResNet50 (Pre-trained, Frozen)
    - 175 layers
    - 23.5M parameters
    - Extracts 2048 features
    ↓
Global Average Pooling
    - Converts (16, 16, 2048) → (2048,)
    ↓
Dense 256 + ReLU
    - 524,544 parameters
    ↓
Dropout (0.5)
    - Prevents overfitting
    ↓
Dense 128 + ReLU
    - 32,896 parameters
    ↓
Dropout (0.5)
    - Prevents overfitting
    ↓
Dense 2 + Softmax
    - Output: [P(benign), P(malignant)]
```

## Files

### Core Model Files

| File | Purpose |
|------|---------|
| `model.py` | Model architecture definition and creation |
| `test_model.py` | Test script to verify model works |
| `train_model.py` | Training script with data loading and evaluation |

### Data Preprocessing Files

| File | Purpose |
|------|---------|
| `roi_extractor.py` | Extract Region of Interest from images |
| `test_roi_extraction.py` | Test ROI extraction |
| `batch_roi_preprocessing.py` | Batch process all images |
| `verify_roi_pipeline.py` | Verify preprocessed data |

## Quick Start

### 1. Test the Model

```bash
python test_model.py
```

This will:
- Load ResNet50
- Create the model architecture
- Test with dummy data
- Print model summary

Expected output:
```
Building Single-View CNN Model
✓ ResNet50 loaded and frozen
✓ Model created and compiled successfully
✓ Total parameters: 23,845,410
✓ Trainable parameters: 557,698
✓ Model is working correctly!
```

### 2. Preprocess Images with ROI Extraction

```bash
python batch_roi_preprocessing.py
```

This will:
- Load all images from `../data/Dataset/train/img/`
- Extract ROI (lesion area) from each image
- Resize to 512×512
- Normalize pixel values to [0, 1]
- Save to `../Data/Preprocessed Data/ROI/`

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Load preprocessed images
- Split into train/validation/test sets
- Create and compile the model
- Train for 20 epochs
- Evaluate on test set
- Save trained model as `breast_cancer_model.h5`
- Plot training history

## Model Details

### Why ResNet50?

- **Pre-trained**: Already learned from millions of ImageNet images
- **Transfer Learning**: Saves training time and improves performance
- **Proven**: Industry standard for image classification
- **Efficient**: Only need to train top layers (~557K parameters)

### Why Frozen Weights?

- ResNet50 already learned general image features (edges, shapes, textures)
- We only need to teach it medical-specific patterns
- Freezing prevents overfitting on our smaller dataset
- Reduces computational cost during training

### Dropout Layers

- Randomly disable 50% of neurons during training
- Forces network to learn robust features
- Prevents overfitting
- Improves generalization to unseen data

### Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Optimizer | Adam | Adaptive learning rate, works well in practice |
| Learning Rate | 1e-4 | Small rate for fine-tuning pre-trained model |
| Loss | Categorical Crossentropy | Standard for multi-class classification |
| Batch Size | 32 | Balance between speed and stability |
| Epochs | 20 | Enough to converge without overfitting |

## Data Flow

```
Raw Images (4651×4651)
    ↓
ROI Extraction (crop to lesion area)
    ↓
Resize to 512×512
    ↓
Normalize to [0, 1]
    ↓
Preprocessed Images (512×512×1)
    ↓
Train/Val/Test Split (70/10/20)
    ↓
Model Training
    ↓
Trained Model
```

## Expected Performance

### Model Capacity

- **Total Parameters**: 23,845,410
- **Trainable Parameters**: 557,698 (2.3%)
- **Frozen Parameters**: 23,287,712 (97.7%)

### Training Metrics

- **Input Shape**: (batch_size, 512, 512, 1)
- **Output Shape**: (batch_size, 2)
- **Output Interpretation**:
  - Output[0] = Probability of benign
  - Output[1] = Probability of malignant
  - Sum = 1.0

## Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution**: Install TensorFlow
```bash
pip install tensorflow>=2.13.0
```

### Issue: "Could not find preprocessed data"

**Solution**: Run ROI preprocessing first
```bash
python batch_roi_preprocessing.py
```

### Issue: Out of memory during training

**Solution**: Reduce batch size in `train_model.py`
```python
history = train_model(
    model, X_train, y_train, X_val, y_val,
    epochs=20,
    batch_size=16  # Reduce from 32
)
```

### Issue: Model not improving

**Solution**: Try different learning rates or epochs
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## Next Steps

After training:

1. **Evaluate on Test Set**: Check accuracy on unseen data
2. **Visualize Predictions**: See what the model predicts
3. **Analyze Errors**: Understand misclassifications
4. **Dual-View Model**: Combine CC and MLO views
5. **Deploy**: Integrate into backend API

## References

- ResNet50: He et al., "Deep Residual Learning for Image Recognition" (2015)
- Transfer Learning: Yosinski et al., "How transferable are features in deep neural networks?" (2014)
- Breast Cancer Detection: Various papers on mammography classification

## Author

Breast Cancer Detection Project - FYP

## License

Educational Use Only
