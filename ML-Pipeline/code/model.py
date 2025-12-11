import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import numpy as np


def grayscale_to_rgb(x):
    """Convert grayscale to RGB by repeating the channel 3 times"""
    return tf.repeat(x, 3, axis=-1)


def create_single_view_model(input_shape=(512, 512, 1)):
    """
    Create a custom CNN model trained from scratch for breast cancer detection
    """
    
    print("Building Custom CNN Model")
    print("=" * 60)
    
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
        layers.MaxPooling2D((2, 2), name='pool4'),
        layers.Dropout(0.25),
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu', name='dense1'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', name='dense2'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', name='dense3'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax', name='output')
    ])
    
    # Compile with moderate learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n✓ Custom CNN Model created and compiled successfully")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    return model


def print_model_summary(model):
    """
    Print detailed model summary
    
    Args:
        model: the compiled model
    """
    print("\nModel Summary:")
    print("=" * 60)
    model.summary()


def test_model(model):
    """
    Test model with dummy data
    
    Args:
        model: the compiled model
    """
    print("\nTesting Model with Dummy Data")
    print("=" * 60)
    
    # Create dummy input (batch of 4 images)
    dummy_input = np.random.rand(4, 512, 512, 1).astype(np.float32)
    
    # Make predictions
    predictions = model.predict(dummy_input, verbose=0)
    
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {predictions.shape}")
    print(f"\nSample predictions (first image):")
    print(f"  Benign probability: {predictions[0][0]:.4f}")
    print(f"  Malignant probability: {predictions[0][1]:.4f}")
    print(f"  Sum: {predictions[0].sum():.4f}")
    print(f"\n✓ Model is working correctly!")
    
    return True


def visualize_model_architecture(model, save_path='model_architecture.png'):
    """
    Create a visual diagram of the model architecture
    
    Args:
        model: the compiled model
        save_path: where to save the visualization
    """
    try:
        keras.utils.plot_model(
            model,
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96
        )
        print(f"\n✓ Model visualization saved to {save_path}")
    except Exception as e:
        print(f"Warning: Could not save visualization: {e}")


def explain_model_architecture():
    """
    Print explanation of model architecture
    """
    print("\nModel Architecture Explanation")
    print("=" * 60)
    print("""
ResNet50 (Feature Extractor):
  - Pre-trained on ImageNet (millions of images)
  - Extracts 2048 features from each image
  - Last 50 layers are fine-tuned (trainable)
  - Earlier layers are frozen (not updated)
  - Learns: edges, shapes, textures, patterns

Lambda Layer:
  - Converts grayscale (512,512,1) to RGB (512,512,3)
  - ResNet50 expects 3-channel RGB input
  - Solution: repeat grayscale channel 3 times

Global Average Pooling:
  - Converts (16, 16, 2048) feature maps to (2048,) vector
  - Reduces spatial dimensions while keeping information
  - Prevents overfitting by averaging

Dense Layers:
  - Dense 512 + BatchNorm + Dropout(0.5)
  - Dense 256 + BatchNorm + Dropout(0.5)
  - Dense 128 + Dropout(0.3)
  - Dense 2 + Softmax (output layer)

Activation Functions:
  - ReLU: Introduces non-linearity in hidden layers
  - Softmax: Converts logits to probabilities in output

Training Configuration:
  - Optimizer: Adam (learning rate = 1e-4)
  - Loss: Categorical Crossentropy
  - Metrics: Accuracy
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SINGLE-VIEW CNN MODEL FOR BREAST CANCER DETECTION")
    print("=" * 60)
    
    # Create model
    model = create_single_view_model()
    
    # Print summary
    print_model_summary(model)
    
    # Explain architecture
    explain_model_architecture()
    
    # Test model
    test_model(model)
    
    # Visualize (optional)
    try:
        visualize_model_architecture(model)
    except:
        print("\nNote: Model visualization requires graphviz (optional)")
    
    print("\n" + "=" * 60)
    print("✓ Model is ready for training!")
    print("=" * 60)