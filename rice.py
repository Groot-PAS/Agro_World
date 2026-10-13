"""
Rice Leaf Disease Classification Training Pipeline

Architecture:
  Custom 2-Convolutional Layer CNN trained from scratch:
  Conv2D(32, 3x3) -> MaxPool2D(2,2) -> Conv2D(64, 3x3) -> MaxPool2D(2,2) ->
  Flatten() -> Dense(128, relu) -> Dropout(0.5) -> Dense(num_classes, softmax)

Evaluation Note:
  This script trains a baseline model for 10 epochs using rescale=1./255.
  In earlier iterations, the 'test' directory was used directly for validation during
  training without a separate final holdout set. For production-grade evaluation,
  use a separate held-out test split, learning rate scheduling, and data augmentation.
"""

import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def train_rice_model(train_dir, validation_dir, output_path, epochs=10, batch_size=32):
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training directory not found at: {train_dir}\n"
            "Please provide a valid path via --train_dir or ensure the rice dataset is present."
        )

    print("=" * 60)
    print("Training Custom Rice Leaf Disease CNN Model")
    print(f"Train directory:      {train_dir}")
    print(f"Validation directory: {validation_dir}")
    print(f"Output model path:    {output_path}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print("=" * 60)

    # Image augmentation & preprocessing (normalized to [0, 1])
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    num_classes = train_data.num_classes
    print(f"Detected {num_classes} classes: {list(train_data.class_indices.keys())}")

    val_data = None
    if validation_dir and os.path.exists(validation_dir):
        val_data = val_datagen.flow_from_directory(
            validation_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )

    # Build CNN Architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    # Save trained model
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.save(output_path)
    print(f"Model saved successfully to: {output_path}")
    return history


def main():
    default_data_dir = os.path.join(BASE_DIR, 'dataset', 'rice_leaf_disease')
    default_train_dir = os.path.join(default_data_dir, 'train')
    default_val_dir = os.path.join(default_data_dir, 'val')
    default_output = os.path.join(BASE_DIR, 'models', 'DL_models', 'rice_model.h5')

    parser = argparse.ArgumentParser(description="Train custom Rice CNN disease classifier")
    parser.add_argument("--train_dir", default=default_train_dir, help="Path to training images directory")
    parser.add_argument("--val_dir", default=default_val_dir, help="Path to validation images directory")
    parser.add_argument("--output", default=default_output, help="Path to save output .h5 model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (baseline: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    train_rice_model(args.train_dir, args.val_dir, args.output, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
