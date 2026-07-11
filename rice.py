import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set directory paths
train_dir = 'C:/Users/Asus/Desktop/Project/Agri_World/Agri_World/rice-leaf-diseases-detection/Rice_Leaf_Diease/Rice_Leaf_Diease/train'
validation_dir = 'C:/Users/Asus/Desktop/Project/Agri_World/Agri_World/rice-leaf-diseases-detection/Rice_Leaf_Diease/Rice_Leaf_Diease/test'

# Define the ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the training and validation data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')  # Adjust for your class count
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Save the trained model
model.save('C:/Users/Asus/Desktop/Project/Agri_World/Agri_World/models/DL_models/rice_model.h5')
