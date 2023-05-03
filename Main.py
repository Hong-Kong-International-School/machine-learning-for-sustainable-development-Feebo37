
import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator


train_path = "path/to/training/data"
val_path = "path/to/validation/data"

# Define the image dimensions
img_width, img_height = 224, 224

# Define the batch size
batch_size = 32

# Define the image normalizing function
def normalize_img(img):
    return img / 255.0

# Define the data augmentation generator
datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

# Define the training data generator
train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

# Define the validation data generator
validation_generator = datagen.flow_from_directory(
        val_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

# Save the generators for later use
np.save('train_generator.npy', train_generator)
np.save('validation_generator.npy', validation_generator)

