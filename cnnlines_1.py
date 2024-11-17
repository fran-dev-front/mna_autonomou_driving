import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_and_labels(image_folder, label_file, target_size):
    images = []
    labels = []

    with open(label_file, 'r') as f:
        for line in f:
            label = line.strip().split(',')
            filename = label[1]  # Assuming label file has filenames matching with images
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, target_size)  # Resize to target size
                images.append(img)
                labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def custom_data_generator(X, y, batch_size, target_size, augment=True):
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    ) if augment else ImageDataGenerator(rescale=1./255)
    
    num_samples = len(X)
    
    while True:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]
            
            augmented_X = []
            for img in batch_X:
                img = img.reshape((1,) + img.shape)  # Reshape image for data augmentation
                augmented_img = next(data_gen.flow(img, batch_size=1))[0]  # Apply augmentation
                augmented_X.append(augmented_img)
            
            yield np.array(augmented_X), np.array(batch_y)

# Example usage:
image_folder = '/home/mitnik/Documents/MNA/autonomous_driving/proyecto_final/image_bank'  # Replace with your folder path
label_file = '/home/mitnik/Documents/MNA/autonomous_driving/proyecto_final/angle.csv'  # Replace with your label file path
target_size = (128, 128)  # Adjust based on your requirement
batch_size = 32

X_train, y_train = load_images_and_labels(image_folder, label_file, target_size)

train_generator = custom_data_generator(X_train, y_train, batch_size, target_size)

# Define a simple model for demonstration
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Single output for regression

model.compile(optimizer=Adam(), loss='mean_squared_error')

# Assuming you have 1000 samples in your dataset
steps_per_epoch = len(X_train) // batch_size

model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=20)
