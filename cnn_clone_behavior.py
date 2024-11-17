import os
import csv
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load data from CSV file
def load_data(data_folder):
    lines = []
    with open(os.path.join(data_folder, 'angle.csv')) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for line in reader:
            #print('Line', line)
            lines.append(line)
    return lines

# Preprocess image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.resize(image, (200, 66))
    image = image / 255.0
    return image

# Load and preprocess images and steering angles
def load_and_preprocess_data(lines, data_folder):
    images = []
    steering_angles = []
    for line in lines:
        source_path = os.path.join(data_folder, line[0])
        image = cv2.imread(source_path)
        image = preprocess_image(image)
        #print('Line', line[1])
        steering_angle = float(line[1])
        images.append(image)
        steering_angles.append(steering_angle)
    return np.array(images), np.array(steering_angles)

# Define the model
def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model

# Load data
data_folder = '/home/mitnik/Documents/MNA/autonomous_driving/proyecto_final/image_bank/'
lines = load_data(data_folder)

# Load and preprocess images and steering angles
images, steering_angles = load_and_preprocess_data(lines, data_folder)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, steering_angles, test_size=0.2, random_state=42)

# Create and compile the model
model = create_model()
model.compile(optimizer=Adam(lr=1e-4), loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('behavioral_cloning_model.h5')
