import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Lambda
from tensorflow.keras.optimizers import Adam

# Load data from CSV file
def load_data(data_folder):
    lines = []
    with open(os.path.join(data_folder, 'angle.csv')) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for line in reader:
            lines.append(line)
    return lines

# Preprocess image
def preprocess_image(image):
    # Convert the image from RGB to YUV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Resize the image to the input size of the model
    image = cv2.resize(image, (128, 64))
    
    # Normalize the image to have pixel values between 0 and 1
    image = image / 255.0
    
    return image

# Data generator
def data_generator(lines, data_folder, batch_size=32):
    num_samples = len(lines)
    while True:  # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = lines[offset:offset+batch_size]
            
            images = []
            steering_angles = []
            for line in batch_lines:
                source_path = os.path.join(data_folder+'/image_bank', line[0])
                image = cv2.imread(source_path)
                image = preprocess_image(image)
                steering_angle = float(line[1])  # Assuming steering angle is in the 4th column
                images.append(image)
                steering_angles.append(steering_angle)
            
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield (X_train, y_train)

# Define the model
def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=(128, 64, 3)))
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
data_folder = '/home/mitnik/Documents/MNA/autonomous_driving/proyecto_final'
lines = load_data(data_folder)

# Split data into training and validation sets
train_lines, val_lines = train_test_split(lines, test_size=0.2, random_state=42)

# Create data generators
batch_size = 32
train_generator = data_generator(train_lines, data_folder, batch_size=batch_size)
val_generator = data_generator(val_lines, data_folder, batch_size=batch_size)

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='mse')

# Train the model using the generator
model.fit(
    train_generator,
    steps_per_epoch=len(train_lines) // batch_size,
    validation_data=val_generator,
    validation_steps=len(val_lines) // batch_size,
    epochs=10
)

# Save the model
model.save('behavioral_cloning_model.h5')
