"""camera_pid controller."""
import sys

sys.path.append('/usr/local/webots/lib/controller/python')

from controller import Display, Keyboard, Robot, Camera, Radar
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import csv
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

def preprocess_image(image, target_size=(64, 128)):
    """
    Preprocess the image: resize and normalize.
    
    Parameters:
    image_path (str): Path to the image file.
    target_size (tuple): Desired size (width, height).
    
    Returns:
    preprocessed_image (numpy array): Preprocessed image ready for model prediction.
    """
    # Load the image
    if image is None:
        raise ValueError(f"Image not found at {image}")
    
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalizar los valores de los píxeles
    img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=-1)  # Añadir el canal de color (resultando en (64, 128, 1))
    img_array = np.expand_dims(img_array, axis=0) 
    
    return image

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    
    return image

#Image processing
def greyscale_cv2(image, ):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

#Display image 
def display_image(display, image):

    
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0


          #robot.step(50)
#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))
#Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

# main
def main():

    model = load_model('/home/mitnik/Documents/MNA/autonomous_driving/proyecto_final/steering_angle_model_town.keras')
    # Create the Robot ins  tance.
    robot = Car()
    driver = Driver()
    speed = 5
    #create img processing
    #imgPro = ImgProcesing()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    radar = robot.getDevice("radar")

    radar.enable(timestep)
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)



    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)
        grey_image = greyscale_cv2(image)
        display_image(display_img, grey_image)


        img_array = np.array(grey_image)
        img_array = img_array / 255.0  # Normalizar los valores de los píxeles
        img_array.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=-1)  # Añadir el canal de color (resultando en (64, 128, 1))
        img_array = np.expand_dims(img_array, axis=0) 

        predicted_angle = model.predict(img_array, verbose = 0)[0][0]

        print('Angle', predicted_angle)
        
        set_steering_angle(predicted_angle)


        # Get radar objects
        number_of_targets = radar.getNumberOfTargets()

        # Print the number of detected targets.
        #print(f'Number of targets detected: {number_of_targets}')

        if number_of_targets > 1:
            print('stop')
            driver.setCruisingSpeed(0)
        else:
            driver.setCruisingSpeed(15)
       
        
        display_image(display_img, grey_image)
        # Read keyboard
        key=keyboard.getKey()
        if key == keyboard.UP: #up
            set_speed(speed + 5.0)
            print("up")
        elif key == keyboard.DOWN: #down
            set_speed(speed - 5.0)
            print("down")
        elif key == keyboard.RIGHT: #right
            change_steer_angle(+1)
            print("right")
        elif key == keyboard.LEFT: #left
            change_steer_angle(-1)
            print("left")
        
        #elif key == ord('A'):
            #filename with timestamp and saved in current directory
        # current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        # file_name = current_datetime + ".png"
        # #print("Image taken")
        # camera.saveImage(os.getcwd() + "/proyecto_final/image_bank/" + file_name, 1)
        # with open('angle.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([file_name, angle])
            
        #update angle and speed
        driver.setSteeringAngle(angle)
        
        


if __name__ == "__main__":
    
    
    main()
    