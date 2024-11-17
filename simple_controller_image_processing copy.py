"""camera_pid controller."""
import sys
sys.path.append('/snap/webots/current/usr/share/webots/lib/controller/python')

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage.color as sc
import numpy as np
import skimage
from scipy.ndimage import gaussian_filter as gauss
import math


class ImgProcesing:
    
  def __init__(self):
      self.img = ''
      self.sigma = 5 # 1, 3, 5, 10, 20
      self.kernel_size = 3 
      # self.vertices es una variable que indica la figura para el descarte de informacion de la imagen.
      self.vertices =   np.array([[(0,64),(0,35),(105,35),(128,64)]], dtype=np.int64)

  # Metodo para aplicar desefoque Gaussiani, esta funcion se mandara llamar en el pipeline.
  def gaussBlur(self, img):
      return cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0)
  
  # Metodo para aplicar sobel a la imagen, en los ejes x y y.
  def Sobel(self, img):
      gX = cv2.Sobel(self.gaussBlur(img), ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
      gY = cv2.Sobel(self.gaussBlur(img), ddepth=cv2.CV_64F, dx=0 , dy=1, ksize=3)
      gX = cv2.convertScaleAbs(gX)
      gY = cv2.convertScaleAbs(gY)
      combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
      return combined
  
  #Metodo Laplacciano 
  def Laplaciano(self, img):
      laplacian_img = cv2.Laplacian(img, ddepth=cv2.CV_64F, ksize=3)
      return laplacian_img
  
  # Metdodo Canny recibe como parametro la imagen y retorna la imagen.
  def Canny(self, img):
      canny_img = cv2.Canny(img, threshold1=150, threshold2=250)
      return canny_img
  
  #Metodo para aplicar erosion a la imagen, la intension es volver mas gruesas las lineas.
  def Erosion(self, img):
      kernel = np.ones((3,3),np.uint8)
      erosion = cv2.dilate(img, kernel, 2)
      return erosion
  
  # Metodo para crear la figura que descartara informacion de la imgen final.
  def RoiFilter(self, img):
      roi_img = np.zeros_like(img)
      roi_img = cv2.fillPoly(roi_img, self.vertices, 255)
      roi_mask = cv2.bitwise_and(img, roi_img)
      return roi_mask
  
  #Metodo para mostrar la imagen si ejecutas el codifo solo.
  def displayImage(self, img):
      fig = plt.figure(figsize=(12,10))
      ax = fig.add_subplot(111)
      ax.imshow(img, cmap='gray')

  #Metodo para aplicar un pipeline a la imgane antes de aplicar Hough. 
  def pipeline(self, img):
      gauss_blur = self.gaussBlur(img)
      #gauss_blur = self.gaussBlur(gauss_blur)
      #sobel_filter= self.Sobel(gauss_blur)
      canny_filter = self.Canny(gauss_blur)
      #canny_filter = self.Canny(canny_filter)
      #sobel_filter= self.Sobel(canny_filter)
      #lapacian_filter= self.Laplaciano(gauss_blur)
      #img_mask = self.RoiFilter(canny_filter)
      return canny_filter
  
  # Metodos para aplicar Hough y retornar ya sea lines o img_lines
  def Hough(self, img):
      rho = 1
      theta = np.pi/180
      threshold = 21
      min_line_len = 10
      max_line_gap = 0
      img_maks = self.pipeline(img)
      lines = cv2.HoughLinesP(img_maks, rho, theta, threshold,
                                      np.array([]), minLineLength=min_line_len,
                                      maxLineGap=max_line_gap)
      img_lines = np.zeros((img_maks.shape[0], img_maks.shape[1], 3), dtype=np.uint8)
    
      return lines
      
      #print('lines',lines)
          
  # Metodo para aplicar el add weight y pintar la imagen con las lineas detectadas.
  def addWeighted(self, rgbImg, img_lines):
      alpha = 1
      beta = 1
      gamma = 1
      img_lane_lines = cv2.addWeighted(rgbImg, alpha, img_lines, beta, gamma)
      return img_lane_lines
  
  # Metodo para calcualr el angulo de manejo con la salida de Hough.
  def calculate_steering_angle(self, lines):
  # Calcular el ángulo promedio de las líneas detectadas
      if lines is not None:
          angles = []
          for line in lines:
              x1, y1, x2, y2 = line[0]
              # obtner los angulos con arctan del arreglo de valores obtenidos de Hough
              # el arreglo de lines es un arreglo de puntos en la matriz de la imagen, son
              # 4 elementos que describen el inicio de la line y el final en la imagen de 64x128
              angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
              angles.append(angle)
          # Se usa la mediana de los angulos obtenidos del arcotangente
          average_angle = np.mean(angles)
          return np.round(average_angle)
      else:
          return 0 

        

#Obtener imagen
def get_image(camera, imgPro):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    
    return image

#Procesamiento de imagen aplicando Hough y el pipeline rn la clase ImgProcesing.
def greyscale_cv2(image, imgPro):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    gray_img = imgPro.pipeline(gray_img)
    return gray_img

#Mostrar imagen
def display_image(display, image):

    
    # Imagen a mostrar
    image_rgb = np.dstack((image, image,image,))
    # Mostrar imagen
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#Angulos iniciales y velocidad
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 20

# COnfigurar velocidad final
def set_speed(kmh):
    global speed            #robot.step(50)
#Acctulizar el angulo de manejo
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Revisar limites de angulo
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # Limite del anulo de manejo
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # Actualizar angulo de manejo
    angle = wheel_angle

#Validar el incremento del angulo de manejo.
def change_steer_angle(inc):
    global manual_steering
    # Aplicar incremento 
    new_manual_steering = manual_steering + inc
    # Validar intervalo
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Banderas
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))

# main
def main():
    # Crear intancia Raiz
    robot = Car()
    driver = Driver()

    # Crear instancia de la clase ImgProcessing
    imgPro = ImgProcesing()

    # Obtener intervalos de tiempo .
    timestep = int(robot.getBasicTimeStep())

    # Crear la instancia de la camara
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # Procesando el display
    display_img = Display("display_image")

    #Crear
    keyboard=Keyboard()
    keyboard.enable(timestep)



    while robot.step() != -1:
        # Obtener imagen de la camara
        image = get_image(camera, imgPro)

        # Procesos de la imagen
        # Convertir a escala de grises
        grey_image = greyscale_cv2(image, imgPro)
        # Aplicar el Metodo de Hough con la imagen en escala de grises
        lines = imgPro.Hough(grey_image)
        # Obtener los angulos de manejo con el metdo de obtencion en este caso arctan2
        angle = imgPro.calculate_steering_angle(lines)
        # Imprimir las lineas obtenidas de arctang
        change_steer_angle(angle)
        display_image(display_img, grey_image)
        # Read keyboard
        #filename with timestamp and saved in current directory
        current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        file_name = current_datetime + ".png"
        print("Image taken")
        #camera.saveImage(os.getcwd() + "/" + file_name, 1)
            
        #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()