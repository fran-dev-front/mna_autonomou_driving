import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
     
#https://python-socketio.readthedocs.io/en/latest/server.html#eventlet
sio = socketio.Server(async_mode='eventlet')

speed_limit = 10

# mismo preprocesamiento de imágenes que en la CNN
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
     
#este evento sirve para obtener las imágenes y la velocidad del simulador
@sio.on('telemetry')
def telemetry(sid, data):
    # se lee la velocidad
    speed = float(data['speed'])
    # se lee la imagen
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    # se le pide al modelo que determine el ángulo de dirección a partir de la imagen
    steering_angle = float(model.predict(image))
    # se calcula la aceleración
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    # se envía al simulador el ángulo y la aceleración
    send_control(steering_angle, throttle)
     
     
#al iniciarse la conexión, el ángulo y la aceleración son cero     
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
     
# función que permite enviarle al simulador la dirección y la aceleración    
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        })
     
     
if __name__ == '__main__':
    model = load_model('/home/mitnik/Documents/MNA/autonomous_driving/proyecto_final/behavioral_cloning_model.h5')
    app = socketio.WSGIApp(sio)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)