import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import socketio
import eventlet
import cv2
import base64
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask
from io import BytesIO
from PIL import Image
from preprocess_images import preprocess_image


sio = socketio.Server() # create a Socket.IO server instance
app = Flask(__name__) # create a Flask application instance
maxSpeed = 10
model = load_model('model.keras') # load the trained model


def preProcess(img):
    """Convert PIL RGB image to BGR and preprocess"""
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return preprocess_image(img)


@sio.on('telemetry')
def telemetry(sid, data):
    img = Image.open(BytesIO(base64.b64decode(data['image'])))
    img = np.asarray(img)
    img = preProcess(img)
    img = np.array([img])
    speed = float(data['speed'])
    steering_angle = model.predict(img)[0][0]
    throttle = 1.0 - speed / maxSpeed
    print(f'Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}')
    send_control(steering_angle, throttle)



@sio.on('connect')
def connect(sid, environ):
    print('Connected:', sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
      
