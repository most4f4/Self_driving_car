"""
Self-Driving Car - Autonomous Mode Driver
Usage: python drive.py model.keras
"""

import os
import sys
import cv2
import base64
import socketio
import eventlet
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask
from io import BytesIO
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Initialize
sio = socketio.Server()
app = Flask(__name__)
model = None
speed_limit = 15


def preprocess_image(img):
    """
    Preprocess image for model prediction.
    Same as training preprocessing.
    """
    # Crop road area
    img = img[60:135, :, :]
    # Convert to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Apply Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Resize to 200x66
    img = cv2.resize(img, (200, 66))
    # Normalize to [0, 1]
    img = img / 255.0
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    """Handle telemetry data from simulator."""
    print(f"DEBUG: Telemetry received! Data type: {type(data)}, Data: {data is not None}")
    
    if data:
        # Get current speed
        speed = float(data['speed'])
        
        # Get image from simulator
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        
        # Preprocess
        image = preprocess_image(image)
        image = np.array([image])
        
        # Predict steering angle
        steering = float(model.predict(image, verbose=0))
        
        # Calculate throttle
        throttle = 1.0 - speed / speed_limit
        
        # Print telemetry
        print(f'Speed: {speed:.1f} | Steering: {steering:.4f} | Throttle: {throttle:.2f}')
        
        # Send control
        send_control(steering, throttle)
    else:
        print("DEBUG: Telemetry data is None or empty")


@sio.on('connect')
def connect(sid, environ):
    """Handle connection."""
    print('=' * 60)
    print('🚗 CONNECTED TO SIMULATOR')
    print('=' * 60)
    send_control(0, 0)


def send_control(steering, throttle):
    """Send control commands to simulator."""
    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle)
    })


if __name__ == '__main__':
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python drive.py <model_file>")
        print("Example: python drive.py model.keras")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Check if custom speed provided
    if len(sys.argv) >= 3:
        speed_limit = int(sys.argv[2])
    
    print('=' * 60)
    print('SELF-DRIVING CAR - AUTONOMOUS MODE')
    print('=' * 60)
    print(f'\nLoading model: {model_path}')
    
    # Load model
    try:
        model = load_model(model_path, compile=False)
        print('✓ Model loaded successfully!')
    except Exception as e:
        print(f'✗ Error loading model: {e}')
        sys.exit(1)
    
    print(f'Speed limit: {speed_limit} MPH')
    print('\n' + '=' * 60)
    print('Starting server on http://localhost:4567')
    print('=' * 60)
    print('\nWaiting for simulator connection...\n')
    
    # Start server
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)