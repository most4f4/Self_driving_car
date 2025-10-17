"""
Self-Driving Car - Autonomous Mode Driver

This script connects to the Udacity self-driving car simulator
and uses your trained model to control the car autonomously.

Usage:
    python drive.py model.keras [max_speed]
    
Example:
    python drive.py model.keras 15
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
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


# Initialize SocketIO server and Flask app
sio = socketio.Server()
app = Flask(__name__)

# Global variables
model = None
max_speed = 15
frame_count = 0


def preprocess(img):
    """
    Preprocess PIL RGB image for model prediction.
    
    Args:
        img: PIL Image object from simulator
        
    Returns:
        Preprocessed numpy array ready for model
    """
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return preprocess_image(img)


@sio.on('telemetry')
def telemetry(sid, data):
    """
    Handle telemetry data from simulator.
    
    Receives image and vehicle data, predicts steering angle,
    and sends control commands back to simulator.
    """
    global frame_count
    
    if data:
        try:
            # Extract data from simulator
            speed = float(data['speed'])
            
            # Decode and preprocess image
            image = Image.open(BytesIO(base64.b64decode(data['image'])))
            image = np.asarray(image)
            image = preprocess(image)
            image = np.array([image])

            # Predict steering angle
            steering_angle = float(model.predict(image, verbose=0)[0][0])
            
            # Calculate throttle based on speed and steering
            # Slow down on sharp turns, speed up on straight sections
            if abs(steering_angle) > 0.3:
                throttle = 0.1  # Sharp turn - slow down
            elif speed < max_speed:
                throttle = min(1.0 - speed / max_speed, 0.3)  # Accelerate smoothly
            else:
                throttle = 0.0  # At max speed
            
            # Display telemetry every 10 frames to reduce console spam
            frame_count += 1
            if frame_count % 10 == 0:
                print(f'Speed: {speed:5.1f} MPH | Steering: {steering_angle:7.4f} | Throttle: {throttle:.2f}')
            
            # Send control commands
            send_control(steering_angle, throttle)
            
        except Exception as e:
            print(f"Error processing telemetry: {e}")
            send_control(0, 0)


@sio.on('connect')
def connect(sid, environ):
    """Handle client connection from simulator."""
    print('=' * 60)
    print(f'🚗 CONNECTED TO SIMULATOR')
    print(f'   Session ID: {sid}')
    print('=' * 60)
    send_control(0, 0)


@sio.on('disconnect')
def disconnect(sid):
    """Handle client disconnection."""
    print('\n' + '=' * 60)
    print('❌ SIMULATOR DISCONNECTED')
    print('=' * 60)


def send_control(steering_angle, throttle):
    """
    Send control commands to simulator.
    
    Args:
        steering_angle: Steering angle (-1 to 1)
        throttle: Throttle value (0 to 1)
    """
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })


def main():
    """Main function to initialize and start the server."""
    global model, max_speed
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python drive.py <model_file> [max_speed]")
        print("Example: python drive.py model.keras 15")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        max_speed = int(sys.argv[2])
    
    # Display startup info
    print('=' * 60)
    print('SELF-DRIVING CAR - AUTONOMOUS MODE')
    print('=' * 60)
    print(f'\n📂 Loading model: {model_path}')
    
    # Load the trained model
    try:
        model = load_model(model_path, compile=False)
        print('✅ Model loaded successfully!')
    except Exception as e:
        print(f'❌ Error loading model: {e}')
        sys.exit(1)
    
    print(f'🚗 Max speed: {max_speed} MPH')
    print('\n' + '=' * 60)
    print('🌐 Starting server on http://localhost:4567')
    print('=' * 60)
    print('\n📋 Instructions:')
    print('   1. Open the simulator')
    print('   2. Select "Autonomous Mode"')
    print('   3. Choose your track')
    print('   4. Watch your car drive!')
    print('\n⌨️  Press Ctrl+C to stop the server')
    print('=' * 60 + '\n')
    
    # Start the server
    try:
        app_wrapped = socketio.Middleware(sio, app)
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app_wrapped)
    except KeyboardInterrupt:
        print('\n\n' + '=' * 60)
        print('🛑 Server stopped by user')
        print('=' * 60)


if __name__ == '__main__':
    main()