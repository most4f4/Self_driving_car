"""
Self-Driving Car Simulator Test Script

This script connects to the Udacity self-driving car simulator
and uses your trained model to control the car autonomously.

Usage:
    python drive.py model.h5
"""

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import argparse

# Import preprocessing from existing module
from preprocess_images import preprocess_image as preprocess_img


# Initialize SocketIO server and Flask app
sio = socketio.Server(cors_allowed_origins='*')
app = Flask(__name__)

# Global variables
model = None
prev_image_array = None
speed_limit = 15  # Maximum speed in MPH


def preprocess_image(pil_image):
    """
    Preprocess PIL image for model prediction.
    Converts PIL image to OpenCV format then uses existing preprocess function.
    
    Args:
        pil_image: PIL Image object from simulator
        
    Returns:
        Preprocessed numpy array ready for model
    """
    # Convert PIL image (RGB) to numpy array
    img = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format expected by preprocess_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Use the existing preprocessing function
    processed = preprocess_img(img)
    
    return processed


@sio.on('telemetry')
def telemetry(sid, data):
    """
    Handle telemetry data from simulator.
    
    This function is called every time the simulator sends data.
    It receives the current image and other telemetry, then sends
    back the steering angle and throttle commands.
    """
    if data:
        # Current steering angle from simulator
        steering_angle = float(data["steering_angle"])
        
        # Current throttle
        throttle = float(data["throttle"])
        
        # Current speed
        speed = float(data["speed"])
        
        # Current image from center camera
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        
        try:
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Reshape for model input: (1, 66, 200, 3)
            processed_image = np.array([processed_image])
            
            # Predict steering angle
            steering_angle = float(model.predict(processed_image, verbose=0))
            
            # Calculate throttle based on speed
            # Slow down on sharp turns, speed up on straight
            if abs(steering_angle) > 0.3:
                # Sharp turn - slow down
                throttle = 0.1
            elif speed < speed_limit:
                # Below speed limit - accelerate
                throttle = 0.3
            else:
                # At speed limit - maintain
                throttle = 0.1
            
            # Print telemetry for monitoring
            print(f'Steering: {steering_angle:.4f} | Throttle: {throttle:.2f} | Speed: {speed:.1f} MPH')
            
            # Send control commands back to simulator
            send_control(steering_angle, throttle)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            # Send safe defaults on error
            send_control(0, 0)
    else:
        # No data received - likely simulator is in manual mode
        print("No telemetry data - simulator may be in manual mode")
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    """Handle client connection."""
    print("=" * 60)
    print("🚗 CONNECTED TO SIMULATOR")
    print("=" * 60)
    print("Waiting for telemetry data...")
    send_control(0, 0)


@sio.on('disconnect')
def disconnect(sid):
    """Handle client disconnection."""
    print("\n" + "=" * 60)
    print("❌ SIMULATOR DISCONNECTED")
    print("=" * 60)


def send_control(steering_angle, throttle):
    """
    Send control commands to simulator.
    
    Args:
        steering_angle: Predicted steering angle (-1 to 1)
        throttle: Throttle value (0 to 1)
    """
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )


def main(model_path, speed):
    """
    Main function to load model and start server.
    
    Args:
        model_path: Path to trained model file
        speed: Maximum speed limit
    """
    global model, speed_limit
    
    print("=" * 60)
    print("SELF-DRIVING CAR - AUTONOMOUS MODE")
    print("=" * 60)
    
    # Load the trained model
    print(f"\nLoading model from: {model_path}")
    try:
        # Try loading with compile=False to avoid metric deserialization issues
        model = load_model(model_path, compile=False)
        print("✓ Model loaded successfully!")
        
        # Recompile the model with simple loss
        from tensorflow.keras.optimizers import Adam
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        print(f"\nModel summary:")
        model.summary()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Set speed limit
    speed_limit = speed
    print(f"\nSpeed limit set to: {speed_limit} MPH")
    
    # Start the server
    print("\n" + "=" * 60)
    print("Starting server on http://localhost:4567")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Open the simulator")
    print("2. Select 'Autonomous Mode'")
    print("3. Watch your car drive itself!")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Wrap Flask application with socketio's middleware (CORRECT WAY)
    app_wrapped = socketio.Middleware(sio, app)
    
    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app_wrapped)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Self-Driving Car Simulator Driver')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Example: model.h5'
    )
    parser.add_argument(
        '--speed',
        type=int,
        default=15,
        help='Maximum speed limit (default: 15 MPH)'
    )
    
    args = parser.parse_args()
    
    # Run the server
    main(args.model, args.speed)