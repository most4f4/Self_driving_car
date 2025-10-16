import cv2
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def augment_brightness(image):
    """
    Randomly adjust brightness of the image.
    This simulates different lighting conditions (day, night, shadows).
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Brightness-adjusted image
    """
    # Convert to HSV (Hue, Saturation, Value)
    # V channel controls brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Random brightness factor between 0.4 and 1.2
    # < 1.0 = darker, > 1.0 = brighter
    brightness_scale = random.uniform(0.4, 1.2)

    # Adjust the V (brightness) channel
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_scale, 0, 255).astype(np.uint8)
    
    # Convert back to BGR color space
    augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return augmented


def augment_flip(image, steering_angle):
    """
    Horizontally flip the image and reverse the steering angle.
    This helps balance left/right turn data.
    
    Args:
        image: Input image
        steering_angle: Original steering angle
        
    Returns:
        Flipped image, negated steering angle
    """
    flipped_image = cv2.flip(image, 1)  # Horizontal flip
    flipped_steering = -steering_angle  # Reverse steering angle
    return flipped_image, flipped_steering


def augment_pan(image, steering_angle, pan_range=50):
    """
    Randomly shift (pan) the image horizontally.
    This simulates the car being at different positions on the road.
    
    Args:
        image: Input image
        steering_angle: Original steering angle
        pan_range: Maximum pixels to shift left or right
        
    Returns:
        Panned image, adjusted steering angle
    """
    rows, cols, _ = image.shape

    # Random horizontal shift
    pan_x = random.randint(-pan_range, pan_range)
        
    # Adjust steering angle based on pan
    # If we shift right, we need to steer more left (and vice versa)
    steering_adjustment = pan_x / pan_range * 0.2  # Scale factor of 0.2
    new_steering = steering_angle + steering_adjustment

    # Create translation matrix
    translation_matrix = np.float32([[1, 0, pan_x], [0, 1, 0]])

    # Apply the translation
    panned_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

    return panned_image, new_steering


def augment_zoom(image, steering_angle, zoom_range=0.2):
    """
    Randomly zoom in/out of the image.
    This simulates different distances and speeds.
    
    Args:
        image: Input image
        steering_angle: Original steering angle
        zoom_range: Maximum zoom factor (0.2 = 20% zoom)
        
    Returns:
        Zoomed image, steering angle (unchanged)
    """
    rows, cols, _ = image.shape

    # Random zoom factor between 1-zoom_range and 1+zoom_range
    zoom_factor = random.uniform(1 - zoom_range, 1 + zoom_range)

    # Calculate the zoomed dimensions
    zoomed_rows = int(rows * zoom_factor)
    zoomed_cols = int(cols * zoom_factor)

    # Resize the image
    zoomed_image = cv2.resize(image, (zoomed_cols, zoomed_rows))

    # Crop or pad to original size
    if zoom_factor > 1:  # Zoomed in - crop center
        start_row = (zoomed_rows - rows) // 2
        start_col = (zoomed_cols - cols) // 2
        result  = zoomed_image[start_row:start_row + rows, start_col:start_col + cols]

    else:  # Zoomed out - pad with black borders
        result = np.zeros_like(image)
        start_row = (rows - zoomed_rows) // 2
        start_col = (cols - zoomed_cols) // 2
        result[start_row:start_row + zoomed_rows, start_col:start_col + zoomed_cols] = zoomed_image

    return result, steering_angle


def augment_shadow(image):
    """
    Add random shadow to the image.
    This simulates shadows from trees, buildings, etc.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Image with shadow added
    """
    rows, cols, _ = image.shape

    # Random shadow intensity
    shadow_darkness = np.random.uniform(0.3, 0.7)

        
    # Create a random polygon for shadow
    x1 = np.random.randint(0, cols)
    x2 = np.random.randint(0, cols)

    # Generate a mask for the shadow
    mask = np.zeros((rows, cols), dtype=np.uint8)

    # Define shadow region (vertical stripe)
    if x1 < x2:
        mask[:, x1:x2] = 1
    else:
        mask[:, x2:x1] = 1

    # Convert to Hsv and darken the V channel in the shadow region
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 - shadow_darkness * mask)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    shadowed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return shadowed_image

def augment_image(image, steering_angle, apply_all=False):
    """
    Apply random augmentation techniques to an image.
    
    Args:
        image: Input image
        steering_angle: Original steering angle
        apply_all: If True, apply all augmentations. If False, randomly select.
        
    Returns:
        Augmented image, adjusted steering angle
    """
    augmented_img = image.copy()
    augmented_steering = steering_angle

    if apply_all:
        # Apply all augmentations for demonstration
        augmented_img = augment_brightness(augmented_img)
        augmented_img = augment_shadow(augmented_img)
        augmented_img, augmented_steering = augment_pan(augmented_img, augmented_steering)
        augmented_img, augmented_steering = augment_zoom(augmented_img, augmented_steering)

        # Flip with 50% probability
        if np.random.random() < 0.5:
            augmented_img, augmented_steering = augment_flip(augmented_img, augmented_steering)

    else:
        # Randomly apply augmentations (more realistic for training)
        if random.random() < 0.5:
            augmented_img = augment_brightness(augmented_img)
        if random.random() < 0.3:
            augmented_img = augment_shadow(augmented_img)
        if random.random() < 0.3:
            augmented_img, augmented_steering = augment_zoom(augmented_img, augmented_steering)
        if random.random() < 0.5:
            augmented_img, augmented_steering = augment_pan(augmented_img, augmented_steering)
        if random.random() < 0.5:
            augmented_img, augmented_steering = augment_flip(augmented_img, augmented_steering)

    return augmented_img, augmented_steering

def visualize_augmentations(image_path, steering_angle):
    """
    Visualize different augmentation techniques on a sample image.
    
    Args:
        image_path: Path to image file
        steering_angle: Steering angle for this image
    """

    # Load image
    original = cv2.imread(image_path)

    if original is None:
        print("Could not load image!")
        return
    
    # Create figure with subplots
    plt.figure(figsize=(16, 8))

    # Original
    plt.subplot(2, 4, 1)  # Row 1, Column 1
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Brightness
    bright_img = augment_brightness(original)
    plt.subplot(2, 4, 2)  # Row 1, Column 2
    plt.imshow(cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB))
    plt.title('Brightness Adjusted')
    plt.axis('off')

    # Flip
    flipped_img, flipped_steering = augment_flip(original, steering_angle)
    plt.subplot(2, 4, 3)  # Row 1, Column 3
    plt.imshow(cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Flipped Image\nSteering: {flipped_steering:.2f}')
    plt.axis('off')

    # Shadow
    shadow_img = augment_shadow(original)
    plt.subplot(2, 4, 4)  # Row 1, Column 4
    plt.imshow(cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB))
    plt.title('Shadow Added')
    plt.axis('off')

    # Pan
    panned_img, panned_steering = augment_pan(original, steering_angle)
    plt.subplot(2, 4, 5)  # Row 2, Column 1
    plt.imshow(cv2.cvtColor(panned_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Panned Image\nSteering: {panned_steering:.2f}')
    plt.axis('off')

    # Zoom
    zoomed_img, zoomed_steering = augment_zoom(original, steering_angle)
    plt.subplot(2, 4, 6)  # Row 2, Column 2
    plt.imshow(cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Zoomed Image\nSteering: {zoomed_steering:.2f}')
    plt.axis('off')

    # Combined Augmentations (example 1)
    combined_img, combined_steering = augment_image(original, steering_angle, apply_all=False)
    plt.subplot(2, 4, 7)  # Row 2, Column 3
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Random Mix 1\nSteering: {combined_steering:.2f}')
    plt.axis('off')

    # Combined augmentations (example 2)
    combined_img2, combined_steering2 = augment_image(original, steering_angle, apply_all=False)
    plt.subplot(2, 4, 8)  # Row 2, Column 4
    plt.imshow(cv2.cvtColor(combined_img2, cv2.COLOR_BGR2RGB))
    plt.title(f'Random Mix 2\nSteering: {combined_steering2:.2f}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=300, bbox_inches='tight')
    print("\n✓ Augmentation examples saved as 'augmentation_examples.png'")
    plt.show()


# Test the augmentation functions
if __name__ == "__main__":
    
    print("=" * 60)
    print("DATA AUGMENTATION DEMONSTRATION")
    print("=" * 60)

    # Load driving log CSV
    column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv('../data/driving_log.csv', header=None, names=column_names)

    # Get a sample image with non-zero steering angle
    sample_idx = data[abs(data['steering']) > 0.1].index[0]
    sample_path = data['center'].iloc[sample_idx].strip()
    sample_steering = data['steering'].iloc[sample_idx]

    sample_filename = os.path.basename(sample_path)
    
    print(f"\nSample image: {sample_filename}")
    print(f"Original steering angle: {sample_steering:.4f}")
    
    # Visualize augmentations
    visualize_augmentations(sample_path, sample_steering)
    
    print("\n" + "=" * 60)
    print("✓ Augmentation demonstration complete!")
    print("=" * 60)
