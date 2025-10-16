import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the driving log CSV file
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pd.read_csv('../data/driving_log.csv', names=column_names)

print(f"Total images: {len(data)}")
print(f"First image path: {data['center'].iloc[0]}")

def load_image(image_path):
    """
    Load an image from the given path.
    
    Args:
        image_path: Full path to the image file
        
    Returns:
        image: BGR image (OpenCV format)
    """

    # Handle path formatting (in case of Windows paths)
    image_path = image_path.strip()

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return None
    
    # Load image using OpenCV (BGR format)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load image at {image_path}")
        return None

    return image


def preprocess_image(image):
    """
    Apply all preprocessing steps to an image.
    
    Steps:
    1. Crop the road area (remove sky and hood)
    2. Convert to YUV color space
    3. Resize to 200x66 pixels
    4. Apply Gaussian blur (optional)
    5. Normalize pixel values
    
    Args:
        image: Original BGR image
        
    Returns:
        processed_image: Preprocessed image ready for training
    """

    # Step 1: Crop the image
    # According to instructions of Nvidia model, we crop from row 60 to 135
    # This removes the sky (top) and car hood (bottom)
    cropped_image = image[60:135, :, :]

    # Step 2: Convert BGR to YUV color space
    # YUV is better for separating luminance from color information
    yuv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2YUV)

    # Step 3: Resize to 200x66 (width x height)
    # This is the input size expected by the Nvidia model
    resized_image = cv2.resize(yuv_image, (200, 66))

    # Step 4: Apply Gaussian blur
    # This can help reduce noise and improve generalization
    blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)

    # Step 5: Normalize pixel values 
    # Scale pixel values to the range [0, 1]
    # This helps the neural network train better
    normalized = blurred_image / 255.0

    return normalized


def visualize_preprocessing_steps(image_path):
    """
    Visualize each preprocessing step on a sample image.
    
    Args:
        image_path: Path to an image file
    """

    # Load the image
    original = load_image(image_path)

    if original is None:
        print("Could not load image!")
        return
    
    # Create figure with subplots
    plt.figure(figsize=(15, 8))

    # 1. Original image
    plt.subplot(2, 3, 1)  # Row 1, Column 1
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 2. Cropped image
    cropped = original[60:135, :, :]
    plt.subplot(2, 3, 2)  # Row 1, Column 2
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title('Cropped (Road Area)')
    plt.axis('off')

    # 3. YUV color space
    yuv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
    plt.subplot(2, 3, 3)  # Row 1, Column 3
    plt.imshow(cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB))
    plt.title('YUV Color Space')
    plt.axis('off')

    # 4. Resized image
    resized = cv2.resize(yuv_image, (200, 66))
    plt.subplot(2, 3, 4)  # Row 2, Column 1
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_YUV2RGB))
    plt.title('Resized Image')
    plt.axis('off')

    # 5. Gaussian blur
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    plt.subplot(2, 3, 5)  # Row 2, Column 2
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_YUV2RGB))
    plt.title('Gaussian Blurred')
    plt.axis('off')

    # 6. Normalized image 
    normalized = blurred / 255.0
    # For display, convert back to 0-255 range as uint8
    normalized_display = (normalized * 255).astype(np.uint8)
    plt.subplot(2, 3, 6)  # Row 2, Column 3
    plt.imshow(cv2.cvtColor(normalized_display, cv2.COLOR_YUV2RGB))
    plt.title('Normalized Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('preprocessing_steps.png', dpi=300, bbox_inches='tight')
    print("\n✓ Preprocessing visualization saved as 'preprocessing_steps.png'")
    plt.show()


def test_preprocessing():
    """
    Test the preprocessing pipeline on a few sample images.
    """

    # Get first 3 image paths
    sample_paths = data['center'].iloc[:3].values

    for i, path in enumerate(sample_paths):
        print(f"\nProcessing image {i+1}: {os.path.basename(path)}")

        # Load original image
        image = load_image(path)
        if image is None:
            continue

        print(f"  Original shape: {image.shape}")

        # Preprocess image
        processed = preprocess_image(image)
        print(f"  Processed shape: {processed.shape}")
        print(f"  Pixel value range: [{processed.min():.3f}, {processed.max():.3f}]")


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("IMAGE PREPROCESSING DEMONSTRATION")
    print("=" * 60)

    # Test on first image
    first_image_path = data['center'].iloc[0].strip()
    print(f"\nVisualizing preprocessing steps on:\n{first_image_path}\n")
    
    visualize_preprocessing_steps(first_image_path)
    
    # Test preprocessing function
    test_preprocessing()
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing demonstration complete!")
    print("=" * 60)