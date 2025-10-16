import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Import from existing files
from data_augmentation import augment_image
from preprocess_images import load_image, preprocess_image


def batch_generator(data, batch_size=32, is_training=True):
    """
    Generate batches of images and steering angles for training/validation.
    
    This is a Python generator that yields batches infinitely.
    During training, it applies data augmentation.
    
    Args:
        data: Pandas DataFrame with 'center' and 'steering' columns
        batch_size: Number of samples per batch
        is_training: If True, apply augmentation. If False, no augmentation.
        
    Yields:
        X_batch: Numpy array of shape (batch_size, 66, 200, 3) - preprocessed images
        y_batch: Numpy array of shape (batch_size,) - steering angles
    """

    num_samples = len(data)

    while True:  # Loop forever so the generator never terminates

        # Shuffle data at the start of each epoch
        # This prevents the model from learning the order of data
        shuffled_data = data.sample(frac=1).reset_index(drop=True)

        for offset in range(0, num_samples, batch_size):
            # Get the batch slice
            batch_data = shuffled_data[offset:offset + batch_size] # First batch : 0:32 , second batch : 32:64

            images = []
            steering_angles = []
            
            # Loop through each row in the batch
            for idx, row in batch_data.iterrows():
                # Get image path and steering angle
                image_path = row['center']
                steering = row['steering']

                # Load image
                image = load_image(image_path)

                if image is None:
                    continue  # Skip if image loading failed

                # Apply augmentation if in training mode
                if is_training:
                    # Random augmentation with 80% probability
                    if np.random.rand() < 0.8:
                        image, steering = augment_image(image, steering, apply_all=False)

                # Preprocess image
                processed_image = preprocess_image(image)

                # Add to batch 
                images.append(processed_image)
                steering_angles.append(steering)

            # Convert to numpy arrays
            X_batch = np.array(images)
            y_batch = np.array(steering_angles)

            yield X_batch, y_batch # Yield keyword to make this a generator


def balance_dataset(data, threshold=0.01, keep_prob=0.2):
    """
    Balance the dataset by removing some near-zero steering samples.
    
    Args:
        data: Pandas DataFrame with steering column
        threshold: Steering angle threshold to consider as "straight"
        keep_prob: Probability of keeping a near-zero steering sample
        
    Returns:
        Balanced DataFrame
    """

    print(f"\n=== Balancing Dataset ===")
    print(f"Original size: {len(data)}")

    # Separate straight driving from turns
    straight_data = data[np.abs(data['steering']) < threshold]
    turn_data = data[np.abs(data['steering']) >= threshold]

    print(f"Straight driving samples: {len(straight_data)} ({len(straight_data)/len(data)*100:.1f}%)")
    print(f"Turn samples: {len(turn_data)} ({len(turn_data)/len(data)*100:.1f}%)")

    # Keep only a fraction of straight samples
    straight_kept = straight_data.sample(frac=keep_prob, random_state=42)

    # Combine the balanced dataset
    balanced_data = pd.concat([straight_kept, turn_data], ignore_index=True)
    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)  # Shuffle

    print(f"Balanced size: {len(balanced_data)}")
    print(f"Reduction: {(1 - len(balanced_data)/len(data))*100:.1f}%")


    return balanced_data


def prepare_data(csv_path, balance=True, test_size=0.2, random_state=42):
    """
    Load and prepare data for training.
    
    Args:
        csv_path: Path to driving_log.csv
        balance: Whether to balance the dataset
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        train_data, val_data: Training and validation DataFrames
    """

    # Load the CSV file
    column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(csv_path, names=column_names)

    print(f"Total samples loaded: {len(data)}")

    # Keep only center images and steering angles
    data = data[['center', 'steering']]


    # Balance the dataset if requested
    if balance:
        data = balance_dataset(data, threshold=0.01, keep_prob=0.2)

    # Split into training and validation sets
    train_data, val_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print(f"\nTraining samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data



def test_batch_generator():
    """
    Test the batch generator and visualize a sample batch.
    """
    print("=" * 60)
    print("BATCH GENERATOR TEST")
    print("=" * 60)

    # Prepare data (adjust path to match your structure)
    train_data, val_data = prepare_data('../data/driving_log.csv', balance=True)

    # Create a generator for training data
    batch_size = 8
    train_generator = batch_generator(train_data, batch_size=batch_size, is_training=True)
    val_generator = batch_generator(val_data, batch_size=batch_size, is_training=False)

    print(f"\nGenerating a training batch of size {batch_size}...")

    # Get a batch
    X_batch, y_batch = next(train_generator)


    print(f"X_batch shape: {X_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}")
    print(f"Pixel value range: [{X_batch.min():.3f}, {X_batch.max():.3f}]")
    print(f"Steering angles: {y_batch}")


    # Visualize the batch
    plt.figure(figsize=(16, 8))
    for i in range(batch_size):
        # Convert from YUV to RGB for display
        display_img = X_batch[i]
        display_img = (display_img * 255).astype(np.uint8)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_YUV2RGB)

        plt.subplot(2, 4, i + 1)
        plt.imshow(display_img)
        plt.title(f"Steering: {y_batch[i]:.3f}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_batch.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sample batch saved as 'sample_batch.png'")
    plt.show()

    # Test multiple batches
    print("\nGenerating 3 batches to test consistency...")
    for i in range(3):
        X, y = next(train_generator)
        print(f"Batch {i+1}: X shape = {X.shape}, y shape = {y.shape}")
    
    print("\n" + "=" * 60)
    print("✓ Batch generator test complete!")
    print("=" * 60)
   

if __name__ == "__main__":
    test_batch_generator()







