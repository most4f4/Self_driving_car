import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Import from our existing files
from batch_generator import prepare_data, batch_generator


def build_nvidia_model():
    """
    Build the Nvidia self-driving car CNN model.
    
    Architecture (based on Figure 7):
    - Input: 66x200x3 (YUV image)
    - Conv Layer 1: 24 filters, 5x5 kernel, 2x2 stride
    - Conv Layer 2: 36 filters, 5x5 kernel, 2x2 stride
    - Conv Layer 3: 48 filters, 5x5 kernel, 2x2 stride
    - Conv Layer 4: 64 filters, 3x3 kernel, 1x1 stride
    - Conv Layer 5: 64 filters, 3x3 kernel, 1x1 stride
    - Flatten
    - Fully Connected: 1164 neurons
    - Fully Connected: 100 neurons
    - Fully Connected: 50 neurons
    - Fully Connected: 10 neurons
    - Output: 1 neuron (steering angle)
    
    Returns:
        model: Compiled Keras model
    """
    model = Sequential()
    
    # Convolutional layers with ReLU activation
    # Layer 1: 24 feature maps, 5x5 kernel, 2x2 stride
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)))
    
    # Layer 2: 36 feature maps, 5x5 kernel, 2x2 stride
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    
    # Layer 3: 48 feature maps, 5x5 kernel, 2x2 stride
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    
    # Layer 4: 64 feature maps, 3x3 kernel, 1x1 stride
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    # Layer 5: 64 feature maps, 3x3 kernel, 1x1 stride
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    # Flatten the output from convolutional layers
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting
    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='relu'))
    
    # Output layer - single neuron for steering angle (regression)
    model.add(Dense(1))
    
    return model


def train_model(csv_path='../data/driving_log.csv', 
                balance=True,
                batch_size=32, 
                epochs=30,
                learning_rate=0.001,
                model_save_path='model.h5'):
    """
    Train the self-driving car model.
    
    Args:
        csv_path: Path to driving_log.csv
        balance: Whether to balance the dataset
        batch_size: Number of samples per batch
        epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        model_save_path: Path to save the trained model
        
    Returns:
        model: Trained Keras model
        history: Training history
    """
    print("=" * 60)
    print("TRAINING SELF-DRIVING CAR MODEL")
    print("=" * 60)
    
    # Step 1: Prepare data
    print("\n[1/5] Preparing data...")
    train_data, val_data = prepare_data(csv_path, balance=balance)
    
    # Step 2: Create batch generators
    print("\n[2/5] Creating batch generators...")
    train_generator = batch_generator(train_data, batch_size=batch_size, is_training=True)
    val_generator = batch_generator(val_data, batch_size=batch_size, is_training=False)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_data) // batch_size
    validation_steps = len(val_data) // batch_size
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Step 3: Build model
    print("\n[3/5] Building Nvidia CNN model...")
    model = build_nvidia_model()
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Step 4: Compile model
    print("\n[4/5] Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )
    
    # Step 5: Set up callbacks
    print("\n[5/5] Setting up callbacks...")
    
    # Save best model based on validation loss
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    # Stop training if validation loss doesn't improve
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    
    # Train the model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {model_save_path}")
    
    return model, history


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot and save training history graphs.
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid()


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to: {save_path}")
    plt.show()


def evaluate_model(model, val_data, batch_size=32):
    """
    Evaluate the trained model on validation data.
    
    Args:
        model: Trained Keras model
        val_data: Validation DataFrame
        batch_size: Batch size for evaluation
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    val_generator = batch_generator(val_data, batch_size=batch_size, is_training=False)
    validation_steps = len(val_data) // batch_size
    
    results = model.evaluate(val_generator, steps=validation_steps, verbose=1)
    
    print(f"\nValidation Loss (MSE): {results[0]:.6f}")
    print(f"Validation MAE: {results[1]:.6f}")
    print("=" * 60)


# Main execution
if __name__ == "__main__":
    # Training configuration
    CONFIG = {
        'csv_path': '../data/driving_log.csv',
        'balance': True,
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 0.001,
        'model_save_path': 'model.keras'
    }
    
    print("Training Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Train the model
    model, history = train_model(**CONFIG)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on validation set
    _, val_data = prepare_data(CONFIG['csv_path'], balance=CONFIG['balance'])
    evaluate_model(model, val_data, batch_size=CONFIG['batch_size'])
    
    print("\n" + "=" * 60)
    print("✓ ALL DONE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check 'training_history.png' to see how training went")
    print("2. Use 'model.h5' for testing in the simulator")
    print("3. Run the simulator in Autonomous Mode to test!")