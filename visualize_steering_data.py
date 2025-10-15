import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the driving log CSV file
data = pd.read_csv('../data/driving_log.csv', header=None)

# The CSV has no headers, so we specify column names
data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

# Display basic information
print("=== DATA SUMMARY ===")
print(f"Total samples: {len(data)}")
print(f"\nFirst few rows:")
print(data.head())
print(f"\nSteering angle statistics:")
print(data['steering'].describe())

# Extract steering angles
steering_angles = data['steering']

# Create the histogram
plt.figure(figsize=(12, 6))

# Plot histogram with 25 bins
n, bins, patches = plt.hist(steering_angles, bins=25, edgecolor='black', alpha=0.7)

# Customize the plot
plt.xlabel('Steering Angle', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('Distribution of Steering Angles in Collected Data', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add vertical line at zero to highlight center
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero steering (straight)')

# Add statistics text box
stats_text = f'Total Samples: {len(steering_angles)}\n'
stats_text += f'Mean: {steering_angles.mean():.4f}\n'
stats_text += f'Std Dev: {steering_angles.std():.4f}\n'
stats_text += f'Min: {steering_angles.min():.4f}\n'
stats_text += f'Max: {steering_angles.max():.4f}'

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=10, family='monospace')

plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('steering_histogram.png', dpi=300, bbox_inches='tight')
print("\n✓ Histogram saved as 'steering_histogram.png'")

# Show the plot
plt.show()

# Additional analysis: Count near-zero steering
near_zero_count = len(steering_angles[abs(steering_angles) < 0.01])
near_zero_percentage = (near_zero_count / len(steering_angles)) * 100

print(f"\n=== BALANCE ANALYSIS ===")
print(f"Near-zero steering (|angle| < 0.01): {near_zero_count} ({near_zero_percentage:.1f}%)")
print(f"Left turns (angle < -0.01): {len(steering_angles[steering_angles < -0.01])}")
print(f"Right turns (angle > 0.01): {len(steering_angles[steering_angles > 0.01])}")

print("\n✓ Visualization complete!")