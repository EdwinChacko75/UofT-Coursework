import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# Read the image
img = cv2.imread('./image1.png')  # numpy array
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define std for Gaussian noise
variances = [250,500,1000,100000]

# 2x2 grid plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Generate and plot edges
for i, var in enumerate(variances):
    mean = 0
    gaussian_noise = np.random.normal(mean, math.sqrt(var), gray.shape).astype(np.float32)
    noisy_image = cv2.add(gray.astype(np.float32), gaussian_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Apply Canny edge detection
    edges = cv2.Canny(noisy_image, threshold1=75, threshold2=100)

    # Plot the edge-detected image
    ax = axes[i // 2, i % 2]  
    ax.imshow(edges, cmap='gray')
    ax.set_title(f'Variance = {var}')
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f'plots/q5/edges_{variances[0]}_to_{variances[-1]}.png', bbox_inches='tight')
plt.close()