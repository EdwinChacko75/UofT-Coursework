import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_black_square_image(size=100, image_size=500):
    """Generate an image with a black square centered on a white background."""
    image = np.ones((image_size, image_size), dtype=np.float32) * 255
    start = (image_size - size) // 2
    end = start + size
    image[start:end, start:end] = 0
    return image

def apply_log(image, sigma):
    """Apply the Laplacian of Gaussian (LoG) filter."""
    log = cv2.GaussianBlur(image, (0, 0), sigma)
    log = np.float64(log) 
    log = cv2.Laplacian(log, cv2.CV_64F)  
    return np.abs(log).max()  

def find_optimal_sigma(image, sigma_range):
    """Find the optimal sigma that maximizes the LoG response."""
    responses = []
    
    for sigma in sigma_range:
        response = apply_log(image, sigma)
        responses.append(response)
    
    optimal_sigma = sigma_range[np.argmax(responses)]
    
    return optimal_sigma, responses

# Generate black square image
image = generate_black_square_image()

# Define sigma values in log scale
sigma_values = np.logspace(-1, 2, 100)  # Log scale from 0.1 to 100

# Find optimal sigma
optimal_sigma, responses = find_optimal_sigma(image, sigma_values)

# Plot results
plt.figure(figsize=(8, 5))
plt.semilogx(sigma_values, responses, marker='o')
plt.xlabel('Sigma')
plt.ylabel('LoG Response Magnitude')
plt.title(f'Optimal Sigma: {optimal_sigma:.2f}')
plt.grid(True)
plt.savefig('fig.png', dpi=50)

print(f"Optimal sigma for black square detection: {optimal_sigma:.2f}")
