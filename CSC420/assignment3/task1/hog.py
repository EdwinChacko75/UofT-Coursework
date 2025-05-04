import cv2
import numpy as np
import matplotlib.pyplot as plt
DPI= 100
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if "flash" in image_path:
    #     h, w = image.shape
    #     new_height = 500
    #     new_width = int((w / h) * new_height)  
        
    #     image = cv2.resize(image, (new_width, new_height))
        
    #     cropped_image = image[100:, :]
    #     cv2.imwrite(image_path, cropped_image)
    #     raise ValueError
    #     return cropped_image
    return image

def compute_image_gradient(image, kernel_size=3, threshold=10):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)  
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)  

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)  

    direction[direction < 0] += np.pi

    magnitude[magnitude < threshold] = 0  
    
    return magnitude, direction


def get_crop(img, tau):
    height, width = img.shape

    m = height // tau 
    n = width // tau   

    return m * tau,  n * tau


def compute_hog_for_grid(magnitude, direction, m, n, tau=8):
    bin_edges = np.deg2rad([-15, 15, 45, 75, 105, 135, 165])
    num_bins = len(bin_edges) - 1

    reshaped_magnitude = magnitude.reshape(m, tau, n, tau).transpose(0, 2, 1, 3)
    reshaped_direction = direction.reshape(m, tau, n, tau).transpose(0, 2, 1, 3)

    bin_masks = np.array([
        (reshaped_direction >= bin_edges[k]) & (reshaped_direction < bin_edges[k + 1])
        for k in range(num_bins)
    ])

    hog_magnitude = np.sum(reshaped_magnitude[None, :, :, :, :] * bin_masks, axis=(-1, -2))
    hog_count = np.sum(bin_masks, axis=(-1, -2))

    hog_magnitude = hog_magnitude.transpose(1, 2, 0)
    hog_count = hog_count.transpose(1, 2, 0)
    return hog_magnitude/np.max(hog_magnitude), hog_count

def plot_hog_grid(image, hog_magnitude, m, n, scale, tau=8, save_path='./assets/hog_mag.png'):
    bin_edges = np.deg2rad([-15, 15, 45, 75, 105, 135, 165])
    num_bins = len(bin_edges) - 1

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    x_centers = np.arange(n) * tau + tau // 2 
    y_centers = np.arange(m) * tau + tau // 2    
    X, Y = np.meshgrid(x_centers, y_centers)
    # hog_magnitude = np.power(hog_magnitude, 0.9)

    for k in range(num_bins):
        angle = (bin_edges[k] + bin_edges[k + 1] + np.pi) / 2
        dx = np.cos(angle) * hog_magnitude[:, :, k] 
        dy = np.sin(angle) * hog_magnitude[:, :, k] 

        ax.quiver(
            X.flatten(), Y.flatten(), dx.flatten(), dy.flatten(),
            angles='xy', 
            scale_units='xy', 
            scale=scale, 
            color='red',
            headwidth=3,
            headlength=3,
            # width=0.008
        )
    plt.title("HOG Visualization (Weighted by Magnitude)")
    plt.axis("off")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', pad_inches=0.1)

def plot_hog_grid_counts(image, hog_counts, m, n, scale, tau=8, save_path='./assets/hog_counts.png'):
    bin_edges = np.deg2rad([-15, 15, 45, 75, 105, 135, 165])
    num_bins = len(bin_edges) - 1

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    x_centers = np.arange(n) * tau + tau // 2
    y_centers = np.arange(m) * tau + tau // 2
    X, Y = np.meshgrid(x_centers, y_centers)

    for k in range(num_bins):
        angle = (bin_edges[k] + bin_edges[k + 1] + np.pi) / 2
        dx = np.cos(angle) * hog_counts[:, :, k]
        dy = np.sin(angle) * hog_counts[:, :, k]

        ax.quiver(
            X.flatten(), Y.flatten(), dx.flatten(), dy.flatten(),
            angles='xy', scale_units='xy', scale=scale, color='red',
            headwidth=3, headlength=3
        )
    plt.title("HOG Visualization (Count-Based)")
    plt.axis("off")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', pad_inches=0.1)

def normalize_hog(hog, block_size=2, eps=0.001):
    m, n, num_bins = hog.shape
    out_m = m - block_size + 1
    out_n = n - block_size + 1

    s0, s1, s2 = hog.strides

    shape = (out_m, out_n, block_size, block_size, num_bins)
    strides = (s0, s1, s0, s1, s2)
    blocks = np.lib.stride_tricks.as_strided(hog, shape=shape, strides=strides)

    descriptors = blocks.reshape(out_m, out_n, -1)

    norms = np.sqrt(np.sum(descriptors ** 2, axis=-1, keepdims=True) + eps ** 2)

    normalized_descriptors = descriptors / norms

    return normalized_descriptors

def write_norm(hog_norm, save_path=f"./assets/random.txt"):
    flattened = hog_norm.flatten()
    flattened_str = ' '.join(map(str, flattened))

    with open(save_path, "w") as file:
        file.write(flattened_str)

def plot_normalized_hog(image, normalized_hog, block_size=2, tau=8, scale=0.5, save_path=None):
    m, n, _ = normalized_hog.shape

    bin_edges = np.deg2rad([-15, 15, 45, 75, 105, 135, 165])
    num_bins = len(bin_edges) - 1

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    x_centers = np.arange(n) * tau + (tau * block_size // 2)  
    y_centers = np.arange(m) * tau + (tau * block_size // 2)
    X, Y = np.meshgrid(x_centers, y_centers)

    for k in range(num_bins):
        angle = (bin_edges[k] + bin_edges[k + 1] + np.pi) / 2  
        dx = np.cos(angle) * normalized_hog[:, :, k]  
        dy = np.sin(angle) * normalized_hog[:, :, k]

        ax.quiver(
            X.flatten(), Y.flatten(), dx.flatten(), dy.flatten(),
            angles='xy', scale_units='xy', scale=scale, color='red',
            headwidth=3, headlength=3
        )

    plt.title("Normalized HOG Visualization")
    plt.axis("off")

    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', pad_inches=0.1)

def normalize_hog_vectorized(hog, block_size=2, eps=0.001):
    """
    Applies L2 normalization on HOG features using a sliding window block of 2×2 cells.
    
    Parameters:
        hog (numpy.ndarray): The input HOG array of shape (m, n, 6) where 6 is the number of orientation bins.
        block_size (int): The size of the block (default is 2x2).
        eps (float): Small constant to prevent division by zero.
    
    Returns:
        numpy.ndarray: Normalized HOG descriptor of shape ((m-1), (n-1), 24).
    """
    m, n, num_bins = hog.shape  # Original HOG shape
    out_m = m - block_size + 1  # New height after block-wise extraction
    out_n = n - block_size + 1  # New width after block-wise extraction

    # Use stride tricks to extract overlapping 2x2 blocks
    s0, s1, s2 = hog.strides
    shape = (out_m, out_n, block_size, block_size, num_bins)
    strides = (s0, s1, s0, s1, s2)
    blocks = np.lib.stride_tricks.as_strided(hog, shape=shape, strides=strides)

    # Reshape into vectors of shape (out_m, out_n, 24)
    descriptors = blocks.reshape(out_m, out_n, -1)

    # Compute L2 norm for each descriptor
    norms = np.sqrt(np.sum(descriptors ** 2, axis=-1, keepdims=True) + eps ** 2)

    # Normalize each descriptor
    normalized_descriptors = descriptors / norms

    return normalized_descriptors  # Shape (m-1, n-1, 24)

def main():
    ###############################
    # Only thing you need to change
    img_name = 'flash'
    ###############################

    img = f"./{img_name}.png"

    img = load_image(img)

    # Image 1 and 2, and both flash
    tau = 15
    threshold = 10
    scale_mag = 0.03
    scale_cnt =4


    magnitude, direction = compute_image_gradient(img, threshold=threshold)

    h, w = get_crop(img=img, tau=tau)
    m, n = h//tau, w//tau

    magnitude = magnitude[:h, :w].copy()
    direction = direction[:h, :w].copy()
    
    hog_magnititude, hog_counts = compute_hog_for_grid(magnitude, direction, m, n , tau=tau)

    plot_hog_grid(img, hog_magnititude, m, n ,scale_mag, tau=tau, save_path=f'./assets/{img_name}_hog_mag.png')

    plot_hog_grid_counts(img, hog_counts, m, n ,scale_cnt, tau=tau, save_path=f'./assets/{img_name}_hog_cnt.png')


    hog_norm = normalize_hog(hog_magnititude)
    plot_normalized_hog(img, hog_norm, block_size=2, tau=tau, scale=scale_mag, save_path=f'./assets/{img_name}_norm.png')

    write_norm(hog_norm, save_path=f"./assets/{img_name}.txt")
    
if __name__ == "__main__":
    main()