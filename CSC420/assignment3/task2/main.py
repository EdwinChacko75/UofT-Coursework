import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(img_name):
    """Load the given image in grayscale."""
    return cv2.imread(f"./{img_name}.png", cv2.IMREAD_GRAYSCALE)

def compute_gradients(image):
    """Compute Sobel gradients in x and y directions."""
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return Ix, Iy

def compute_second_moment_matrix(Ix, Iy, window_size=3, sigma=1.5):
    """Compute the averaged second-moment matrix components (Sxx, Syy, Sxy)."""
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    # Generate Gaussian kernel
    kernel = cv2.getGaussianKernel(ksize=window_size, sigma=sigma)
    gaussian_window = kernel @ kernel.T  

    # Apply Gaussian filtering
    Sxx = cv2.filter2D(Ixx, -1, gaussian_window)
    Syy = cv2.filter2D(Iyy, -1, gaussian_window)
    Sxy = cv2.filter2D(Ixy, -1, gaussian_window)

    return Sxx, Syy, Sxy
def compute_eigenvalues(Sxx, Syy, Sxy):
    """
    Compute eigenvalues λ1, λ2 for each pixel.
    Clipped for numerical stability.
    """
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy**2
    discriminant = np.clip(trace**2 - 4 * det, a_min=0, a_max=None)

    lambda1 = (trace + np.sqrt(discriminant)) / 2
    lambda2 = (trace - np.sqrt(discriminant)) / 2
    return lambda1, lambda2

def mark_corners(image_gray, lambda1, lambda2, threshold):
    """
    Highlight corners (where min(λ1, λ2) > threshold) in red on the original image.
    Returns a BGR color image with corners marked.
    """
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    corner_mask = (np.minimum(lambda1, lambda2) > threshold)

    image_color[corner_mask] = [0, 0, 255]

    return image_color

def plot_scatter(lambda1_1, lambda2_1,
                 lambda1_2, lambda2_2,
                 save1_path='./assets/sf1_scatter.png',
                 save2_path='./assets/sf2_scatter.png'):
    """Create and save scatter plots of λ1 vs λ2 for both images."""
    plt.figure(figsize=(10, 5))
    plt.scatter(lambda1_1.flatten(), lambda2_1.flatten(), s=1, alpha=0.5, color='blue')
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel(r"$\lambda_2$")
    plt.title(r"Scatter Plot of $\lambda_1$ vs $\lambda_2$ (Image 1)")
    plt.grid(True)
    plt.savefig(save1_path, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.scatter(lambda1_2.flatten(), lambda2_2.flatten(), s=1, alpha=0.5, color='red')
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel(r"$\lambda_2$")
    plt.title(r"Scatter Plot of $\lambda_1$ vs $\lambda_2$ (Image 2)")
    plt.grid(True)
    plt.savefig(save2_path, bbox_inches='tight')
    plt.close()

def main():
    sigma=500

    # sigma_scale = 1/10
    # sigma = sigma*sigma_scale
    # Load images
    img1_gray = load_image('sf1')
    img2_gray = load_image('sf2')

    # Compute gradients
    Ix1, Iy1 = compute_gradients(img1_gray)
    Ix2, Iy2 = compute_gradients(img2_gray)

    # Second moment matrices
    Sxx1, Syy1, Sxy1 = compute_second_moment_matrix(Ix1, Iy1, sigma=sigma)
    Sxx2, Syy2, Sxy2 = compute_second_moment_matrix(Ix2, Iy2, sigma=sigma)

    # Eigenvalues
    lambda1_1, lambda2_1 = compute_eigenvalues(Sxx1, Syy1, Sxy1)
    lambda1_2, lambda2_2 = compute_eigenvalues(Sxx2, Syy2, Sxy2)


    # Create scatter plots
    plot_scatter(lambda1_1, lambda2_1,
        lambda1_2, lambda2_2,
        save1_path=f'./assets/sf1_scatter_{sigma}.png',
        save2_path=f'./assets/sf2_scatter_{sigma}.png')
    
    # Thresholds
    threshold_1 = 3000
    threshold_2 = 20000 

    # Mark corners
    corners_img1 = mark_corners(img1_gray, lambda1_1, lambda2_1, threshold_1)
    corners_img2 = mark_corners(img2_gray, lambda1_2, lambda2_2, threshold_2)

    # Save corner-marked images
    cv2.imwrite(f"./assets/sf1_corners_{sigma}.png", corners_img1)
    cv2.imwrite(f"./assets/sf2_corners_{sigma}.png", corners_img2)


if __name__ == "__main__":
    main()
