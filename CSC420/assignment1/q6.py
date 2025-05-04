import cv2
import numpy as np
import matplotlib.pyplot as plt

##############
### STEP 1 ###
##############
def plot_kernel(kernel: np.ndarray, scale: float, input_size: int):
    """
    Plots the gaussian kernel computed from gaussian_blurring(...).
    Args:
        kernel (np.ndarray): The kernel to be plotted.
        scale       (float): The standard deviaton, for the plot title.
        input_size    (int): The kernel size, for the plot title.        
    """
    # init plot vars
    x = np.arange(kernel.shape[0])
    y = np.arange(kernel.shape[1])
    X, Y = np.meshgrid(x, y)

    # 3d fig
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot
    ax.plot_surface(X, Y, kernel, cmap='viridis', edgecolor='k', alpha=0.8)

    # labels
    ax.set_title(rf'3D Gaussian Kernel for $\sigma = {scale}$ and $k = {input_size}$', fontsize=16)
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.set_zlabel('Kernel Value', fontsize=12)

    # set viewing angle
    ax.view_init(elev=30, azim=210)

    # save
    plt.savefig(f'./plots/q6/gaussian_kernel_{scale}_{input_size}.png')
    plt.close()

def gaussian_blurring(input_size: int, scale: float):
    """
    Computes the gaussian kernel given a kernel size and standard deviation.
    Args:
        input_size    (int): The kernel size, for the plot title.      
        scale       (float): The standard deviaton, for the plot title.
    Returns:
        kernel (np.ndarray): Gaussian Kernel of size input_size and standard deviation scale.
    """
    # init vars
    shape = (input_size, input_size)
    center = input_size //2
    variance = scale **2
    coefficent = 1/(2*np.pi* variance)
    
    # define gaussian funcion
    def gaussian(x, y):
        # Make center (0,0)
        x -= center
        y -= center

        return coefficent * np.exp(-(x**2 + y**2)/(2*variance))

    # populate kernel from the defined function
    kernel = np.fromfunction(gaussian, shape, dtype=float)

    return kernel
##################
### END STEP 1 ###
##################

##############
### STEP 2 ###
##############
def gradient(image: np.ndarray):
    """
    Computes the gradient of an image using sobel filters.
    Args:
        image    (np.ndarray): The image whose gradient is being computed.
    Returns:
        gradient (np.ndarray): The gradient of the image.
    """
    # define Sobel filters
    g_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]
    ])
    g_y = g_x.T

    padding_size = (g_x.shape[0] - 1) //2
    h, w = image.shape

    # Pad the image
    image = np.pad(image, padding_size, 'constant', constant_values=0)

    # Intialize gradients for x and y
    grad_x = np.zeros((h,w))
    grad_y = np.zeros((h,w))

    # Convolve
    for i in range(h):
        for j in range(w):
            # Find the patch and do element wise multiplication
            # then sum to get the final value
            patch = image[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(patch * g_x) 
            grad_y[i, j] = np.sum(patch * g_y)
    
    # get image gradient from g_x and g_y
    gradient = np.sqrt(grad_x**2+grad_x**2)

    # plot image
    return gradient

def plot_image(image: np.ndarray, id: int, type: str='gradient'):
    """
    Plots an np.ndarray as an image. Used in both steps 2 and 3.
    """
    plt.imshow(image, cmap='gray')
    plt.title(f"Image {id} {type[0].capitalize()}{type[1:]}")
    plt.axis('off')
    plt.savefig(f'./plots/q6/image_{id}_{type}.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
##################
### END STEP 2 ###
##################

##############
### STEP 3 ###
##############
def threshold_algorithm(gradient: np.ndarray, epsilon: float=0.1):
    """
    Performs the threshold algorithm on an image gradient. Stopping criterion is threshold but
    max_iters is included for safety.
    Args:
        gradient (np.ndarray): The gradient of the image. Computed using gradient(...).
        epsilon (float): The convergence criterion.
    Returns:
        edges    (np.ndarray): The image after performing the algorithm. Edges are more pronounced.
    """
    # initialize variables
    h, w = gradient.shape
    tau = gradient.sum() / (h*w)
    iterations = 0

    # perform updates on tau
    while 1:
        iterations+=1

        # Split into lower and upper classes
        lower_class = gradient[gradient < tau]
        upper_class = gradient[gradient >= tau]

        # Get means
        m_l = np.mean(lower_class) if lower_class.size > 0 else 0
        m_r = np.mean(upper_class) if upper_class.size > 0 else 0

        # Update tau
        tau_prev = tau
        tau = (m_l + m_r) / 2

        # Stopping criterion
        difference = np.abs(tau - tau_prev)
        if difference <= epsilon:
            break
    print(iterations)

    # Update values to 0 or 255 according to tau
    edges = np.where(gradient <  tau, 0, 255)

    return edges
##################
### END STEP 3 ###
##################

def main():
    # Toggle image 1 or 2
    IMG_ID = 1

    # Read and grascale image
    img = cv2.imread(f'./image{IMG_ID}.png')  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### STEP 1 ###
    # Define params
    input_size = 13
    scale = 1

    # Get gaussian kernel 
    kernel = gaussian_blurring(input_size, scale)
    plot_kernel(kernel, scale, input_size)
    ### END STEP 1 ###

    ### STEP 2 ###
    img_gradient = gradient(gray)
    plot_image(img_gradient, id=IMG_ID)
    ### END STEP 2 ###


    ### STEP 3 ###
    edges = threshold_algorithm(img_gradient, epsilon=0.001)
    plot_image(edges, id=IMG_ID, type=f"edges")
    ### END STEP 3 ###

if __name__=="__main__":
    main()