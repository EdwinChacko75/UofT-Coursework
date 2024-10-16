import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import pandas as pd

FACE_PATH = './data/yalefaces.mat'

def load_faces():
    face_data = loadmat(FACE_PATH)
    M = face_data['M'] 
    
    M = M.reshape((32*32, M.shape[2]))  # Shape will be (1024, 2414)

    # Compute the mean face
    mean_face = np.mean(M, axis=1, keepdims=True)

    # Center the data by subtracting the mean face from each image
    M_centered = M - mean_face
    
    # (X_centered * X_centered.T)
    C = M_centered @ M_centered.T / M_centered.shape[1]
    
    return M, C, mean_face
def compute_eigen(C):
    # Compute eigenvalues and eigenvectors of the covariance matrix C
    eigenvalues, eigenvectors = np.linalg.eig(C)
    
    # Sort eigenvalues in decreasing order and reorder eigenvectors
    sorted_indices = np.argsort(-eigenvalues)  # Sort in decreasing order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    return eigenvalues, eigenvectors

def plot_eigenvalues_log(eigenvalues):
    # Plot log of eigenvalues against index
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(eigenvalues))
    plt.title('Log of Eigenvalues vs. Index')
    plt.xlabel('Index')
    plt.ylabel('log(λ_j)')
    plt.grid(True)
    plt.show()
def plot_eigenfaces(eigenvectors, num_faces=10):
    """
    Plot eigenfaces by reshaping the eigenvectors.
    :param eigenvectors: Matrix of eigenvectors, each column is an eigenvector.
    :param num_faces: Number of top and bottom eigenfaces to plot.
    """
    # Plot the largest 10 eigenfaces
    plt.figure(figsize=(10, 5))
    for i in range(num_faces):
        # Reshape the i-th eigenvector to a 32x32 matrix
        eigenface = np.reshape(eigenvectors[:, i], (32, 32))
        plt.subplot(2, num_faces//2, i + 1)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {i+1}')
        plt.axis('off')
    
    plt.suptitle('Top 10 Eigenfaces')
    plt.show()

    # Plot the smallest 10 eigenfaces (from the end of the sorted eigenvectors)
    plt.figure(figsize=(10, 5))
    for i in range(num_faces):
        eigenface = np.reshape(eigenvectors[:, -i-1], (32, 32))
        plt.subplot(2, num_faces//2, i + 1)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {len(eigenvectors[0]) - i}')
        plt.axis('off')
    
    plt.suptitle('Bottom 10 Eigenfaces')
    plt.show()

def l2_projection_and_reconstruction(X_centered, mean_face, eigenvectors, image_indices, j_values):
    num_images = len(image_indices)
    num_j_values = len(j_values)
    
    # Plotting setup, set 7 columns per row
    num_columns = 7
    num_rows = int(np.ceil(num_j_values / num_columns)) * num_images  
    
    plt.figure(figsize=(15, num_rows * 2))  

    for i, img_idx in enumerate(image_indices):
        x_i = X_centered[:, img_idx]  
        
        for j, num_eigenvectors in enumerate(j_values):
            B_j = eigenvectors[:, :num_eigenvectors]  
            
            # l2 projection
            y_i_j = B_j @ (B_j.T @ x_i)  

            reconstructed_image = y_i_j + mean_face.flatten()  # Ensure mean_face is (1024,)

            reconstructed_image = np.reshape(reconstructed_image, (32, 32))
            
            # Plotting
            plt.subplot(num_rows, num_columns, i*num_j_values + j + 1)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title(f'Image {img_idx+1}, j={num_eigenvectors}')
            plt.axis('off')
    
    plt.suptitle('Reconstructed Images with Varying Eigenvectors')
    plt.show()



def compute_projection_coefficients(X_centered, eigenvectors, indices, num_eigenvectors):
    B_25 = eigenvectors[:, :num_eigenvectors]  
    coefficients = []
    
    for idx in indices:
        x_i = X_centered[:, idx]  
        c_i = B_25.T @ x_i  
        coefficients.append(c_i)
    
    return np.array(coefficients)

def compute_pairwise_distances(coefficients):
    num_images = coefficients.shape[0]
    distance_matrix = np.zeros((num_images, num_images))
    
    for i in range(num_images):
        for j in range(num_images):
            distance_matrix[i, j] = euclidean(coefficients[i], coefficients[j])
    
    return distance_matrix
def display_distance_matrix(distance_matrix, labels):
    df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    print(df)

def main():
    M, C, mean_face  = load_faces()
    print(f"Shape of M (flattened images): {M.shape}")
    print(f"Shape of Covariance Matrix C: {C.shape}")
    print(f"Mean face shape: {mean_face.shape}")
    
    eigenvalues, eigenvectors = compute_eigen(C)
    # Plot the log of eigenvalues
    # plot_eigenvalues_log(eigenvalues)
    # plot_eigenfaces(eigenvectors, num_faces=10)

    image_indices = [0, 1075, 2042]  # Subtract 1 for Python
    j_values = [2**i for i in range(1, 11)]  # j 
    
    # l2 projection and reconstruction
    # l2_projection_and_reconstruction(M - mean_face, mean_face, eigenvectors, image_indices, j_values)

    # Python adjusted indices
    I1 = [0, 1, 6]  
    I2 = [2042, 2043, 2044]  

    # Combine I1 and I2 for processing
    indices = I1 + I2
    labels = ["I1-1", "I1-2", "I1-7", "I2-2043", "I2-2044", "I2-2045"]  # Labels for better readability
    
    X_centered = M - mean_face  
    coefficients = compute_projection_coefficients(X_centered, eigenvectors, indices, num_eigenvectors=25)
    
    # Compute pairwise Euclidean distances
    distance_matrix = compute_pairwise_distances(coefficients)
    
    # Display the pairwise distance matrix with labels
    print("Pairwise Euclidean distances between coefficient vectors:")
    display_distance_matrix(distance_matrix, labels)
    
    # Interpret results: smaller distances within same group, larger between groups
    I1_distances = distance_matrix[:3, :3]  # Distances within I1
    I2_distances = distance_matrix[3:, 3:]  # Distances within I2
    cross_distances = distance_matrix[:3, 3:]  # Distances between I1 and I2
    
    print("\nDistances within I1:")
    display_distance_matrix(I1_distances, labels[:3])
    
    print("Distances within I2:")
    display_distance_matrix(I2_distances, labels[3:])
    
    print("Distances between I1 and I2:")
    display_distance_matrix(cross_distances, labels[3:])
    

if __name__ ==  "__main__":
    main()

"""
2a
We can garuntee the eigenvectors are real as they are formed from 
a SPD matrix. It being the product of centered matrices ensures
non negative real eigenvalues.
2b
The top eigenfaces (biggest eigenval) resemble actual faces, cus they 
more noticibale variations in the dataset, like broad facial structures. 
On the other hand, the bottom eigenfaces (smallest eigenvalues) appear 
as noise because they represent minor or subtle variations that contribute 
little to the overall structure of the faces. They interestingly look more like noise
where i would expect the jaw to be, suggesting it dosetn care what is below the
jawline, which lines with face variance.

2d
Pairwise Euclidean distances between coefficient vectors:
                I1-1         I1-2         I1-7      I2-2043      I2-2044      I2-2045
I1-1        0.000000   592.803157   472.412801  1082.612836  1527.827645  1366.002152
I1-2      592.803157     0.000000   392.239403  1403.774856  1902.767217  1612.360178
I1-7      472.412801   392.239403     0.000000  1264.009419  1790.746499  1578.931300
I2-2043  1082.612836  1403.774856  1264.009419     0.000000   758.797143   681.225102
I2-2044  1527.827645  1902.767217  1790.746499   758.797143     0.000000   828.386016
I2-2045  1366.002152  1612.360178  1578.931300   681.225102   828.386016     0.000000

Distances within I1:
            I1-1        I1-2        I1-7
I1-1    0.000000  592.803157  472.412801
I1-2  592.803157    0.000000  392.239403
I1-7  472.412801  392.239403    0.000000

The distances between the same person are a couple 100 but never 1000s. However different 
people are consistently in the 1000s. This indicates that distance between pictures of the 
same person are generally smaller than distances between different people.

This idea can be used to create a facial recognition scheme by setting a threshold, say 1000
for distance and if the distance is less than 1000, the people are classified as the same. If
not, the people are calssified differently. 
"""