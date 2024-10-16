import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

ADJ_PATH = './data/pagerank_adj.mat'
 

def load_adjacency():
    adj_data = loadmat(ADJ_PATH)  
    J = adj_data['J'] 
    return J

def create_link_matrix(J):
    column_sums = np.sum(J, axis=0)
    A = J / column_sums
    return A

def verify_col_sum(A):
    column_sums = np.sum(A, axis=0)
    status_bool = np.allclose(column_sums, 1)
    
    if status_bool:
        print(f"The columns {'' if status_bool else 'do not '}sum to 1.")
    else:
        raise ValueError(f"The columns {'' if status_bool else 'do not '}sum to 1.")

def plot_error(e_k):
    # This function plots the error
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(e_k), marker='o', linestyle='-', color='b')
    plt.title('Error log(e(k+1) Over 10 Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

def power_iteration_algorithm(A):
    # Get random guess x, and nomralize to have l2 norm = 1
    x = np.random.rand(A.shape[0])  
    x = x / np.linalg.norm(x, 2)
    # Error 
    e_k = []
    eigenval = -1

    for k in range(10): 
        y_k_1 = A @ x
        x = y_k_1/ np.linalg.norm(y_k_1, 2)
        eigenval = x @(A @ x)
        # Calculate error
        error_unnorm = A @ x - x
        error = np.linalg.norm(error_unnorm, 2)
        e_k.append(error) # log taken in plot_error

    plot_error(e_k)
    return eigenval, x


def shift_invert_power_iteration(A, sigma=0.99, num_iterations=10):
    n = A.shape[0]
    x = np.random.rand(n)  
    x = x / np.linalg.norm(x, 2)
    
    I = np.eye(n)
    shifted_matrix = A - sigma * I
    inv_shifted_matrix = np.linalg.inv(shifted_matrix)

    e_k = []
    
    for k in range(num_iterations):
        y_k_1 = inv_shifted_matrix @ x
        x = y_k_1 / np.linalg.norm(y_k_1, 2)
        eigenval = x @ (A @ x)
        error = np.linalg.norm(A @ x - x, 2)
        e_k.append(error)
    plot_error(e_k)
    return eigenval, x
def rayleigh_quotient_iteration(A, num_iterations=10):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x, 2)
    
    e_k = []
    sigma = 0.99  # For first two iterations

    for k in range(num_iterations):
        if k > 1:
            sigma = (x @ (A @ x)) / (x @ x)  # Rayleigh quotient for k > 2

        I = np.eye(n)
        shifted_matrix = A - sigma * I
        try:
            inv_shifted_matrix = np.linalg.inv(shifted_matrix)
        except:
            break
        
        y_k_1 = inv_shifted_matrix @ x
        x = y_k_1 / np.linalg.norm(y_k_1, 2)
        eigenval = (x @ (A @ x)) / (x @ x)
        error = np.linalg.norm(A @ x - x, 2)
        e_k.append(error)
        print(k)
    plot_error(e_k)

    return eigenval, x


def rank_pages(eigenvector, num_top=5):
    # Get the indices sorted by eigenvector scores in descending order
    sorted_indices = np.argsort(-eigenvector)
    
    # Top N pages
    top_pages = sorted_indices[:num_top]
    top_scores = eigenvector[top_pages]
    
    # Bottom N pages
    bottom_pages = sorted_indices[-num_top:]
    bottom_scores = eigenvector[bottom_pages]
    
    return list(zip(top_pages + 1, top_scores)), list(zip(bottom_pages + 1, bottom_scores))

def main():
    J = load_adjacency()
    A = create_link_matrix(J)
    verify_col_sum(A)

    # eigenvalue, eigenvector = power_iteration_algorithm(A)

    # print(f"Dominant Eigenvalue: {eigenvalue}")
    # print(f"Corresponding Eigenvector: {eigenvector}")
    # eigenvalue, eigenvector = shift_invert_power_iteration(A)
    # print(f"Dominant Eigenvalue: {eigenvalue}")
    # print(f"Corresponding Eigenvector: {eigenvector}")
    eigenvalue, eigenvector = rayleigh_quotient_iteration(A)
    print(f"Dominant Eigenvalue: {eigenvalue}")
    print(f"Corresponding Eigenvector: {eigenvector}")
    top_pages, bottom_pages = rank_pages(eigenvector)
    print("Top 5 Pages:")
    for page, score in top_pages:
        print(f"Page {page}: {score:.2f}")
    
    print("\nBottom 5 Pages:")
    for page, score in bottom_pages:
        print(f"Page {page}: {score:.2f}")

if __name__ ==  "__main__":
    main()


"""
1a
This function is for part 1a. 
The columns of A must sum to 1 since these values represent the 
distribution of importance from the current page to the next. 
This needs to be a valid probability distribution and therefore
must sum to 1.  
1c
Results consistent with the plots on 7.1. Though the shift inverse
appears more inverse exp than linear but it can be due to few points.
"""