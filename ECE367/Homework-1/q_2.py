import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    # Initialize parameters
    n_comps = 30       
    period = 10       
    fundFreq = 1 / period  
    
    time_pos = np.arange(0, 2 * period, 0.0001)
    harmonics = 2 * np.arange(1, n_comps + 1) - 1
    sq_wave = np.floor(0.9 * np.sin(2 * np.pi * fundFreq * time_pos)) + 0.5
    
    B_unnorm = np.sin(2 * np.pi * fundFreq * np.outer(harmonics, time_pos)) / 2
    
    return time_pos, sq_wave, B_unnorm

def plot_sq(time_pos, sq_wave):
    plt.figure(figsize=(10, 4))
    plt.plot(time_pos, sq_wave, label="Square Wave")
    plt.title('Square Wave Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_B(time_pos, B_unnorm):
    colors = ['orange', 'red', 'blue', 'green', 'brown', 'purple', 'black']
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for i in range(6):
        axes[0].plot(time_pos, B_unnorm[i, :], label=f'{i+1}th Basis Vector')
    axes[0].set_title('First 6 Basis Vectors')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()

    axes[1].plot(time_pos, B_unnorm[29, :], label='30th Basis Vector', color='orange')
    axes[1].set_title('30th Basis Vector')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()

    plt.show()

def check_orthogonality(vectors):
    '''
    This fuction checks the orthogonality of the vectors by taking the dot product of each
    vector with every other vector and producing a heatmap to display the results. If the
    vectors are orthogonal, then only the diagonal will be 1, and everything else, 0.
    '''
    dot_products = np.dot(vectors, vectors.T)
    plt.imshow(dot_products, cmap='hot', interpolation='nearest')
    plt.title('Dot Product Matrix (Orthogonality Check)')
    plt.colorbar()
    plt.show()

def l2_normalize_B(B_unnorm):
    norms = np.linalg.norm(B_unnorm, axis=1, keepdims=True)
    B_norm = B_unnorm / norms
    return B_norm

def get_projection_coeffs(B_norm, sq_wave):
    return np.array([np.dot(sq_wave, basis_vector) for basis_vector in B_norm])

def plot_approximations(time_pos, sq_wave, B_norm, proj_coeffs):
    basis_counts = [1, 2, 3, 4, 5, 6, 30]
    fig, axes = plt.subplots(len(basis_counts), 1, figsize=(10, 12), sharex=True)
    
    for idx, count in enumerate(basis_counts):
        approximation = np.zeros_like(sq_wave)
        for i in range(count):
            approximation += proj_coeffs[i] * B_norm[i, :]
        
        axes[idx].plot(time_pos, sq_wave, label="Original Signal", linestyle='dashed')
        axes[idx].plot(time_pos, approximation, label=f"Approximation with {count} Basis Vectors")
        axes[idx].set_ylabel('Amplitude')
        axes[idx].legend()
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def part_c(B_unnorm, sq_wave):
    B_norm = l2_normalize_B(B_unnorm)
    proj_coeffs = get_projection_coeffs(B_norm, sq_wave)

    sorted_indices = np.argsort(np.abs(proj_coeffs))[::-1]
    B_norm_sorted = B_norm[sorted_indices]
    proj_coeffs_sorted = proj_coeffs[sorted_indices]
    return proj_coeffs_sorted, B_norm_sorted

def main():
    time_pos, sq_wave, B_unnorm = generate_data()
    
    check_orthogonality(B_unnorm)
    plot_sq(time_pos, sq_wave)
    plot_B(time_pos, B_unnorm)
    proj_coeffs_sorted, B_norm = part_c(B_unnorm, sq_wave)
    
    plot_approximations(time_pos, sq_wave, B_norm, proj_coeffs_sorted)


if __name__ == "__main__":
    main()

