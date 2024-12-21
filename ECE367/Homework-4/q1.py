from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import yaml
MAT_PATH = './data/wordVecV.mat'
TITLES_PATH = './data/wordVecTitles.txt'
def load_data(mat_path):
    data = loadmat(mat_path)
    V = data['V'] 

    # Convert to binary and normalize
    M = np.where(V>0, 1.0, 0.0)
    norm =  np.linalg.norm(M, axis=0, keepdims=True)
    M = M/norm
    # Get titles array
    with open(TITLES_PATH, 'r') as file:
        titles = file.readlines()

    titles = [line.strip() for line in titles]
    return M, titles

def top_k_singular_values(S, k=10):
    print(f"Top {k} singular values:")
    for i in range(1, k+1):
        print(f"{i}: {S[i-1]:.4f}")

def plot_heatmap(matrix, title):
    plt.figure(figsize=(10, 8))


    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True)
    
    plt.title(title)
    new_labels = list(range(1, len(matrix) + 1))
    ax.set_xticklabels(new_labels)
    ax.set_yticklabels(new_labels)
    plt.show()

def similarity(M, U, k=9):

    projections = U[:,:k].T @ M # Shape: (k, m)
    
    norms = np.linalg.norm(projections, axis=0,keepdims=True) # (1, m)
    
    projections /= norms
    sim = projections.T @ projections

    # plot_heatmap(sim, f"Most similar documents for k = {k}")
    np.fill_diagonal(sim, 0.0)
    # Get max indices
    sim = np.triu(sim)
    x = np.max(sim, axis=1, keepdims=True)
    y = np.max(sim, axis=0, keepdims=True)
    
    return np.argmax(x), np.argmax(y)

def similaritiy_for_all_k(TITLES, M, U):
    
    for i in range(8, 0, -1):
        t1, t2 = similarity(M, U, i)

        print(f"For k = {i}, the two most similar documents are {TITLES[t1]} and {TITLES[t2]}")
    

def main():
    M, TITLES = load_data(MAT_PATH)
    U, S, Vt = np.linalg.svd(M)

    # Part c:
    # top_k_singular_values(S)

    # Part d:
    t1, t2 = similarity(M, U, k=9)
    print(f"For k = {9}, the two most similar documents are {TITLES[t1]} and {TITLES[t2]}")

    # Part e:
    similaritiy_for_all_k(TITLES, M, U)
    

if __name__ == "__main__":
    main()

### DISCUSSION ###
"""
Part c - List Top 10 Singular Values in order:
Top 10 singular values:
1: 1.5366
2: 1.0192
3: 0.9587
4: 0.9539
5: 0.9413
6: 0.9289
7: 0.8977
8: 0.8919
9: 0.8687
10: 0.8161

Part d - Titles of the Two Most Similar Documents:
For k = 9, the two most similar documents are Barack Obama and George W. Bush
For k = 8, the two most similar documents are Barack Obama and George W. Bush
For k = 7, the two most similar documents are Barack Obama and George W. Bush
For k = 6, the two most similar documents are Barack Obama and George W. Bush
For k = 5, the two most similar documents are Barack Obama and George W. Bush
For k = 4, the two most similar documents are Barack Obama and George W. Bush
For k = 3, the two most similar documents are Barack Obama and George W. Bush
For k = 2, the two most similar documents are B. J. Cole and John Holland (composer)
For k = 1, the two most similar documents are B. J. Cole and Mary J. Blige

What is the lowest k that does not change your answer for part (d)? 
k = 3 is the lowest k that does not change my answer for part (d).

What is the pair of most similar documents for k - 1?
The pair of most similar documents for k - 1 (2) are B. J. Cole and John Holland (composer)

"""