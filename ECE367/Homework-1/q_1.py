from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.preprocessing import normalize as norm
from sklearn.feature_extraction.text import TfidfTransformer

NORMALIZE = False
TFIDF = True
MAT_PATH = './data/wordVecV.mat'
E_TITLE = "Heatmap of Pairwise Euclidean Distances"
A_TITLE = "Heatmap of Pairwise Cosine Distances (Degrees)"

def load_data(mat_path, normalize=False, tfidf=False):
    data = loadmat(mat_path)
    V = data['V']
    V_T = V.T
    if normalize:
        V_T = norm(V_T, axis=0, norm='l1') # (see discussion)
    if tfidf:
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(V_T)
        V_T = tfidf_matrix.toarray()

    print(V_T.shape)  
    return V_T

def compute_distance(matrix, metric):
    distance = pdist(matrix, metric=metric)
    dist_matrix = squareform(distance)
    dist_matrix = np.triu(dist_matrix)
    if metric == 'cosine':
        dist_matrix = np.arccos(1 - dist_matrix)
        dist_matrix = np.degrees(dist_matrix)
    return dist_matrix

def custom_log_scale(matrix, max_angle=90):
    non_zero_mask = matrix != 0
    scaled = np.zeros_like(matrix)

    scaled[non_zero_mask] = np.log1p(max_angle - matrix[non_zero_mask])
    scaled = (scaled - np.min(scaled)) / (np.max(scaled) - np.min(scaled))
    
    return scaled

def plot_heatmap(matrix, title, use_log_scale=False):
    plt.figure(figsize=(10, 8))

    if use_log_scale:
        matrix_scaled = custom_log_scale(matrix)
        vmin, vmax = 0, 1
        ax = sns.heatmap(matrix_scaled, annot=matrix, fmt=".2f", cmap="viridis", 
                         cbar=True, vmin=vmin, vmax=vmax, square=True)
    else:
        ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True)
    
    plt.title(title)
    new_labels = list(range(1, len(matrix) + 1))
    ax.set_xticklabels(new_labels)
    ax.set_yticklabels(new_labels)
    plt.show()

def main():
    V_T = load_data(MAT_PATH, normalize=NORMALIZE, tfidf=TFIDF)
    euclidean_distance_matrix = compute_distance(V_T, 'euclidean')
    angle_distance_matrix = compute_distance(V_T, 'cosine')
    
    plot_heatmap(euclidean_distance_matrix, E_TITLE)
    plot_heatmap(angle_distance_matrix, A_TITLE, use_log_scale=NORMALIZE)

if __name__ == "__main__":
    main()

### DISCUSSION ###
'''
1a) NORMALIZE = False in line 9, TFIDF = False in line 10
From the produced heatmaps, it is clear that:
nearest euclidean distance = 24.72 between the 7th and 8th documents
nearest angle distance     = 30.42 degrees between the 9th and 10th documents

it is beacuse when we measure E distance, we measure the number of common words between
the documents. 
when measuing the angle distance, we measure the compare how many times each word appears
in the two texts. the criteria is analgous to the difference in number of times word wi is
used in both texts. 

this explains why the two distances differ, they measure different distances.

1b) NORMALIZE = True in line 9
From the produced heatmaps, it is clear that:
nearest euclidean distance = 12.23 between the 5th and 8th documents
nearest angle distance     = 86.52 degrees between the 9th and 10th documents

there have been noticable differences. 
the euclidean distance is now lowest between the 5th and 8th documents. adding l1 normalization,
added proportion. Indead of the number of common words, we count the proportion of common words
eliminating differences in document length due to not all words being used in all documents. 
the 5th and 8th documents have a greater frequency of common words.

similarly the angle distance switches from measureing how many times each word is used to the proportional
use of each word comapred to the size of the text. this accounts for varying size of text. however, it is interesting
to note that the 9th and 10th documents are still the most similar however all the documents have 
distances nearly orthogonal. this sugests that the texts are highly dissimilar or that l1 normalization is not ideal
or that angle distance is not the best measure of similarity between these texts.

we apply normalization to account for varying length of text that will skew word count to similarity. this removes
bias toward larger documents.
Note the sklearn documentation (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html)
states that the normalize() funcion perfroms l1 nomalization as we learned.

1c) NORMALIZE = False in line 9, TFIDF = True in line 10
using TFIDF, the 9th and 10th documents are the closest in euclidean distance.

1d)
the inverse document frequency (IDF) allows a more nuanced perspective by scaling down words that are necessary for
writing sentences and relatively scaling up words that are less common. Words like 'the', 'a', or 'it' are extremely 
common and almost impossible to remove from a text. These words dont provide any insight into the similarity of two 
texts so removing them allows greater consideration for more context specific words that are relavent to the topic of 
the text. Essentially it filters out words that are likely to occur in any given text and emphazize words that are not 
common to all texts so that the words less used but likely more meaningful are not overshone.

Geometrically speaking, IDF scales doument vector dimensions that correspond to common words to be scaled down and dimensions
corresponding to less common words to be scaled up, modifying the geometry of the document space. IDF makes common terms cause 
document vectors to align less by underweighing these terms and 'pushing apart' the document vectors, and vice versa for less 
common words.
'''