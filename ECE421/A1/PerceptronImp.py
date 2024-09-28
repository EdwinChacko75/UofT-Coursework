import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris


def fit_perceptron(X_train, y_train, max_epochs=5000):
    """
    This function computes the parameters w of a linear plane which separates
    the input features from the training set into two classes specified by the
    training dataset.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes the value +1 or −1 to represent
        the first class and second class respectively.

    Returns
    -------
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of the line computed by the pocket
        algorithm that best separates the two classes of training data points.
        The dimensions of this vector is (d+1) as the offset term is accounted
        in the computation.
    """
    row, col = X_train.shape
    Ein_best = np.inf

    # Create an array of ones to horizontally stack with X_train
    ones = np.ones((row, 1))

    X_train = np.hstack((ones, X_train)) # Stack hor. ones with x_train

    # Initialize weights to 0
    w = np.zeros(X_train.shape[1]) # Cant use col since it dosent account for the ones added
    w_best = np.copy(w) # Create a deep copy of w

    # Run the Pocket Algorithm for each instance over max_epochs epochs
    for _ in range(max_epochs): 
        # PLA
        for x_i, y_i in zip(X_train, y_train):
            logit = np.dot(w, x_i) # Raw dot product shares sign with post activation value so we can use it

            if logit * y_i <= 0: # If misclassification -> update weights
                w = w + y_i * x_i
        
        # Compute error
        error = errorPer(X_train, y_train, w)
        
        # Update w_best if w(t+1) is a better classifier
        if Ein_best > error:
           Ein_best = error
           w_best = np.copy(w)
        
    return w_best



def errorPer(X_train, y_train, w):
    """
    This function finds the average number of points that are misclassified by
    the plane defined by w.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d+1)
        Represents the matrix of input features where N is the total number of
        training samples and d is the dimension of each input feature vector.
        Note the additional dimension which is for the additional column of ones
        added to the front of the original input.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.

    Returns
    -------
    avgError: float
        The average number of points that are misclassified by the plane
        defined by w.
    """

    N = X_train.shape[0] # Total numnber of training points

    # Prevent division by 0
    if N == 0:
        return N
    
    misclassified = 0 

    for x_i, y_i in zip(X_train, y_train):
        logit = np.dot(x_i, w)

        # if the product is not nonnegative, it is missclassified
        if logit * y_i <= 0:
            misclassified += 1

    # Arithmetic mean
    avg = misclassified/N

    return avg


def pred(x_i, w):
    """
    This function finds finds the prediction by the classifier defined by w.

    Parameters
    ----------
    x_i: numpy.ndarray with shape (d+1,)
        Represents the feature vector of (d+1) dimensions of the ith test
        datapoint.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.

    Returns
    -------
    pred_i: int
        The predicted class.
    """

    pred_i = -1 if np.dot(x_i, w) < 0 else 1
    return pred_i


def confMatrix(X_train, y_train, w):
    """
    This function populates the confusion matrix.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.

    Returns
    -------
    conf_mat: numpy.ndarray with shape (2,2), composed of integer values
        - conf_mat[0, 0]: True Negative
            number of points correctly classified to be class −1.
        - conf_mat[0, 1]: False Positive
            number of points that are in class −1 but are classified to be class
            +1 by the classifier.
        - conf_mat[1, 0]: False Negative
            number of points that are in class +1 but are classified to be class
            −1 by the classifier.
        - conf_mat[1, 1]: True Positive
            number of points correctly classified to be class +1.
    """
    # Add batch dimension
    ones = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((ones, X_train))

    conf_mat = np.zeros((2,2), dtype=np.int32) # Initalize conf matrix to 0

    # For each training instance, add it to the confusion matrix accordingly
    for x_i, y_i in zip(X_train, y_train):
      
      y_pred = pred(x_i, w) # Consider boundary predictions as correctly classified

      if y_pred == y_i:         # if correct prediciton
        if y_i == -1:           # if correct and false (True Negative)
          conf_mat[0, 0] += 1 
        else:                   # if correct and true (True Positive)
          conf_mat[1, 1] += 1
      else:                     # if misclassified
        if y_i == -1:           # if misclassified and false (False Positive)
          conf_mat[0, 1] += 1
        else:                   # if misclassified and true (False Negative)
          conf_mat[1, 0] += 1

    return conf_mat


def test_SciKit(X_train, X_test, y_train, y_test):
    """
    This function uses Perceptron imported from sklearn.linear_model to fit the
    linear classifer using the Perceptron learning algorithm. Then it returns
    the result obtained from the confusion_matrix function imported from
    sklearn.metrics to report the performance of the fitted model.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of
        training samples and d is the dimension of each input feature vector.
    X_test: numpy.ndarray with shape (M,d)
        Represents the matrix of input features where M is the total number of
        testing samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature.
    y_test: numpy.ndarray with shape (M,)
        The ith component represents the output observed in the test set
        for the ith row in X_test matrix.

    Returns
    -------
    conf_mat: numpy.ndarray with shape (2,2), composed of integer values
        - conf_mat[0, 0]: True Negative
            number of points correctly classified to be class −1.
        - conf_mat[0, 1]: False Positive
            number of points that are in class −1 but are classified to be class
            +1 by the classifier.
        - conf_mat[1, 0]: False Negative
            number of points that are in class +1 but are classified to be class
            −1 by the classifier.
        - conf_mat[1, 1]: True Positive
            number of points correctly classified to be class +1.
    """
    # Initialize Perceptron object
    perceptron_sk = Perceptron(
        max_iter=5000,          # Set max iterations
        tol=None,               # Prevent early stopping due to tolerance
        # shuffle=False,          # Default=True. This is what causes different outputs in the conf matrices
                                # The rest of the parameters can be default
        # verbose=1               # This allows verification of full epochs being run
    )

    # Train model
    perceptron_sk.fit(X_train, y_train)

    # Make predictions
    y_pred = perceptron_sk.predict(X_test)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    return conf_matrix


def test_Part1():
    """
    This is the main routine function. It loads IRIS dataset, picks its last
    100 datapoints and split them into train and test set. Then finds and prints
    the confusion matrix from part 1a and 1b.
    """

    # Loading and splitting IRIS dataset into train and test set
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],
                                                        y_train[50:],
                                                        test_size=0.2,
                                                        random_state=42)

    # Set the labels to +1 and -1.
    # The original labels in the IRIS dataset are 1 and 2. We change label 2 to -1.
    y_train[y_train != 1] = -1
    y_test[y_test != 1] = -1

    # Pocket algorithm using Numpy
    w = fit_perceptron(X_train, y_train)

    my_conf_mat = confMatrix(X_test, y_test, w)

    # Pocket algorithm using scikit-learn
    scikit_conf_mat = test_SciKit(X_train, X_test, y_train, y_test)

    # Print the result
    print(f"{12*'-'}Test Result{12*'-'}")
    print("Confusion Matrix from Part 1a is: \n", my_conf_mat)
    print("\nConfusion Matrix from Part 1b is: \n", scikit_conf_mat)


if __name__ == "__main__":
    test_Part1()