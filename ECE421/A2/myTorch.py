import re
import numpy as np
import random
import collections
from numpy.random import beta
import util

# YOU ARE NOT ALLOWED TO USE sklearn or Pytorch in this assignment


class Optimizer:

    def __init__(
        self, name, lr=0.001, gama=0.9, beta_m=0.9, beta_v=0.999, epsilon=1e-8
    ):
        # self.lr will be set as the learning rate that we use upon creating the object, i.e., lr
        # e.g., by creating an object with Optimizer("sgd", lr=0.0001), the self.lr will be set as 0.0001
        self.lr = lr

        # Based on the name used for creating an Optimizer object,
        # we set the self.optimize to be the desiarable method.
        if name == "sgd":
            self.optimize = self.sgd
        elif name == "heavyball_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.heavyball_momentum
        elif name == "nestrov_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.nestrov_momentum
        elif name == "adam":
            # setting beta_m, beta_v, and epsilon
            # (read the handout to see what these parametrs are)
            self.beta_m = beta_m
            self.beta_v = beta_v
            self.epsilon = epsilon

            # setting the initial first momentum of the gradient
            # (read the handout for more info)
            self.v = 0

            # setting the initial second momentum of the gradient
            # (read the handout for more info)
            self.m = 0

            # initializing the iteration number
            self.t = 1

            self.optimize = self.adam

    def sgd(self, gradient):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # compute the update as -epsilon * grad_f 
        return  -self.lr * gradient
        "*** YOUR CODE ENDS HERE ***"

    def heavyball_momentum(self, gradient):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        sgd_term = self.sgd(gradient)
        momentum_term = self.gama * self.v
        
        self.v = sgd_term + momentum_term

        return self.v
        "*** YOUR CODE ENDS HERE ***"

    def nestrov_momentum(self, gradient):
        return self.heavyball_momentum(gradient)

    def adam(self, gradient):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        m_t_1 = (1 - self.beta_m) * gradient + self.beta_m * self.m
        v_t_1 = (1 - self.beta_v) * np.square(gradient) + self.beta_v * self.v
        M_t_1 = m_t_1 / (1 - self.beta_m ** self.t)
        V_t_1 = v_t_1 / (1 - self.beta_v ** self.t)
        update = (-self.lr * M_t_1) / (np.sqrt(V_t_1) + self.epsilon)

        self.t += 1
        self.m = m_t_1
        self.v = v_t_1

        return update
        "*** YOUR CODE ENDS HERE ***"


class MultiClassLogisticRegression:
    def __init__(self, n_iter=10000, thres=1e-3):
        self.n_iter = n_iter
        self.thres = thres

    def fit(
        self,
        X,
        y,
        batch_size=64,
        lr=0.001,
        gama=0.9,
        beta_m=0.9,
        beta_v=0.999,
        epsilon=1e-8,
        rand_seed=4,
        verbose=False,
        optimizer="sgd",
    ):
        # setting the random state for consistency.
        np.random.seed(rand_seed)

        # find all classes in the train dataset.
        self.classes = self.unique_classes_(y)

        # assigning an integer value to each class, from 0 to (len(self.classes)-1)
        self.class_labels = self.class_labels_(self.classes)

        # one-hot-encode the labels.
        self.y_one_hot_encoded = self.one_hot(y)

        # add a column of 1 to the leftmost column.
        X = self.add_bias(X)

        # initialize the E_in list to keep track of E_in after each iteration.
        self.loss = []

        # initialize the weight parameters with a matrix of all zeros.
        # each row of self.weights contains the weights for one of the classes.
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))

        # create an instance of optimizer
        opt = Optimizer(
            optimizer, lr=lr, gama=gama, beta_m=beta_m, beta_v=beta_v, epsilon=epsilon
        )

        i, update = 0, 0
        while i < self.n_iter:
            self.loss.append(
                self.cross_entropy(self.y_one_hot_encoded, self.predict_with_X_aug_(X))
            )
            "*** YOUR CODE STARTS HERE ***"
            # TODO: sample a batch of data, X_batch and y_batch, with batch_size number of datapoint uniformly at random
            idxs = np.random.choice(X.shape[0],batch_size)
            X_batch = X[idxs]
            y_batch = self.y_one_hot_encoded[idxs]
            # TODO: find the gradient that should be inputed the optimization function.
            # NOTE: for nestrov_momentum, the gradient is derived at a point different from self.weights
            # See the assignments handout or the lecture note for more information.
            w_grad = self.weights + opt.v*gama if optimizer == "nestrov_momentum" else self.weights
            gradient = self.compute_grad(X_batch, y_batch, w_grad)
            # TODO: find the update vector by using the optimization method and update self.weights, accordingly.
            update = opt.optimize(gradient)
            self.weights = self.weights + update
            # TODO: stopping criterion. check if norm infinity of the update vector is smaller than self.thres.
            # if so, break the while loop.
            if np.max(np.abs(update)) < self.thres:
                break

            "*** YOUR CODE ENDS HERE ***"
            if i % 1000 == 0 and verbose:
                print(
                    " Training Accuray at {} iterations is {}".format(
                        i, self.evaluate_(X, self.y_one_hot_encoded)
                    )
                )
            i += 1
        return self

    def add_bias(self, X):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # More efficent approach for large arrays:
        bias = np.ones((X.shape[0], 1), dtype=X.dtype)
        X = np.concatenate((bias, X), axis=1)
        # Suggested approach:
        # X = np.insert(X,0,1,axis=1)
        return X
        "*** YOUR CODE ENDS HERE ***"

    def unique_classes_(self, y):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        return np.unique(y)
        "*** YOUR CODE ENDS HERE ***"

    def class_labels_(self, classes):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"

        return {cls: i for i, cls in enumerate(classes)}
        "*** YOUR CODE ENDS HERE ***"

    def one_hot(self, y):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # get unique classes
        unique_classes = self.unique_classes_(y)

        # Find the ohe index for each instance
        class_indices = np.array([self.class_labels[cls] for cls in y])

        # Generate ohe matrix
        ohe = np.eye(len(unique_classes))[class_indices]

        return ohe
        "*** YOUR CODE ENDS HERE ***"

    def softmax(self, z):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # Get the numerator (e^{z_i})
        z_exp = np.exp(z)

        # Sum over the rows, ensuring rows are kept aligned
        z_sum = np.sum(z_exp, axis=1,keepdims=True)

        # broadcast and return
        return z_exp/z_sum
        
        "*** YOUR CODE ENDS HERE ***"

    def predict_with_X_aug_(self, X_aug):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # Get logits
        z = X_aug @ self.weights.T
        # Apply softmax
        return self.softmax(z)
        "*** YOUR CODE ENDS HERE ***"

    def predict(self, X):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        X_aug = self.add_bias(X)
        return self.predict_with_X_aug_(X_aug)
        "*** YOUR CODE ENDS HERE ***"

    def predict_classes(self, X):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # Get predictions
        preds = self.predict(X)

        # Extract highes index along row axis
        class_indices = np.argmax(preds, axis=1)

        # Essentailly get unique labels
        labels = np.array(list(self.class_labels.keys()))

        # Return predicted labels
        return labels[class_indices]
        "*** YOUR CODE ENDS HERE ***"

    def score(self, X, y):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # get preds
        preds = self.predict_classes(X)

        # Count how many predictions are correct
        correct_predictions = np.sum(preds == y)

        # Calculate score
        return correct_predictions / len(y)
        "*** YOUR CODE ENDS HERE ***"

    def evaluate_(self, X_aug, y_one_hot_encoded):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # Get preds
        preds = self.predict_with_X_aug_(X_aug)

        # Get idxs of max probabailities
        label_idx = np.argmax(preds, axis=1)

        # Get idxs of y_ohe
        actual_idx = np.argmax(y_one_hot_encoded, axis=1)

        # Count correct predictions
        correct_predictions = np.sum(actual_idx == label_idx)

        # Return evaluation
        return correct_predictions / len(actual_idx)

        "*** YOUR CODE ENDS HERE ***"

    def cross_entropy(self, y_one_hot_encoded, probs):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        # Compute P(k)logQ(k) for all k
        loss = y_one_hot_encoded*np.log(probs)

        # sum over all k
        CE = np.sum(loss, axis=1)
        
        # Sum over all classes and take average
        average = -np.sum(CE) / probs.shape[0]
        
        return average
        "*** YOUR CODE ENDS HERE ***"

    def compute_grad(self, X_aug, y_one_hot_encoded, w):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        n = X_aug.shape[0]  
        # Compute z and apply softmax
        z = X_aug @ w.T
        pred = self.softmax(z)
        
        return  (pred - y_one_hot_encoded).T @ X_aug / n
        
        "*** YOUR CODE ENDS HERE ***"
def kmeans(examples, K, maxIters):
    """
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """

    # TODO: add your implementation here
    "*** YOUR CODE STARTS HERE ***"
    all_features = set()
    for counter in examples:
        all_features.update(counter.keys())
    all_features = list(all_features)  # Convert to a list to have consistent ordering

    # Step 2: Convert Counter objects to full vectors
    full_vectors = []
    for counter in examples:
        vector = [counter.get(feature, 0) for feature in all_features]
        full_vectors.append(vector)

    full_vectors = np.array(full_vectors)
    centroids = full_vectors[np.random.choice(full_vectors.shape[0], K, replace=False)]
    assignments = [np.random.randint(0, K) for _ in full_vectors]
    for _ in range(maxIters):

        for i in range(K):
            mean = 0
            counter = 0
            for j in range(full_vectors.shape[0]):
                if assignments[j] == i:
                    mean += full_vectors[j]
                    counter += 1
            if counter != 0:
                centroids[i] = mean / counter
        for idx in assignments:
            distances = [np.linalg.norm(full_vectors[idx] - mean) for mean in centroids] 
            assignments[idx] = np.argmin(distances)
    print(len(centroids))
    print(len(assignments))
    print(K)
    return centroids, assignments, 0
    "*** YOUR CODE ENDS HERE ***"