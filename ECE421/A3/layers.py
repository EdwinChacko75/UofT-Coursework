import numpy as np
from collections import OrderedDict


from activations import Activation



class FullyConnectedLayer():
    """
    A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(self, n_out, activation):
        self.activation = Activation(activation)
        
        self.n_in = None
        self.n_out = n_out

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

    def _init_parameters(self, X_shape):
        """
        Initialize all layer parameters (weights, biases).
        """
        self.n_in = X_shape[1]
        W = np.random.normal(0, 1, size=(self.n_in, self.n_out))
        b = np.zeros((self.n_out,))[np.newaxis, :]

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = {}
        self.gradients: OrderedDict = OrderedDict({"W": np.zeros(W.shape), "b": np.zeros(b.shape)})

    def forward(self, X):
        """
        Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim) which is the output of the layer
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        
        # perform an affine transformation and activation

        W = self.parameters['W']

        # Afine transformation
        XW = X @ self.parameters['W']   # [batch_size, input_dim] x [input_dim, output_dim] -> [batch_size, output_dim]
        b = self.parameters['b']        # Shape: [np.newaxis, output_dim]
        Z = XW + b                      # Shape: [batch_size, output_dim]

        # Activation
        out = self.activation.forward(Z)
        
        # store information necessary for backprop in `self.cache`
        self.cache["Vars"] = (X, W, b, Z)

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY):
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        X, W, b, Z = self.cache["Vars"] 

        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer

        # Compute loss wrt input (using chain rule)
        dL_dZ = self.activation.backward(Z, dLdY)   # Shape: [batch_size, output_dim]
        dZ_dX = W                                   # Shape: [input_dim, output_dim]

        # Get output
        dX = dL_dZ @ dZ_dX.T                        # Shape: [batch_size, input_dim]

        # Compute other losses, using chain rule
        dL_dW = X.T @ dL_dZ                         # Shape: [input_dim, output_dim]
        dL_db = np.sum(dL_dZ, axis=0, keepdims=1)   # Shape: [np.newaxis, output_dim]

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        self.gradients['W'] = dL_dW
        self.gradients['b'] = dL_db
        
        ### END YOUR CODE ###

        return dX
    
    def clear_gradients(self):
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict({a: np.zeros_like(b) for a, b in self.gradients.items()})

    def forward_with_param(self, param_name, X):
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val):
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self):
        return [b for a, b in self.parameters.items()]

    def _get_cache(self):
        return [b for a, b in self.cache.items()]

    def _get_gradients(self):
        return [b for a, b in self.gradients.items()]
    