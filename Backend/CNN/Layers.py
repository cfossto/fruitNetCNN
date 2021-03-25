import numpy as np

class Dense_layer:

    # layer initialization
    # wr = weight regularization; br = bias regularization
    def __init__(self, n_inputs, n_neurons, wr_l1=0, wr_l2=0,
                 br_l1=0, br_l2=0):
        # Initializing weights and biases
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Initializing L1 and L2 regularizer
        self.wr_l1 = wr_l1
        self.wr_l2 = wr_l2
        self.br_l1 = br_l1
        self.br_l2 = br_l2

    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Create gradients (2d-array/matrices of out partial derivatives)
        # by performing dot products on inputs and dvalues
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 on weights
        if self.wr_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.wr_l1 * dL1
        
        # L2 on weights
        if self.wr_l2 > 0:
            self.dweights += 2 * self.wr_l2 * self.weights

        # L1 on biases
        if self.br_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.weights < 0] = -1
            self.dweights += self.br_l1 * dL1 

        # L2 on biases
        if self.br_l2 > 0:
            self.dbiases += 2 * self.br_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
