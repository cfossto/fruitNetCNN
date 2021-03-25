import numpy as np

class ReLU:
    # Forward pass 
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # dinputs = 0 då inputs värdet är mindre än 0 
        self.dinputs[self.inputs <= 0] = 0

class Softmax:
    def forward(self, inputs):
        # För stora inputs kan skapa overflow när de exponentieras
        # Därmed tas (varje element - största elementet) på raden
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        proba = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = proba
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_ouput = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)