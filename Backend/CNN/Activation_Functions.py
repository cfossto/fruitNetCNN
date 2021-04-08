import numpy as np

class ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # dinputs = 0 då inputs värdet är mindre än 0 
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs

class Softmax:
    def forward(self, inputs, training):
        self.inputs = inputs
        # För stora inputs kan skapa overflow när de exponentieras
        # Därmed tas (varje element - största elementet) på raden
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        proba = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = proba

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (otputs > 0.5) * 1 

class Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # linjär funktion där x=y; derivatan = 1
        # 1*dvalues = dvalues (kedjeregeln)
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs
