import numpy as np

class Dense_layer:

    # wr = weight regularization; br = bias regularization
    def __init__(self, n_inputs, n_neurons, wr_l1=0, wr_l2=0,
                 br_l1=0, br_l2=0):
        # Varje node har weights och biases
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Initializing L1 and L2 regularizer
        self.wr_l1 = wr_l1
        self.wr_l2 = wr_l2
        self.br_l1 = br_l1
        self.br_l2 = br_l2

    def forward(self, inputs):
        
        self.inputs = inputs
        # detta är outputsen från nodes
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
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






















































class Pooling_layer:
    def __init__(self, name, stride=1, size=2):
        self.last_input = None
        self.stride = stride
        self.size = size

    def forward(self, inputs):
        self.last_input = inputs

        num_channels, h_prev, w_prev = inputs.shape
        h = int((h_prev - self.size) / self.stride) + 1
        w = int((w_prev - self.size) / self.stride) + 1

        downsampled = np.zeros((num_channels, h, w))

        for i in range(num_channels):
            out_y = 0
            curr_y = out_y
            while curr_y + self.size <= h_prev:
                out_x = 0
                curr_x = out_x
                while curr_x + self.size <= w_prev:
                    patch = inputs[i, curr_y:curr_y + self.size, curr_x:curr_x + self.size]
                    downsampled[i, out_y, out_x] = np.max(patch)
                    curr_x += self.stride
                    out_x += 1
                curr_y += self.stride
                out_y += 1
        
        self.output = downsampled

    def backward(self, din, learning_rate):
        num_channels, orig_dim, *_ = self.last_input.shape

        doutput = np.zeros(self.last_input.shape)

        for c in range(num_channels):
            out_y = 0
            tmp_y = out_y
            while tmp_y + self.size <= orig_dim:
                out_x = 0
                tmp_x = out_x
                while tmp_x + self.size <= orig_dim:
                    patch = self.last_input[c, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    (x,y) = np.unravel_index(np.nanargmax(patch), patch.shape)
                    doutput[c, tmp_y + x, tmp_x + y] += din[c, out_y, out_x]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        
        self.output = doutput

    def get_weights(self):
        self.output = 0
            


class Dropout_layer:
    def __init__(self, rate):
        # Dropout rate (rate), procenttal på andel av neuroner som slås av vid varje forward pass
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask