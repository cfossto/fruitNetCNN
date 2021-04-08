import numpy as np

class Dense_layer:

    # wr = weight regularization; br = bias regularization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Varje node har weights och biases
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Initializing L1 and L2 regularizer
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs
        # detta är outputsen från nodes
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.weights < 0] = -1
            self.dweights += self.bias_regularizer_l1 * dL1 

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Conv_layer:
    def __init__(self, filters=16, stride=1, kernel_size=3):
        self.filters = np.random.randn(filters, 3, 3) * 0.1
        self.stride = stride
        self.kernel_size = kernel_size
        self.last_input = None

    def forward(self, inputs, training):
        self.last_input = inputs
        input_dimension = inputs.shape[1]
        output_dimension = int((input_dimension - self.kernel_size) / self.stride) + 1
        out = np.zeros((self.filters.shape[0], output_dimension, output_dimension))

        for i in range(self.filters.shape[0]):
            out_y = 0
            tmp_y = out_y
            while tmp_y + self.kernel_size <= input_dimension:
                out_x = 0
                tmp_x = out_x 
                while tmp_x + self.kernel_size <= input_dimension:
                    patch = inputs[:, tmp_y:tmp_y + self.kernel_size, tmp_x:tmp_x + self.kernel_size]
                    out[i, out_y, out_x] += np.sum(self.filters[i] * patch)
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        self.output = out

    def backward(self, dvalues, learn_rate=0.005):
        input_dimension = self.last_input.shape[1]
        doutput = np.zeros(self.last_input.shape)
        dfilt = np.zeros(self.filters.shape)

        for i in range(self.filters.shape[0]):
            out_y = 0
            tmp_y = out_y
            while tmp_y + self.kernel_size <= input_dimension:
                out_x = 0
                tmp_x = out_x
                while tmp_x + self.kernel_size <= input_dimension:
                    patch = self.last_input[:, tmp_y:tmp_y + self.kernel_size, tmp_x:tmp_x + self.kernel_size]
                    dfilt[i] += np.sum(dvalues[i, out_y, out_x] * patch, axis=0)
                    doutput[:, tmp_y:tmp_y + self.kernel_size, tmp_x:tmp_x + self.kernel_size] += dvalues[i, out_y, out_x] * self.filters[i]

                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        self.filters -= learn_rate * dfilt
        self.output = doutput

    def get_weights(self):
        self.output = np.reshape(self.filters, -1)

class Pooling_layer:
    def __init__(self, name, stride=1, size=2):
        self.last_input = None
        self.stride = stride
        self.size = size

    def forward(self, inputs, training):
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

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

# Lager som är nödvändigt för training men tom output.
# Training bygger på data från föregående lager men första lager
# har ej ett föregående lager. Hence
class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs


