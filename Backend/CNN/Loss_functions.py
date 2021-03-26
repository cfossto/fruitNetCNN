import numpy as np
from Backend.CNN.Activation_Functions import Softmax

class Loss:

    def regularization_loss(self, layer):
        regularization_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        samples_losses = self.forward(output, y)
        data_loss = np.mean(samples_losses)

        return data_loss, self.regularization_loss

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        labels = len(y_pred[0])

        # 100% confidence blir - värde i -np.log()
        # lägsta confidence får runtimewarning (inf)
        # klipp högsta och lägsta värde
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Omvandla labels till onehot-encoded
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        negativ_log = -np.log(correct_confidence)

        return negativ_log

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

# Kan i vissa fall göra processen mer effektiv genom att kombinera
# Softmax Aktivering och Categorical Cross entopy loss
class Act_Softmax_Loss_CCentropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # output-lagrets aktiverings funktion
        self.activation.forward(inputs)
        self.output = self.activation.output
        # Retunera loss
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # Om våra labels är one-hot encoded så omvandlas de till diskreta värden
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Gör kopia av indatan för modifiering
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        # Normalisera
        self.dinputs = self.dinputs / samples
        
class Dropout:
    def __init__(self, rate):
        # Dropout rate (rate), procenttal på andel av neuroner som slås av vid varje forward pass
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask