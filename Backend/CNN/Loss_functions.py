import numpy as np
from Activation_Functions import Softmax

class Loss:

    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss


    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers


    def calculate(self, output, y, *, include_regularization=False):
        samples_losses = self.forward(output, y)
        data_loss = np.mean(samples_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()


    def calculate_accumulated(self, *. include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss
    
        return data_loss, self.regularization_loss()


    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count


class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        labels = len(y_pred[0])

        # 100% confidence blir - värde i -np.log()
        # lägsta confidence får runtimewarning (inf)
        # klipp högsta och lägsta värde
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Omvandla labels till onehot-encoded
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples) * y_true]
        elif len(y_true.shape) == 2:
            correct_confidence np.sum(y_pred * y_true, axis=1)
        
        negativ_log_likelihoods = -np.log(correct_confidence)

        return negativ_log_likelihoods


    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

# Kan i vissa fall göra processen mer effektiv genom att kombinera
# Softmax Aktivering och Categorical Cross entopy loss
class Act_Softmax_Loss_CCentropy():
    # def __init__(self):
    #     self.activation = Softmax()
    #     self.loss = CategoricalCrossentropy()

    # def forward(self, inputs, y_true):
    #     # output-lagrets aktiverings funktion
    #     self.activation.forward(inputs)
    #     self.output = self.activation.output
    #     # Retunera loss
    #     return self.loss.calculate(self.output, y_true)

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


class BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)

        return sample_losses


    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)

        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples


class MeanSquaredError(Loss):       # L2 loss
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=1)
        return sample_losses
    

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # outputs per sample, utgår från första index
        outputs = len(dvalues[0])

        # gradient
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # normaliserar gradienterna
        self.dinputs = self.dinputs / samples


class MeanAbsoluteError(Loss):  # L1 loss
    def forward(self, y_pred, y_true):
        samples_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        
        return samples_losses


    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true- dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy


    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy


    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary


    def init(self, y):
        pass


    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)    
        
        return predictions == y


class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None


    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250


    def compare(self, predictions, y):

        return np.absolute(predictions - y) < self.precision