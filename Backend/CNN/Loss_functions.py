import numpy as np

class Loss:

    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer * np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def calculate(self, output, y):
        samples_losses = self.forward(output, y)
        data_loss = np.mean(samples_losses)

        return data_loss

class CategoricalCrossentropy:
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

