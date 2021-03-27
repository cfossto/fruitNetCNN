from Layers import Dense_layer, Conv_layer, Pooling_layer, Dropout_layer
from Activation_Functions import ReLU, Softmax
from Loss_functions import Loss, Softmax, CategoricalCrossentropy, Act_Softmax_Loss_CCentropy
from Optimizers import SGD, Adam
from Model import Model, Accuracy, Accuracy_Categorical

# Exempel model
hugh = Model()

hugh.add(Dense_layer(2, 512, weight_regularizer_l2=0.0004, bias_regularizer_l2=0.0004))
hugh.add(ReLU())
hugh.add(Dropout_layer(0.2))
hugh.add(Dense_layer(512, 3))
hugh.add(Softmax())

hugh.setters(loss=CategoricalCrossentropy(), 
             optimizer=Adam(learning_rate=0.05, 
             decay=0.00005), accuracy=Accuracy_Categorical())

hugh.finalize()

hugh.train(X_train, y_train, validation_data=(X_val, y_val),
           epochs=1000, print_every=100)

