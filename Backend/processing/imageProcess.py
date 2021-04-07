import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, optimizers


def predict_image(path_to_file):
    pred_label, pred_conf = load_model(path_to_file)
    # print(pred + " Funkish")
    return pred_label, pred_conf



def sendToCNN():
    predicted_image_class = ""
    return predicted_image_class




def load_model(img_path):
    # model_path = r"/Users/christopherfossto/Desktop/Landscape-model.h5"  # user specific path (local)
    model_path = r"C:\Users\Kevin\Desktop\PetImages\Landscape-model.h5"
    model = models.load_model(model_path)

    labels = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']

    pic = tf.keras.preprocessing.image.load_img(img_path, target_size=(150,150))
    pic = tf.keras.preprocessing.image.img_to_array(pic)
    pic = tf.expand_dims(pic, 0)
    prediction = model.predict(pic)
    print('label: ', labels[np.argmax(prediction)], 'confidence: ', 100 * np.max(prediction))
    return labels[np.argmax(prediction)], np.max(prediction)