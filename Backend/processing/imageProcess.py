import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, optimizers
from os import path


def predict_image(path_to_file):
<<<<<<< HEAD
    pred_label, pred_conf = load_model(path_to_file)
    # print(pred + " Funkish")
    return pred_label, pred_conf
=======
    pred = load_model(path_to_file)
    print(pred)
    return pred
>>>>>>> 6187d68b41bd2a9de9d702a6711c45b4abd754b8



def sendToCNN():
    predicted_image_class = ""
    return predicted_image_class




def load_model(img_path):
<<<<<<< HEAD
    # model_path = r"/Users/christopherfossto/Desktop/Landscape-model.h5"  # user specific path (local)
    model_path = r"C:\Users\Kevin\Desktop\PetImages\Landscape-model.h5"
=======
    model_path = Backend.Landscape-model.h5 # User Specific Path!
>>>>>>> 6187d68b41bd2a9de9d702a6711c45b4abd754b8
    model = models.load_model(model_path)

    labels = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']

    pic = tf.keras.preprocessing.image.load_img(img_path, target_size=(150,150))
    pic = tf.keras.preprocessing.image.img_to_array(pic)
    pic = tf.expand_dims(pic, 0)
    prediction = model.predict(pic)
    confidence = 100 * np.max(prediction)
    print('label: ', labels[np.argmax(prediction)], 'confidence: ', 100 * np.max(prediction))
<<<<<<< HEAD
    return labels[np.argmax(prediction)], np.max(prediction)
=======
    return labels[np.argmax(prediction)],confidence
>>>>>>> 6187d68b41bd2a9de9d702a6711c45b4abd754b8
