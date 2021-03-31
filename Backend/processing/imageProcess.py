import numpy as np
from cv2 import cv2

def imageprocess(path_to_file):
    processed_image = cv2.imread(path_to_file)
    print(np.shape(processed_image))
    return processed_image

imageprocess("uploads/Skarmavbild_2021-03-27_kl._15.44.07.png")



def sendToCNN():
    predicted_image_class = ""
    return predicted_image_class