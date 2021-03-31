import numpy as np
from cv2 import cv2

def imageprocess(path_to_file):
    processed_image = cv2.imread(path_to_file)
    print(np.shape(processed_image))
    return processed_image



def sendToCNN():
    predicted_image_class = ""
    return predicted_image_class