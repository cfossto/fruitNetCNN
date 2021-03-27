from cv2 import cv2

def greeting():
    hello = "Hello"
    return hello


def process(y):
    x = y*25
    return x



def animg(img):
    sh = cv2.imread(img)
    return sh