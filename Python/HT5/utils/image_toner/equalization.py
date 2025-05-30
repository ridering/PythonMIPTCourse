import cv2


def equalize_image(img):
    return cv2.equalizeHist(img)
