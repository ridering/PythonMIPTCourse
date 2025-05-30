import cv2
import numpy as np

class ImageFilters:
    @staticmethod
    def blur(image):
        """Применяет размытие к изображению"""
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    @staticmethod
    def sharpen(image):
        """Применяет повышение резкости к изображению"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def gradient(image):
        """Применяет градиентный фильтр к изображению"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)