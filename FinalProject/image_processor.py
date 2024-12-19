import numpy as np
import cv2

class ImageProcessor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def additive_noise(image, percent):
        mask = np.random.rand(*image.shape) < 0.01 * percent
        noise = np.random.randint(-50, 51, image.shape) * mask
        return np.clip(image + noise, 0, 255)

    @staticmethod
    def mean_filter(image, kernel_size):
        from scipy.ndimage import convolve

        kernel = np.ones((kernel_size, 1)) / kernel_size
        return np.clip(convolve(convolve(image, kernel), kernel.T), 0, 255).astype(int)
    
    @staticmethod
    def gauss_filter(image, kernel_size):
        sigma = kernel_size // 2 / 3
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def image_equalization(image):
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        
        equalized = cdf_normalized[image]
        return equalized.astype(int)
    
    @staticmethod
    def statistic_correction(image, new_mean, new_std):
        image = image.astype(np.float32)
        mean, std = image.mean(), image.std()
        
        corrected_image = (image - mean) * (new_std / std) + new_mean
        corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
        
        return corrected_image

    @staticmethod
    def resize(image, new_width, new_height):
        return cv2.resize(image, (new_width, new_height))

    @staticmethod
    def shift(image, x, y):
        h, w = image.shape
        shifted = np.zeros_like(image)
        
        x_indices = np.arange(w) - x
        y_indices = np.arange(h) - y

        valid_x = (x_indices >= 0) & (x_indices < w)
        valid_y = (y_indices >= 0) & (y_indices < h)

        shifted[np.ix_(valid_y, valid_x)] = image[np.ix_(valid_y, valid_x)]
        
        return shifted

    @staticmethod
    def rotation(image, k, l, angle):
        angle_rad = np.deg2rad(angle)

        height, width = image.shape[:2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        x_new = (x - k) * np.cos(angle_rad) - (y - l) * np.sin(angle_rad) + k
        y_new = (x - k) * np.sin(angle_rad) + (y - l) * np.cos(angle_rad) + l
        
        rotated_image = np.zeros_like(image)
        x_new = np.clip(x_new, 0, width - 1).astype(int)
        y_new = np.clip(y_new, 0, height - 1).astype(int)

        rotated_image[y_new, x_new] = image[y, x]
        rotated_image[y_new, np.clip(x_new + 1,0,  width - 1)] = image[y, x]
        
        return rotated_image

    @staticmethod
    def glass_effect(image):
        height, width = image.shape[:2]
    
        x_offset = np.random.normal(0, 3, (height, width))
        y_offset = np.random.normal(0, 3, (height, width))

        x, y = np.meshgrid(np.arange(width), np.arange(height))

        x_new = np.clip(x + x_offset, 0, width - 1).astype(np.float32)
        y_new = np.clip(y + y_offset, 0, height - 1).astype(np.float32)

        distorted_image = cv2.remap(image, x_new, y_new, 
                                    interpolation=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_REFLECT)

        return distorted_image
    
    @staticmethod
    def waves(image):
        height, width = image.shape[:2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        rand_x = np.clip(x + 20 * np.sin(2 * np.pi * y / 30), 0, width - 1).astype(int)

        return image[y, rand_x]
    
    @staticmethod
    def motion_blur(image, n):
        from scipy.ndimage import convolve
        E = np.eye(n) / n
        return convolve(image, E)
