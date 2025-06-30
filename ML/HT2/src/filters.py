import cv2
import numpy as np


class ImageAugmentation:
    @staticmethod
    def augment_brightness(image):
        """
        Аугментация яркости изображения
        
        Args:
            image (numpy.ndarray): Входное изображение
            
        Returns:
            numpy.ndarray: Изображение с измененной яркостью
        """
        alpha = np.random.uniform(0.8, 1.2)  # коэффициент яркости
        beta = np.random.uniform(-30, 30)    # смещение яркости
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def augment_contrast(image):
        """
        Аугментация контраста изображения
        
        Args:
            image (numpy.ndarray): Входное изображение
            
        Returns:
            numpy.ndarray: Изображение с измененным контрастом
        """
        alpha = np.random.uniform(0.8, 1.2)  # коэффициент контраста
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    @staticmethod
    def augment_rotation(image):
        """
        Аугментация поворота изображения
        
        Args:
            image (numpy.ndarray): Входное изображение
            
        Returns:
            numpy.ndarray: Повернутое изображение
        """
        angle = np.random.uniform(-15, 15)  # угол поворота в градусах
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    @staticmethod
    def augment_flip(image):
        """
        Аугментация отражения изображения
        
        Args:
            image (numpy.ndarray): Входное изображение
            
        Returns:
            numpy.ndarray: Отраженное изображение
        """
        if np.random.random() > 0.5:
            return cv2.flip(image, 1)  # горизонтальное отражение
        return image

    @staticmethod
    def augment_noise(image):
        """
        Добавление шума к изображению
        
        Args:
            image (numpy.ndarray): Входное изображение
            
        Returns:
            numpy.ndarray: Изображение с добавленным шумом
        """
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

    @classmethod
    def apply_augmentations(cls, image, augment=True):
        """
        Применяет все аугментации к изображению
        
        Args:
            image (numpy.ndarray): Входное изображение
            augment (bool): Применять ли аугментацию
            
        Returns:
            numpy.ndarray: Изображение с примененными аугментациями
        """
        if not augment:
            return image

        # Применяем аугментации с вероятностью 0.5
        if np.random.random() > 0.5:
            image = cls.augment_brightness(image)
        if np.random.random() > 0.5:
            image = cls.augment_contrast(image)
        if np.random.random() > 0.5:
            image = cls.augment_rotation(image)
        if np.random.random() > 0.5:
            image = cls.augment_flip(image)
        if np.random.random() > 0.5:
            image = cls.augment_noise(image)
            
        return image 