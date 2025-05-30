from .base_model import BaseModel
import numpy as np
import cv2

class MLModel(BaseModel):
    def __init__(self):
        # Параметры для бинаризации
        self.blur_size = (5, 5)
        self.threshold_block_size = 11
        self.threshold_C = 2
        
        # Параметры для поиска контуров
        self.min_contour_area = 100
        self.max_contour_area = 1000
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения"""
        # Преобразуем в оттенки серого, если изображение цветное
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Применяем размытие
        image = cv2.GaussianBlur(image, self.blur_size, 0)
        
        # Применяем адаптивную бинаризацию
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.threshold_block_size,
            self.threshold_C
        )
        
        # Морфологические операции для удаления шума
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def find_cells(self, binary: np.ndarray) -> list:
        """Поиск контуров клеток"""
        # Находим все контуры
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Фильтруем контуры по площади
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                valid_contours.append(contour)
                
        return valid_contours
    
    def predict(self, image: np.ndarray) -> int:
        """Предсказание количества клеток на изображении"""
        # Предобработка изображения
        binary = self.preprocess_image(image)
        
        # Поиск контуров клеток
        contours = self.find_cells(binary)
        
        # Возвращаем количество найденных клеток
        return len(contours)
    
    def train(self, images: list, labels: list) -> None:
        """Метод не используется, так как модель не требует обучения"""
        pass 