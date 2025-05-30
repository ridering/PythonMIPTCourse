from .base_model import BaseModel
import numpy as np
from ultralytics import YOLO
import cv2
import os


class CNNModel(BaseModel):
    def __init__(self):
        # Получаем путь к файлу weights.pt
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        args_path = os.path.join(base_dir, 'src', 'models', 'weights.pt')
        
        # Загружаем предобученную модель YOLOv8n с конфигурацией и весами из weights.pt
        self.model = YOLO(args_path)
        
    def predict(self, image: np.ndarray) -> int:
        """Предсказание количества клеток на изображении"""
        # Запускаем предсказание
        image = cv2.resize(image, (640, 640))
        results = self.model(image)
        
        # Считаем количество обнаруженных объектов
        count = 0
        for result in results:
            count += len(result.boxes)
            
        return count
    
    def train(self, images: list, labels: list) -> None:
        """Метод не используется, так как используем предобученную модель"""
        pass 