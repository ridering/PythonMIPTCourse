from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Предсказывает количество клеток на изображении
        
        Args:
            image: Изображение в формате numpy array
            
        Returns:
            int: Количество обнаруженных клеток
        """
        pass
    
    @abstractmethod
    def train(self, images: list, labels: list) -> None:
        """
        Обучает модель на предоставленных данных
        
        Args:
            images: Список изображений
            labels: Список меток (количество клеток на каждом изображении)
        """
        pass 