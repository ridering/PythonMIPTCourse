from .base_model import BaseModel
import numpy as np
import cv2
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN

class ClusteringModel(BaseModel):
    def __init__(self):
        # Инициализация фильтров Laws
        self.L5 = np.array([1, 4, 6, 4, 1])
        self.E5 = np.array([-1, -2, 0, 2, 1])
        self.S5 = np.array([-1, 0, 2, 0, -1])
        self.R5 = np.array([1, -4, 6, -4, 1])
        
        self.filters_1d = [self.L5, self.E5, self.S5, self.R5]
        self.filter_names = ['L5', 'E5', 'S5', 'R5']
        
    def create_laws_filters(self):
        """Создание 2D фильтров Laws"""
        filters = {}
        for i, f1 in enumerate(self.filters_1d):
            for j, f2 in enumerate(self.filters_1d):
                name = f"{self.filter_names[i]}{self.filter_names[j]}"
                filters[name] = np.outer(f1, f2)
        return filters
    
    def zero_mean(self, image, kernel_size=15):
        """Вычитание локального среднего"""
        mean_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        local_mean = convolve2d(image, mean_kernel, mode='same', boundary='symm')
        return image - local_mean
    
    def apply_laws_filters(self, image, kernel_size=15):
        """Применение фильтров Laws"""
        image = self.zero_mean(image, kernel_size=kernel_size)
        filters = self.create_laws_filters()
        energy_maps = {}
        for name, kernel in filters.items():
            filtered = convolve2d(image, kernel, mode='same', boundary='symm')
            energy = cv2.boxFilter(np.abs(filtered), ddepth=-1, ksize=(kernel_size, kernel_size))
            energy_maps[name] = energy
        return energy_maps
    
    def combine_symmetric_energies(self, energy_maps):
        """Объединение симметричных фильтров"""
        combined = {
            'E5E5': energy_maps['E5E5'],
            'S5S5': energy_maps['S5S5'],
            'R5R5': energy_maps['R5R5'],
            'L5E5': 0.5 * (energy_maps['L5E5'] + energy_maps['E5L5']),
            'L5S5': 0.5 * (energy_maps['L5S5'] + energy_maps['S5L5']),
            'L5R5': 0.5 * (energy_maps['L5R5'] + energy_maps['R5L5']),
            'E5S5': 0.5 * (energy_maps['E5S5'] + energy_maps['S5E5']),
            'E5R5': 0.5 * (energy_maps['E5R5'] + energy_maps['R5E5']),
            'S5R5': 0.5 * (energy_maps['S5R5'] + energy_maps['R5S5']),
        }
        return combined
    
    def cluster_texture(self, image, combined_maps, eps=0.5, min_samples=50):
        """Кластеризация текстур"""
        # Преобразуем карты в список признаков [N, 9]
        feature_stack = np.dstack([cv2.resize(combined_maps[key], (100, 100)) for key in sorted(combined_maps)])
        feature_stack = np.dstack([feature_stack, cv2.resize(image, (100, 100))])
        feature_vectors = feature_stack.reshape(-1, 10)

        # Применение DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(feature_vectors)
        labels = clustering.labels_

        cell_count = len(set(labels)) - (1 if -1 in labels else 0)  # Убираем шум
        return cell_count
    
    def predict(self, image: np.ndarray) -> int:
        """Предсказание количества клеток на изображении"""
        # Преобразуем в оттенки серого, если изображение цветное
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применяем размытие
        image = cv2.GaussianBlur(image, (9, 9), 0.5)
        
        # Применяем фильтры Laws
        energy_maps = self.apply_laws_filters(image)
        combined_maps = self.combine_symmetric_energies(energy_maps)
        
        # Кластеризуем и считаем клетки
        return self.cluster_texture(image, combined_maps, eps=10, min_samples=5)
    
    def train(self, images: list, labels: list) -> None:
        """Метод не используется, так как модель не требует обучения"""
        pass 