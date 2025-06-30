import numpy as np
from abc import ABC, abstractmethod
import cv2
from typing import Tuple, List, Optional
import logging
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class OptimizationResult:
    position: Tuple[float, float]
    width: float
    score: float
    iterations: int

class BaseOptimizer(ABC):
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def optimize(self, template: np.ndarray, image: np.ndarray, 
                initial_position: Optional[Tuple[float, float]] = None,
                initial_width: Optional[float] = None) -> OptimizationResult:
        pass
    
    def _calculate_mse(self, template: np.ndarray, image: np.ndarray, 
                      position: Tuple[float, float], width: float) -> float:
        """Вычисляет MSE между шаблоном и изображением в заданной позиции и ширине"""
        h, w = template.shape
        aspect_ratio = h / w
        new_width = int(width)
        new_height = int(new_width * aspect_ratio)
        resized_template = cv2.resize(template, (new_width, new_height))
        
        x, y = int(position[0]), int(position[1])
        if x < 0 or y < 0 or x + new_width > image.shape[1] or y + new_height > image.shape[0]:
            return float('inf')
            
        roi = image[y:y+new_height, x:x+new_width]
        if roi.shape != resized_template.shape:
            return float('inf')
            
        return np.mean((roi - resized_template) ** 2)

class MonteCarloGradientOptimizer(BaseOptimizer):
    def __init__(self, n_samples: int = 100, width_range: Tuple[float, float] = (10, 50), **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.width_range = width_range
        
    def optimize(self, template: np.ndarray, image: np.ndarray,
                initial_position: Optional[Tuple[float, float]] = None,
                initial_width: Optional[float] = None) -> OptimizationResult:
        
        if initial_position is None:
            initial_position = (image.shape[1]//2, image.shape[0]//2)
        if initial_width is None:
            initial_width = (self.width_range[0] + self.width_range[1]) / 2
            
        best_position = initial_position
        best_width = initial_width
        best_score = float('inf')
        
        # Монте-Карло поиск начальной точки
        for _ in range(self.n_samples):
            x = np.random.randint(0, image.shape[1])
            y = np.random.randint(0, image.shape[0])
            width = np.random.uniform(self.width_range[0], self.width_range[1])
            
            score = self._calculate_mse(template, image, (x, y), width)
            if score < best_score:
                best_score = score
                best_position = (x, y)
                best_width = width
        
        # Градиентный спуск
        current_position = best_position
        current_width = best_width
        current_score = best_score
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            old_score = current_score
            
            # Градиент по позиции
            dx = 1
            dy = 1
            width_step = 1.0
            
            # Проверяем соседние точки
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                new_pos = (current_position[0] + dx, current_position[1] + dy)
                new_score = self._calculate_mse(template, image, new_pos, current_width)
                if new_score < current_score:
                    current_position = new_pos
                    current_score = new_score
            
            # Проверяем соседние ширины
            for dw in [width_step, -width_step]:
                new_width = current_width + dw
                if self.width_range[0] <= new_width <= self.width_range[1]:
                    new_score = self._calculate_mse(template, image, current_position, new_width)
                    if new_score < current_score:
                        current_width = new_width
                        current_score = new_score
            
            if abs(old_score - current_score) < self.tolerance:
                break
                
        return OptimizationResult(
            position=current_position,
            width=current_width,
            score=current_score,
            iterations=iterations
        )

class GridSearchOptimizer(BaseOptimizer):
    def __init__(self, grid_step: float = 1.0, width_range: Tuple[float, float] = (10, 50),
                 width_steps: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.grid_step = grid_step
        self.width_range = width_range
        self.width_steps = width_steps
        
    def optimize(self, template: np.ndarray, image: np.ndarray,
                initial_position: Optional[Tuple[float, float]] = None,
                initial_width: Optional[float] = None) -> OptimizationResult:
        
        best_position = (0, 0)
        best_width = (self.width_range[0] + self.width_range[1]) / 2
        best_score = float('inf')
        iterations = 0
        
        widths = np.linspace(self.width_range[0], self.width_range[1], self.width_steps)
        
        for width in tqdm(widths, desc="Grid Search"):
            for x in range(0, image.shape[1], int(self.grid_step)):
                for y in range(0, image.shape[0], int(self.grid_step)):
                    iterations += 1
                    score = self._calculate_mse(template, image, (x, y), width)
                    if score < best_score:
                        best_score = score
                        best_position = (x, y)
                        best_width = width
                        
        return OptimizationResult(
            position=best_position,
            width=best_width,
            score=best_score,
            iterations=iterations
        )

class PeanoOptimizer(BaseOptimizer):
    def __init__(self, width_range: Tuple[float, float] = (80, 120), 
                 grid_size: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.width_range = width_range
        self.grid_size = grid_size
        
    def _generate_peano_points(self, image_shape: Tuple[int, int]) -> List[Tuple[float, float, float]]:
        """Генерирует точки для поиска на основе развертки Пеано"""
        height, width = image_shape
        points = []
        
        # Создаем сетку точек
        x_step = width / self.grid_size
        y_step = height / self.grid_size
        w_step = (self.width_range[1] - self.width_range[0]) / self.grid_size
        
        # Генерируем точки в порядке развертки Пеано
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    # Используем разные смещения для лучшего покрытия
                    x = (i + 0.5) * x_step
                    y = (j + 0.5) * y_step
                    w = self.width_range[0] + (k + 0.5) * w_step
                    points.append((x, y, w))
        
        # Перемешиваем точки для лучшего покрытия
        np.random.shuffle(points)
        return points
        
    def optimize(self, template: np.ndarray, image: np.ndarray,
                initial_position: Optional[Tuple[float, float]] = None,
                initial_width: Optional[float] = None) -> OptimizationResult:
        
        best_position = (0, 0)
        best_width = (self.width_range[0] + self.width_range[1]) / 2
        best_score = float('inf')
        iterations = 0
        
        # Генерируем точки для поиска
        search_points = self._generate_peano_points(image.shape)
        
        # Перебираем точки
        for x, y, width in tqdm(search_points, desc="Peano Search"):
            score = self._calculate_mse(template, image, (x, y), width)
            iterations += 1
            
            if score < best_score:
                best_score = score
                best_position = (x, y)
                best_width = width
                
        # Локальная оптимизация вокруг лучшей точки
        current_position = best_position
        current_width = best_width
        current_score = best_score
        
        # Градиентный спуск с адаптивным шагом
        step_sizes = [10, 5, 2, 1]  # Начинаем с больших шагов
        
        for step_size in step_sizes:
            improved = True
            while improved and iterations < self.max_iterations:
                improved = False
                iterations += 1
                
                # Проверяем соседние точки в радиусе step_size
                for dx in range(-step_size, step_size + 1, step_size):
                    for dy in range(-step_size, step_size + 1, step_size):
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = (current_position[0] + dx, current_position[1] + dy)
                        new_score = self._calculate_mse(template, image, new_pos, current_width)
                        if new_score < current_score:
                            current_position = new_pos
                            current_score = new_score
                            improved = True
                
                # Проверяем соседние ширины
                width_step = step_size
                for dw in [width_step, -width_step]:
                    new_width = current_width + dw
                    if self.width_range[0] <= new_width <= self.width_range[1]:
                        new_score = self._calculate_mse(template, image, current_position, new_width)
                        if new_score < current_score:
                            current_width = new_width
                            current_score = new_score
                            improved = True
                
                if abs(current_score - best_score) < self.tolerance:
                    break
                    
        return OptimizationResult(
            position=current_position,
            width=current_width,
            score=current_score,
            iterations=iterations
        ) 