import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
from optimizer import BaseOptimizer, OptimizationResult
import pandas as pd
from datetime import datetime

@dataclass
class CellDetection:
    cell_path: str
    image_path: str
    position: Tuple[float, float]
    width: float
    score: float
    iterations: int
    optimizer_name: str

class CellFinder:
    def __init__(self, optimizer: BaseOptimizer, 
                 cells_dir: str = "data/cells",
                 images_dir: str = "data/images",
                 results_dir: str = "results"):
        self.optimizer = optimizer
        self.cells_dir = cells_dir
        self.images_dir = images_dir
        self.results_dir = results_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Создаем директорию для результатов если её нет
        os.makedirs(results_dir, exist_ok=True)
        
    def _load_image(self, path: str) -> np.ndarray:
        """Загружает изображение и конвертирует в оттенки серого"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {path}")
        return img
    
    def _get_all_files(self, directory: str, extension: str = ".png") -> List[str]:
        """Получает список всех файлов с заданным расширением в директории"""
        return [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.lower().endswith(extension)]
    
    def find_cells(self) -> List[CellDetection]:
        """Находит все клетки на всех изображениях"""
        cell_files = self._get_all_files(self.cells_dir, ".png")
        image_files = self._get_all_files(self.images_dir, ".jpg")
        
        results = []
        
        for cell_path in cell_files:
            cell_img = self._load_image(cell_path)
            self.logger.info(f"Обработка клетки: {os.path.basename(cell_path)}")
            
            for image_path in image_files:
                image = self._load_image(image_path)
                self.logger.info(f"Поиск на изображении: {os.path.basename(image_path)}")
                
                # Поиск клетки на изображении
                result = self.optimizer.optimize(cell_img, image)
                
                detection = CellDetection(
                    cell_path=cell_path,
                    image_path=image_path,
                    position=result.position,
                    width=result.width,
                    score=result.score,
                    iterations=result.iterations,
                    optimizer_name=self.optimizer.__class__.__name__
                )
                results.append(detection)
                
                # Сохраняем результаты в файл
                self._save_results(results)
                
        return results
    
    def _save_results(self, results: List[CellDetection]):
        """Сохраняет результаты в CSV файл"""
        df = pd.DataFrame([{
            'cell_path': r.cell_path,
            'image_path': r.image_path,
            'position_x': r.position[0],
            'position_y': r.position[1],
            'width': r.width,
            'score': r.score,
            'iterations': r.iterations,
            'optimizer': r.optimizer_name
        } for r in results])
        
        filename = os.path.join(self.results_dir, f"results_{self.optimizer.__class__.__name__}.csv")
        df.to_csv(filename, index=False)
        self.logger.info(f"Результаты сохранены в {filename}")
        
    def visualize_results(self, results: List[CellDetection], max_results: int = 5):
        """Визуализирует результаты поиска"""
        for detection in results[:max_results]:
            cell_img = self._load_image(detection.cell_path)
            image = self._load_image(detection.image_path)
            
            # Создаем цветное изображение для визуализации
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Рисуем рамку вокруг найденной клетки
            h, w = cell_img.shape
            aspect_ratio = h / w
            new_width = int(detection.width)
            new_height = int(new_width * aspect_ratio)
            x, y = int(detection.position[0]), int(detection.position[1])
            
            cv2.rectangle(vis_image, (x, y), (x + new_width, y + new_height), (0, 255, 0), 2)
            
            # Сохраняем результат
            output_path = os.path.join(
                self.results_dir,
                f"vis_{os.path.basename(detection.cell_path)}_{os.path.basename(detection.image_path)}"
            )
            cv2.imwrite(output_path, vis_image)
            self.logger.info(f"Визуализация сохранена в {output_path}") 