import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import random
from .generator import BloodCellGenerator

class YOLODatasetGenerator:
    def __init__(self, data_dir, output_dir, image_size=(640, 640), num_samples=1000, train_ratio=0.8):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        
        # Создаем директории для датасета
        self.train_images_dir = self.output_dir / 'train' / 'images'
        self.train_labels_dir = self.output_dir / 'train' / 'labels'
        self.val_images_dir = self.output_dir / 'val' / 'images'
        self.val_labels_dir = self.output_dir / 'val' / 'labels'
        
        for dir_path in [self.train_images_dir, self.train_labels_dir, 
                        self.val_images_dir, self.val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем генератор изображений
        self.generator = BloodCellGenerator(data_dir, image_size)
    
    def generate_dataset(self):
        """Генерирует датасет для обучения YOLO"""
        print(f"Генерация датасета из {self.num_samples} изображений...")
        
        # Определяем количество изображений для тренировки и валидации
        num_train = int(self.num_samples * self.train_ratio)
        num_val = self.num_samples - num_train
        
        # Генерируем тренировочные данные
        print("Генерация тренировочных данных...")
        for i in range(num_train):
            if (i + 1) % 100 == 0:
                print(f"Сгенерировано {i + 1} тренировочных изображений")
                
            # Генерируем изображение с координатами клеток
            image, bboxes = self.generator.generate_image(return_bboxes=True)
            
            # Сохраняем изображение
            image_path = self.train_images_dir / f'image_{i:04d}.jpg'
            cv2.imwrite(str(image_path), image)
            
            # Сохраняем аннотации в формате YOLO
            label_path = self.train_labels_dir / f'image_{i:04d}.txt'
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    # Нормализуем координаты для формата YOLO
                    x_center = bbox['center_x'] / self.image_size[1]
                    y_center = bbox['center_y'] / self.image_size[0]
                    width = bbox['width'] / self.image_size[1]
                    height = bbox['height'] / self.image_size[0]
                    
                    # Записываем в формате: class x_center y_center width height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Генерируем валидационные данные
        print("Генерация валидационных данных...")
        for i in range(num_val):
            if (i + 1) % 100 == 0:
                print(f"Сгенерировано {i + 1} валидационных изображений")
                
            # Генерируем изображение с координатами клеток
            image, bboxes = self.generator.generate_image(return_bboxes=True)
            
            # Сохраняем изображение
            image_path = self.val_images_dir / f'image_{i:04d}.jpg'
            cv2.imwrite(str(image_path), image)
            
            # Сохраняем аннотации в формате YOLO
            label_path = self.val_labels_dir / f'image_{i:04d}.txt'
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    # Нормализуем координаты для формата YOLO
                    x_center = bbox['center_x'] / self.image_size[1]
                    y_center = bbox['center_y'] / self.image_size[0]
                    width = bbox['width'] / self.image_size[1]
                    height = bbox['height'] / self.image_size[0]
                    
                    # Записываем в формате: class x_center y_center width height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print("Генерация датасета завершена!")
        return str(self.output_dir) 