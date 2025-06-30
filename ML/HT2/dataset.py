import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from filters import ImageAugmentation


class BloodCellDataset(Dataset):
    def __init__(self, data_dir, image_size=(1024, 1024), num_samples=1000, augment=True):
        """
        Инициализация датасета
        
        Args:
            data_dir (str): Путь к директории с данными
            image_size (tuple): Размер выходного изображения
            num_samples (int): Количество сгенерированных изображений в датасете
            augment (bool): Применять ли аугментацию
        """
        self.cells_dir = os.path.join(data_dir, "cells")
        self.backgrounds_dir = os.path.join(data_dir, "backgrounds")
        self.image_size = image_size
        self.num_samples = num_samples
        self.augment = augment

        self.cells = self.load_cells()
        self.backgrounds = self.load_backgrounds()
    
    def __len__(self):
        """
        Возвращает количество экземпляров в датасете
        """
        return self.num_samples

    def load_cells(self):
        cells = []
        for file in os.listdir(self.cells_dir):
            img = cv2.imread(os.path.join(self.cells_dir, file))
            if img is not None:
                cells.append(img)
        return cells

    def load_backgrounds(self):
        backgrounds = []
        for file in os.listdir(self.backgrounds_dir):
            img = cv2.imread(os.path.join(self.backgrounds_dir, file))
            if img is not None:
                img = cv2.resize(img, (500, 500))
                backgrounds.append(img)
        return backgrounds

    def generate_background(self):
        canvas = self.backgrounds[np.random.randint(0, len(self.backgrounds))]
        canvas = cv2.resize(canvas, self.image_size)

        for _ in range(10):
            background = self.backgrounds[np.random.randint(0, len(self.backgrounds))]
            mask = np.full_like(background, 255)
            
            h, w = background.shape[:2]
            
            center_y = np.random.randint(0, self.image_size[0])
            center_x = np.random.randint(0, self.image_size[1])

            h_left_up = center_y - h // 2
            h_right_down = center_y + h // 2
            w_left_up = center_x - w // 2
            w_right_down = center_x + w // 2

            if w_left_up < 0:
                background = background[:, -w_left_up:]
                mask = mask[:, -w_left_up:]
                center_x -= w_left_up // 2

            elif w_right_down > canvas.shape[1]:
                background = background[:, :canvas.shape[1] - w_right_down]
                mask = mask[:, :canvas.shape[1] - w_right_down]
                center_x += (canvas.shape[1] - w_right_down) // 2

            if h_left_up < 0:
                background = background[-h_left_up:, :]
                mask = mask[-h_left_up:, :]
                center_y -= h_left_up // 2

            elif h_right_down > canvas.shape[0]:
                background = background[:canvas.shape[0] - h_right_down, :]
                mask = mask[:canvas.shape[0] - h_right_down, :]
                center_y += (canvas.shape[0] - h_right_down) // 2
            center = (center_x, center_y)

            canvas = cv2.seamlessClone(background, canvas, mask, center, cv2.MIXED_CLONE)

        return canvas
    
    def generate_cells(self, canvas):
        # Добавляем реальные клетки
        for _ in range(np.random.randint(5, 30)):
            cell = self.cells[np.random.randint(0, len(self.cells))]
            cell = cv2.resize(cell, (100, 100))
            mask = np.full_like(cell, 255)
            
            h, w = cell.shape[:2]
            
            center_y = np.random.randint(0, self.image_size[0])
            center_x = np.random.randint(0, self.image_size[1])

            h_left_up = center_y - h // 2
            h_right_down = center_y + h // 2
            w_left_up = center_x - w // 2
            w_right_down = center_x + w // 2

            if w_left_up < 0:
                cell = cell[:, -w_left_up:]
                mask = mask[:, -w_left_up:]
                center_x -= w_left_up // 2

            elif w_right_down > canvas.shape[1]:
                cell = cell[:, :canvas.shape[1] - w_right_down]
                mask = mask[:, :canvas.shape[1] - w_right_down]
                center_x += (canvas.shape[1] - w_right_down) // 2

            if h_left_up < 0:
                cell = cell[-h_left_up:, :]
                mask = mask[-h_left_up:, :]
                center_y -= h_left_up // 2

            elif h_right_down > canvas.shape[0]:
                cell = cell[:canvas.shape[0] - h_right_down, :]
                mask = mask[:canvas.shape[0] - h_right_down, :]
                center_y += (canvas.shape[0] - h_right_down) // 2
            center = (center_x, center_y)

            canvas = cv2.seamlessClone(cell, canvas, mask, center, cv2.MIXED_CLONE)
        
        # Добавляем искусственные клетки в виде окружностей
        for _ in range(np.random.randint(3, 15)):
            radius = np.random.randint(20, 50)
            circle_size = radius * 2 + 10
            
            # Проверяем, что круг полностью поместится в изображение
            max_x = self.image_size[1] - radius - circle_size // 2
            max_y = self.image_size[0] - radius - circle_size // 2
            min_x = radius + circle_size // 2
            min_y = radius + circle_size // 2
            
            if max_x <= min_x or max_y <= min_y:
                continue
                
            center_x = np.random.randint(min_x, max_x)
            center_y = np.random.randint(min_y, max_y)
            
            circle_img = np.full((circle_size, circle_size, 3), 255, dtype=np.uint8)
            
            color = (
                np.random.randint(100, 255),
                np.random.randint(0, 100),
                np.random.randint(150, 255)
            )
            
            center_circle = (circle_size // 2, circle_size // 2)
            cv2.circle(circle_img, center_circle, radius, color, -1)
            
            mask = np.full_like(circle_img, 255)
            
            canvas = cv2.seamlessClone(circle_img, canvas, mask, (center_x, center_y), cv2.MIXED_CLONE)
                
        return canvas
    
    def __getitem__(self, idx):
        """
        Генерирует и возвращает изображение по индексу
        
        Args:
            idx (int): Индекс изображения
            
        Returns:
            numpy.ndarray: Сгенерированное изображение с примененными аугментациями
        """
        background = self.generate_background()
        result = self.generate_cells(background)
        result = ImageAugmentation.apply_augmentations(result, self.augment)
        
        return result
