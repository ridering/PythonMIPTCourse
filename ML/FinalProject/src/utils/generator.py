import os
import cv2
import numpy as np

class BloodCellGenerator:
    def __init__(self, data_dir, image_size=(1024, 1024)):
        self.cells_dir = os.path.join(data_dir, "cells")
        self.backgrounds_dir = os.path.join(data_dir, "backgrounds")
        self.image_size = image_size

        self.cells = self.load_cells()
        self.backgrounds = self.load_backgrounds()
    
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
        bboxes = []  # Список для хранения координат клеток
        
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
                w = cell.shape[1]

            elif w_right_down > canvas.shape[1]:
                cell = cell[:, :canvas.shape[1] - w_right_down]
                mask = mask[:, :canvas.shape[1] - w_right_down]
                center_x += (canvas.shape[1] - w_right_down) // 2
                w = cell.shape[1]

            if h_left_up < 0:
                cell = cell[-h_left_up:, :]
                mask = mask[-h_left_up:, :]
                center_y -= h_left_up // 2
                h = cell.shape[0]

            elif h_right_down > canvas.shape[0]:
                cell = cell[:canvas.shape[0] - h_right_down, :]
                mask = mask[:canvas.shape[0] - h_right_down, :]
                center_y += (canvas.shape[0] - h_right_down) // 2
                h = cell.shape[0]

            center = (center_x, center_y)
            canvas = cv2.seamlessClone(cell, canvas, mask, center, cv2.MIXED_CLONE)
            
            # Добавляем координаты клетки в список
            bboxes.append({
                'center_x': center_x,
                'center_y': center_y,
                'width': w,
                'height': h
            })
                
        return canvas, bboxes
    
    def generate_image(self, return_bboxes=False):
        background = self.generate_background()
        if return_bboxes:
            image, bboxes = self.generate_cells(background)
            return image, bboxes
        else:
            image, _ = self.generate_cells(background)
            return image 