import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from utils.generator import BloodCellGenerator
from models.ml_model import MLModel
from models.clustering_model import ClusteringModel
from models.cnn_model import CNNModel
from experiment_db import load_experiments
from preprocessing.filters import ImageFilters

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.current_image = None
        self.setup_ui()
        
        # Инициализация генератора
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        self.generator = BloodCellGenerator(data_dir, image_size=(640, 640))
        
        # Инициализация моделей
        self.ml_model = MLModel()
        self.clustering_model = ClusteringModel()
        self.cnn_model = CNNModel()
        
    def setup_ui(self):
        # Создание основного контейнера
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Область для отображения изображения
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Изображение", padding="5")
        self.image_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Кнопки загрузки изображения
        self.load_buttons_frame = ttk.Frame(self.main_frame)
        self.load_buttons_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(self.load_buttons_frame, text="Загрузить изображение", 
                  command=self.load_image).grid(row=0, column=0, padx=5)
        ttk.Button(self.load_buttons_frame, text="Сгенерировать изображение", 
                  command=self.generate_image).grid(row=0, column=1, padx=5)
        ttk.Button(self.load_buttons_frame, text="Таблица экспериментов", 
                  command=self.show_experiments_table).grid(row=0, column=2, padx=5)
        
        # Выбор фильтра
        self.filter_frame = ttk.LabelFrame(self.main_frame, text="Фильтры", padding="5")
        self.filter_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.filter_var = tk.StringVar()
        filters = ["Без фильтра", "Размытие", "Резкость"]
        self.filter_combo = ttk.Combobox(self.filter_frame, textvariable=self.filter_var, 
                                       values=filters, state="readonly")
        self.filter_combo.grid(row=0, column=0, padx=5)
        self.filter_combo.set(filters[0])
        
        ttk.Button(self.filter_frame, text="Применить фильтр", 
                  command=self.apply_filter).grid(row=0, column=1, padx=5)
        
        # Выбор метода анализа
        self.method_frame = ttk.LabelFrame(self.main_frame, text="Метод анализа", padding="5")
        self.method_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.method_var = tk.StringVar()
        methods = ["Классическое ML", "Кластеризация", "Свёрточная сеть"]
        self.method_combo = ttk.Combobox(self.method_frame, textvariable=self.method_var, 
                                       values=methods, state="readonly")
        self.method_combo.grid(row=0, column=0, padx=5)
        self.method_combo.set(methods[0])
        
        ttk.Button(self.method_frame, text="Анализировать", 
                  command=self.analyze_image).grid(row=0, column=1, padx=5)
        
        # Область результатов
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Результаты", padding="5")
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.result_label = ttk.Label(self.result_frame, text="")
        self.result_label.grid(row=0, column=0, padx=5)
        
        # Настройка весов для растяжения
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
    
    def show_experiments_table(self):
        df = load_experiments()
        window = tk.Toplevel(self.root)
        window.title("Результаты экспериментов")
        frame = ttk.Frame(window)
        frame.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(frame, columns=list(df.columns), show='headings')
        tree.pack(fill=tk.BOTH, expand=True)
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center')
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=list(row))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def display_image(self, image):
        """Отображает изображение в GUI"""
        if image is None:
            return
            
        # Конвертируем BGR в RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Изменяем размер изображения для отображения
        height, width = image.shape[:2]
        max_size = 800
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            
        # Конвертируем в формат для tkinter
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        
        # Обновляем изображение
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Сохраняем ссылку
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
    
    def generate_image(self):
        self.current_image = self.generator.generate_image()
        self.display_image(self.current_image)
    
    def apply_filter(self):
        if self.current_image is None:
            return
            
        filter_name = self.filter_var.get()
        if filter_name == "Без фильтра":
            return
        
        image = self.current_image
        if filter_name == "Размытие":
            filtered = ImageFilters.blur(image)
        elif filter_name == "Резкость":
            filtered = ImageFilters.sharpen(image)
        else:
            filtered = image
        self.display_image(filtered)
        self.current_image = filtered
    
    def analyze_image(self):
        if self.current_image is None:
            return
            
        method = self.method_var.get()
        count = 0
        
        # try:
        if method == "Классическое ML":
            count = self.ml_model.predict(self.current_image)
        elif method == "Кластеризация":
            count = self.clustering_model.predict(self.current_image)
        elif method == "Свёрточная сеть":
            count = self.cnn_model.predict(self.current_image)
            
        # Обновляем результат
        self.result_label.configure(
            text=f"Найдено клеток: {count}",
            font=("Arial", 12)
        )
            
        # except Exception as e:
        #     self.result_label.configure(
        #         text=f"Ошибка при анализе: {str(e)}",
        #         font=("Arial", 12),
        #         foreground="red"
        #     ) 