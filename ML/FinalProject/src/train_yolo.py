import os
from ultralytics import YOLO

def train_yolo():
    # Пути к директориям
    base_dir = os.path.dirname(os.path.dirname(__file__))
    yaml_path = os.path.join(base_dir, 'dataset', 'dataset.yaml')
    
    # Проверяем существование датасета
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Файл конфигурации датасета не найден: {yaml_path}\n"
            "Сначала запустите generate_dataset.py для создания датасета"
        )
    
    # Проверяем наличие тренировочных и валидационных данных
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    val_dir = os.path.join(base_dir, 'dataset', 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            "Директории с тренировочными или валидационными данными не найдены.\n"
            "Сначала запустите generate_dataset.py для создания датасета"
        )
    
    # Загружаем предобученную модель
    model = YOLO('yolov8n.pt')
    
    # Обучаем модель
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='blood_cell_detector'
    )
    
    # Сохраняем обученную модель
    model.save(os.path.join(base_dir, 'models', 'blood_cell_detector.pt'))
    print(f"Модель сохранена в: {os.path.join(base_dir, 'models', 'blood_cell_detector.pt')}")

if __name__ == '__main__':
    train_yolo() 