import os
from utils.dataset_generator import YOLODatasetGenerator

def generate_dataset():
    # Пути к директориям
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'dataset')
    
    # Генерируем датасет
    generator = YOLODatasetGenerator(
        data_dir=data_dir,
        output_dir=output_dir,
        image_size=(640, 640),
        num_samples=1000,  # Количество изображений для обучения
        train_ratio=0.9    # 80% для тренировки, 20% для валидации
    )
    train_path = generator.generate_dataset()
    
    # Создаем конфигурацию для обучения
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {output_dir}
train: train/images
val: val/images

nc: 1  # количество классов
names: ['blood_cell']  # имена классов
""")
    
    print(f"Датасет сгенерирован в директории: {output_dir}")
    print(f"Конфигурация сохранена в: {yaml_path}")
    return yaml_path

if __name__ == '__main__':
    generate_dataset() 