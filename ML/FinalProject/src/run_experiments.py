import os
from datetime import datetime
import cv2
from experiment_db import init_db, save_experiment
from models.cnn_model import CNNModel
from models.ml_model import MLModel
from models.clustering_model import ClusteringModel
from utils.generator import BloodCellGenerator

def run_methods_on_image(image_path, ml_model, clustering_model, cnn_model):
    """Запускает все методы на одном изображении и возвращает результаты."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    method1_result = ml_model.predict(image)
    method2_result = clustering_model.predict(image)
    method3_result = cnn_model.predict(image)

    return method1_result, method2_result, method3_result

def batch_process_images(images_dir):
    """Массово обрабатывает все изображения в папке и сохраняет результаты в БД."""
    init_db()

    files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    ml_model = MLModel()
    clustering_model = ClusteringModel()
    cnn_model = CNNModel()
    for idx, filename in enumerate(files, 1):
        image_path = os.path.join(images_dir, filename)
        try:
            m1, m2, m3 = run_methods_on_image(image_path, ml_model, clustering_model, cnn_model)
            save_experiment(
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                real_data_path=image_path,
                gen_params="",
                method1=m1,
                method2=m2,
                method3=m3
            )
            print(f"[{idx}/{len(files)}] {filename}: OK")
        except Exception as e:
            print(f"[{idx}/{len(files)}] {filename}: ERROR — {e}")

def process_generated_images(num_images):
    """Генерирует изображения, считает клетки, прогоняет через все модели и сохраняет результат в БД."""
    init_db()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    generator = BloodCellGenerator(data_dir)
    ml_model = MLModel()
    clustering_model = ClusteringModel()
    cnn_model = CNNModel()
    for i in range(num_images):
        image, bboxes = generator.generate_image(return_bboxes=True)
        num_cells = len(bboxes)
        m1 = ml_model.predict(image)
        m2 = clustering_model.predict(image)
        m3 = cnn_model.predict(image)
        save_experiment(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            real_data_path="",
            gen_params=str(num_cells),
            method1=m1,
            method2=m2,
            method3=m3
        )
        print(f"[Сгенерировано {i+1}/{num_images}]: OK")
        # print(f"[Сгенерировано {i+1}/{num_images}] Клеток: {num_cells} | ML: {m1} | Clust: {m2} | CNN: {m3}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    images_dir = os.path.join(base_dir, "dataset", "val", "images")

    # Эксперименты над реальными изображениями
    batch_process_images(images_dir)
    # Эксперименты над сгенерированными изображениями
    process_generated_images(num_images=50) 