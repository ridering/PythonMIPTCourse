import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "experiments.db"

def init_db():
    """Создаёт таблицу для хранения экспериментов, если её ещё нет."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            real_data_path TEXT,
            gen_params TEXT,
            method1_result INTEGER,
            method2_result INTEGER,
            method3_result INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_experiment(date, real_data_path, gen_params, method1, method2, method3):
    """Сохраняет результат эксперимента в БД."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO experiments (date, real_data_path, gen_params, method1_result, method2_result, method3_result)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (date, real_data_path, gen_params, method1, method2, method3))
    conn.commit()
    conn.close()

def load_experiments():
    """Загружает все эксперименты в виде pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM experiments", conn)
    conn.close()
    return df

def load_experiment_by_id(exp_id):
    """Загружает эксперимент по id."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM experiments WHERE id = ?", conn, params=(exp_id,))
    conn.close()
    return df

if __name__ == "__main__":
    init_db()
    # Пример добавления эксперимента
    save_experiment(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        real_data_path="path/to/image.jpg",
        gen_params="{'cells': 20, 'noise': 0.1}",
        method1=18,
        method2=20,
        method3=19
    )
    # Пример вывода всех экспериментов
    print(load_experiments())
