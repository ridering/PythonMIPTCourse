import logging
import os
from optimizer import MonteCarloGradientOptimizer, GridSearchOptimizer, PeanoOptimizer
from cell_finder import CellFinder

def setup_logging():
    """Настраивает логирование"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cell_finder.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # Настраиваем логирование
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Создаем директорию для результатов
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Создаем оптимизаторы
    monte_carlo_optimizer = MonteCarloGradientOptimizer(
        n_samples=10000,
        width_range=(80, 120),
        max_iterations=1000,
        tolerance=1e-6
    )
    
    grid_optimizer = GridSearchOptimizer(
        grid_step=5.0,
        width_range=(80, 120),
        width_steps=10,
        max_iterations=1000
    )
    
    peano_optimizer = PeanoOptimizer(
        width_range=(80, 120),
        grid_size=20,
        max_iterations=1000,
        tolerance=1e-6
    )
    
    # Создаем поисковик клеток с разными оптимизаторами
    logger.info("Запуск поиска с оптимизатором Монте-Карло + градиентный спуск")
    mc_finder = CellFinder(
        optimizer=monte_carlo_optimizer,
        results_dir=os.path.join(results_dir, "monte_carlo")
    )
    mc_results = mc_finder.find_cells()
    mc_finder.visualize_results(mc_results, max_results=13)
    
    logger.info("Запуск поиска с оптимизатором перебора по сетке")
    grid_finder = CellFinder(
        optimizer=grid_optimizer,
        results_dir=os.path.join(results_dir, "grid_search")
    )
    grid_results = grid_finder.find_cells()
    grid_finder.visualize_results(grid_results, max_results=13)
    
    logger.info("Запуск поиска с оптимизатором развертки Пеано")
    peano_finder = CellFinder(
        optimizer=peano_optimizer,
        results_dir=os.path.join(results_dir, "peano")
    )
    peano_results = peano_finder.find_cells()
    peano_finder.visualize_results(peano_results, max_results=13)
    
    logger.info("Поиск завершен")

if __name__ == "__main__":
    main() 