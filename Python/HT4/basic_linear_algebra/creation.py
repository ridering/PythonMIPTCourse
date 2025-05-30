import random


def create_vector(n: int, low=0, high=1) -> list[float]:
    return [random.uniform(low, high) for _ in range(n)]


def create_vector_gauss(n: int, m=0, s=1) -> list[float]:
    return [random.gauss(m, s) for _ in range(n)]


def create_matrix(n: int, m: int, low=0, high=1) -> list[list[float]]:
    return [[random.uniform(low, high) for _ in range(m)] for _ in range(n)]
