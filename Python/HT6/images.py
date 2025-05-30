from abc import ABC, abstractmethod

import numpy as np


class Image(ABC):
    @property
    @abstractmethod
    def width(self) -> int:
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        ...


class BinaryImage(Image):
    def __init__(self, image: np.ndarray) -> None:
        super().__init__()

        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 2

        self.pixels: np.ndarray = np.array(image, dtype=bool)

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.pixels.shape[1]


class HalftoneImage(Image):
    def __init__(self, image: np.ndarray) -> None:
        super().__init__()

        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 2
        assert 0 <= image.min() <= image.max() <= 255

        self.pixels: np.ndarray = np.array(image, dtype=int)

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.pixels.shape[1]


class ColourImage(Image):
    def __init__(self, image: np.ndarray) -> None:
        super().__init__()

        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        assert 0 <= image.min() <= image.max() <= 255

        self.pixels: np.ndarray = np.array(image, dtype=int)

    @property
    def depth(self) -> int:
        return self.pixels.shape[0]

    @property
    def height(self) -> int:
        return self.pixels.shape[1]

    @property
    def width(self) -> int:
        return self.pixels.shape[2]
