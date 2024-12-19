import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Hist(ABC):
    @abstractmethod
    def read(self, file_path):
        pass

    @abstractmethod
    def write(self, file_path):
        pass

    @abstractmethod
    def draw_hist(self, file_path):
        pass


class Hist_from_csv(Hist):
    def read(self, file_path):
        self.__data = pd.read_csv(file_path)

    def write(self, file_path):
        self.__data.to_csv(file_path, index=False)

    def draw_hist(self, file_path):
        plt.hist(self.__data, color='skyblue', edgecolor='black')
        plt.savefig(file_path)


class Hist_from_xlsx(Hist):
    def read(self, file_path):
        self.__data = pd.read_excel(file_path)

    def write(self, file_path):
        self.__data.to_excel(file_path, index=False)

    def draw_hist(self, file_path):
        plt.hist(self.__data, color='skyblue', edgecolor='black')
        plt.savefig(file_path)
