'''
Сжатие изображения с потерей качества на основе МГК.
Изображение разбирается на три матрицы каждая из которых отвечает за слои из RGB,
для каждой матрицы используется сингулярное разложение, на основе которого и можно рассчитать оценку значимости "информации" в изображении.
Реализовать возможность регулирования уровня сжатия изображения
(например хочется оставить не менее 85% данных в изображении, тогда на сколько уменьшиться вес итогового изображения)
'''


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import svd
from concurrent.futures import ThreadPoolExecutor as TPE
from typing import Tuple
import time

start_time = time.time()

class ImageCompressor:
    def __init__(self, info_ratio: float = 0.8) -> None:
        self.info_ratio: float = info_ratio  # Процент сохраняемой информации

    # Метод для загрузки изображения с помощью Pillow
    def load_image(self, image_path: str) -> np.ndarray:
        image: Image.Image = Image.open(image_path).convert('RGB')
        return np.array(image) / 255.0

    # Ускоряем SVD, используя scipy.linalg.svd
    def svd_manual(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, sigma, Vt = svd(matrix, full_matrices=False)
        S: np.ndarray = np.diag(sigma)  # Преобразуем сингулярные значения в диагональную матрицу
        return U, S, Vt

    # Сжатие одной компоненты изображения
    def compress_component(self, matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        U, S, Vt = self.svd_manual(matrix)
        total_variance: float = np.sum(S ** 2)

        # Определяем нужное число сингулярных значений
        retained_variance: float = 0
        k: int = 0
        while retained_variance / total_variance < self.info_ratio and k < S.shape[0]:
            retained_variance += S[k, k] ** 2
            k += 1

        # Используем только k сингулярных значений
        U_k: np.ndarray = U[:, :k]
        S_k: np.ndarray = S[:k, :k]
        Vt_k: np.ndarray = Vt[:k, :]
        compressed_matrix: np.ndarray = U_k @ S_k @ Vt_k
        return compressed_matrix, k

    # Параллелим сжатие компонентов R, G и B
    def compress_image(self, image_path: str) -> Tuple[np.ndarray, float]:
        image: np.ndarray = self.load_image(image_path)
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Используем ThreadPoolExecutor для параллельного сжатия каналов
        with TPE(max_workers=3) as executor:
            R_future = executor.submit(self.compress_component, R)
            G_future = executor.submit(self.compress_component, G)
            B_future = executor.submit(self.compress_component, B)

            # Получаем результаты выполнения
            R_compressed, k_r = R_future.result()
            G_compressed, k_g = G_future.result()
            B_compressed, k_b = B_future.result()

        # Собираем сжатое изображение
        compressed_image: np.ndarray = np.zeros_like(image)
        compressed_image[:, :, 0] = np.clip(R_compressed, 0, 1)
        compressed_image[:, :, 1] = np.clip(G_compressed, 0, 1)
        compressed_image[:, :, 2] = np.clip(B_compressed, 0, 1)

        # Коэффициент сжатия
        original_size: int = R.size + G.size + B.size
        compressed_size: int = (
            k_r * (R.shape[0] + R.shape[1]) +
            k_g * (G.shape[0] + G.shape[1]) +
            k_b * (B.shape[0] + B.shape[1])
        )
        compression_ratio: float = original_size / compressed_size

        return compressed_image, compression_ratio

    # Метод для визуализации исходного и сжатого изображения
    def show_compressed_image(self, image_path: str) -> None:
        original_image: np.ndarray = self.load_image(image_path)
        compressed_image: np.ndarray
        compression_ratio: float
        compressed_image, compression_ratio = self.compress_image(image_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_image)
        axes[0].set_title("Исходное изображение")
        axes[0].axis('off')
        axes[1].imshow(compressed_image)
        axes[1].set_title(f"Сжатое изображение ({self.info_ratio * 100:.1f}% информации)")
        axes[1].axis('off')
        plt.show()

        print(f"Коэффициент сжатия: {compression_ratio:.2f}")

        end_time = time.time()
        print(f"Время выполнения скрипта: {int(end_time - start_time)} секунд")

# Использование класса
compressor = ImageCompressor(info_ratio=0.98)  # Создаем объект с нужным уровнем информации
compressor.show_compressed_image('path_to_image.jpg')
