from numba import cuda
import numpy as np
import time

@cuda.jit
def matrix_multiply_cuda(a, b, c):
    """
    Умножение матриц a и b, результат сохраняется в C.
    """
    row, col = cuda.grid(2)  # Получаем текущие индексы строки и столбца

    if row < c.shape[0] and col < c.shape[1]:
        temp = 0.0
        for k in range(a.shape[1]):
            temp += a[row, k] * b[k, col]
        c[row, col] = temp

def main():
    # Размерность матриц
    n = 10_000
    print(f"Matrix size: {n}x{n}")

    # Создание случайных матриц
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    c = np.zeros((n, n), dtype=np.float32)

    # Перенос данных на устройство (GPU)
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array((n, n), dtype=np.float32)

    # Определение параметров сетки и блоков
    threads_per_block = (16, 16)  # Блоки 16x16 потоков
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    print(f"Threads per block: {threads_per_block}")
    print(f"Blocks per grid: {blocks_per_grid}")

    # Запуск ядра и измерение времени
    start = time.time()
    matrix_multiply_cuda[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()
    end = time.time()

    # Перенос результата обратно на хост (CPU)
    result = d_c.copy_to_host()

    print(f"Matrix multiplication result: {result}")
    print(f"Parallel execution using Numba CUDA on the GPU in: {end - start:.6f} seconds")

if __name__ == "__main__":
    main()
