import pycuda.driver as cuda
import pycuda.autoinit  # Автоматическая инициализация контекста
from pycuda.compiler import SourceModule
import numpy as np
import time

def main():
    # CUDA ядро для умножения матриц
    kernel_code = """
    __global__ void matrix_multiply(float *a, float *b, float *c, int n) {
        int row = blockIdx.y * blockDim.y + threadIdx.y; // Индекс строки
        int col = blockIdx.x * blockDim.x + threadIdx.x; // Индекс столбца

        if (row < n && col < n) {
            float value = 0.0;
            for (int k = 0; k < n; ++k) {
                value += a[row * n + k] * b[k * n + col];
            }
            c[row * n + col] = value;
        }
    }
    """

    # Компиляция ядра
    mod = SourceModule(kernel_code, options=["-lineinfo", "-w"])

    # Получение ссылки на функцию ядра
    matrix_multiply = mod.get_function("matrix_multiply")

    # Размер матриц
    n = 10_000
    print(f"Matrix size: {n}x{n}")

    # Создание случайных матриц
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    c = np.zeros((n, n), dtype=np.float32)

    # Передача данных на устройство (GPU)
    d_a = cuda.mem_alloc(a.nbytes)
    d_b = cuda.mem_alloc(b.nbytes)
    d_c = cuda.mem_alloc(c.nbytes)

    cuda.memcpy_htod(d_a, a)
    cuda.memcpy_htod(d_b, b)

    # Настройка параметров сетки и блоков
    threads_per_block = (16, 16)  # 16x16 потоков в блоке
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    print(f"Threads per block: {threads_per_block}")
    print(f"Blocks per grid: {blocks_per_grid}")

    # Запуск ядра
    start = time.time()
    matrix_multiply(
        d_a, d_b, d_c,
        np.int32(n),
        block=(threads_per_block[0], threads_per_block[1], 1),
        grid=(blocks_per_grid[0], blocks_per_grid[1], 1)
    )
    cuda.Context.synchronize()  # Синхронизация для точного измерения времени
    end = time.time()

    # Перенос результата обратно на хост (CPU)
    cuda.memcpy_dtoh(c, d_c)

    print(f"Result: {c}")
    print(f"Matrix multiplication using PyCUDA on the GPU in: {end - start:.6f} seconds")

if __name__ == "__main__":
    main()
