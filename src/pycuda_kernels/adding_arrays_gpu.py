import pycuda.driver as cuda
import pycuda.autoinit  # Автоматическая инициализация контекста
from pycuda.compiler import SourceModule
import numpy as np
import time

def main():
    # Определение CUDA ядра на языке C++
    kernel_code = """
    __global__ void add_arrays(float *a, float *b, float *c, int n) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    # Компиляция ядра
    # SourceModule компилирует строку `kernel_code` и делает функции доступными для вызова из Python
    mod = SourceModule(kernel_code, options=["-lineinfo"])

    # Получение ссылки на функцию ядра `add_arrays` для последующего вызова
    add_arrays = mod.get_function("add_arrays")

    # Создание данных
    n = 1_000_000
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    c = np.zeros_like(a)
    print(f"input size: {n}")

    # Передача данных на устройство
    # Выделение памяти на GPU для массивов
    d_a = cuda.mem_alloc(a.nbytes)
    d_b = cuda.mem_alloc(b.nbytes)
    d_c = cuda.mem_alloc(c.nbytes)

    # Хост -> устройство
    # Копирование данных с хоста (CPU) на устройство (GPU)
    cuda.memcpy_htod(d_a, a)
    cuda.memcpy_htod(d_b, b)

    # Настройка количества потоков и блоков
    threads_per_block = 1024
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block # Вычисление количества блоков в сетке
    print(f"threads per block: {threads_per_block}")
    print(f"blocks per grid: {blocks_per_grid}")

    # Запуск CUDA ядра
    # Вызываем функцию `add_arrays` на GPU с указанными параметрами сетки и блоков
    start = time.time()
    add_arrays(d_a, d_b, d_c, np.int32(n), block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1, 1))
    end = time.time()

    # Перенос результата обратно на хост
    # Копирование результата с устройства (GPU) на хост (CPU)
    cuda.memcpy_dtoh(c, d_c)
    print(f"result: {c}")
    print(f"parallel execution using PyCUDA on the GPU in: {end - start:.10f} seconds")

if __name__ == "__main__":
    main()
