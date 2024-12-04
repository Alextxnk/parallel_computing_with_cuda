from numba import cuda  # Библиотека Nvidia для работы с GPU
import numpy as np
import time

@cuda.jit('void(float32[:], float32[:], float32[:])')  # Динамический компилятор Cuda
def cuda_addition(a, b, c):
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Индекс текущего потока
    i = cuda.grid(1) # поиск индекса выполняется аналогично с помощью библиотеки
    if i >= c.size:
        return
    c[i] = a[i] + b[i]

def main():
    # Создание массивов на хосте (CPU)
    n = 10_000_000
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    print(f"input size: {n}")

    # Перенос данных на устройство (GPU) в глобальную память
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(a)  # Создание пустого массива

    # Получение информации об устройстве
    device = cuda.get_current_device()
    print(f"device name: {device.name.decode()}")

    # Определение параметров запуска ядра
    # threads_per_block = device.WARP_SIZE # Количество потоков в блоке: стандартно 32 потока
    threads_per_block = 256 # Задаем значения кратные 32
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block # Вычисление количества блоков в сетке
    print(f"threads per block: {threads_per_block}")
    print(f"blocks per grid: {blocks_per_grid}")

    start = time.time()
    cuda_addition[blocks_per_grid, threads_per_block](d_a, d_b, d_c)  # Вызов ядра
    cuda.synchronize()  # Синхронизация потоков для точного измерения
    end = time.time()

    # Перенос результата обратно на хост (CPU)
    result = d_c.copy_to_host()

    print(f"result: {result}")
    print(f"parallel execution using Numba CUDA on the GPU in: {end - start:.6f} seconds")

if __name__ == "__main__":
    main()
