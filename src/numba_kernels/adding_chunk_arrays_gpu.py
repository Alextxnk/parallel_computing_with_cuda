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
    n = 1_000_000_000
    chunk_size = 10_000_000  # Размер чанка, подходящий для GPU
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    c = np.zeros_like(a, dtype=np.float32)  # Массив для результата
    print(f"Input size: {n}, chunk size: {chunk_size}")

    # Получение информации об устройстве
    device = cuda.get_current_device()
    print(f"device name: {device.name.decode()}")
    #print(f"Total memory: {device.total_memory / 1e9:.2f} GB")
    print(f"Max threads per block: {device.MAX_THREADS_PER_BLOCK}")

    # Определение параметров запуска ядра
    # threads_per_block = device.WARP_SIZE # Количество потоков в блоке: стандартно 32 потока
    threads_per_block = 1024 # Задаем значения кратные 32
    blocks_per_grid = (chunk_size + threads_per_block - 1) // threads_per_block # Вычисление количества блоков в сетке
    print(f"threads per block: {threads_per_block}")
    print(f"blocks per grid: {blocks_per_grid}")

    start = time.time()
    # Обработка данных чанками
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)  # Убедиться, что не выйдем за пределы массива

        # Перенос чанков на устройство
        d_a = cuda.to_device(a[i:end])
        d_b = cuda.to_device(b[i:end])
        d_c = cuda.device_array_like(d_a)

        # Вызов CUDA ядра
        cuda_addition[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
        cuda.synchronize()  # Синхронизация потоков для точного измерения времени

        # Копирование результата обратно на хост
        # Перенос результата обратно на хост (CPU)
        c[i:end] = d_c.copy_to_host()

    end = time.time()

    print(f"result: {c}")
    print(f"parallel execution using Numba CUDA on the GPU in: {end - start:.6f} seconds")

if __name__ == "__main__":
    main()
