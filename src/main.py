from numba import cuda
import numpy as np

# Вывод информации о GPU
print("GPU Info:", cuda.gpus)
print(cuda.detect())

# Простая функция для запуска на GPU
@cuda.jit
def add_kernel(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

# Создание данных
n = 1000
a = np.arange(n, dtype=np.float32)
b = np.arange(n, dtype=np.float32)
c = np.zeros(n, dtype=np.float32)

# Копирование данных на устройство
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(c)

# Запуск ядра
# threads_per_block = 256
threads_per_block = 16
blocks = (n + threads_per_block - 1) // threads_per_block
add_kernel[blocks, threads_per_block](d_a, d_b, d_c) # вызов ядра

# Копирование результата обратно на хост
result = d_c.copy_to_host()
print("Result:", result[:10])  # Печать первых 10 элементов
