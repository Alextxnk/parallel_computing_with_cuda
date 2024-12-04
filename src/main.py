import matplotlib.pyplot as plt

# Массивы с временем выполнения для каждого метода (время для каждого размера n)
sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

# Для сложения массивов
numpy_add = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # Примерные значения времени для NumPy
serial_add = [0.02, 0.08, 0.15, 0.25, 0.4, 0.6, 0.9, 1.2]  # Примерные значения для последовательного
numba_add = [0.015, 0.07, 0.12, 0.22, 0.35, 0.55, 0.8, 1.1]  # Примерные значения для Numba
pycuda_add = [0.005, 0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5]  # Примерные значения для PyCUDA

# Для умножения матриц
numpy_mult = [0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Примерные значения для NumPy
serial_mult = [0.1, 0.3, 0.7, 1.3, 2.0, 3.0, 4.0, 5.0]  # Примерные значения для последовательного
numba_mult = [0.07, 0.25, 0.6, 1.1, 1.7, 2.4, 3.2, 4.0]  # Примерные значения для Numba
pycuda_mult = [0.02, 0.1, 0.3, 0.6, 1.0, 1.4, 1.9, 2.5]  # Примерные значения для PyCUDA

# Основная функция для построения графиков
def plot_comparison():
    # График для сложения массивов
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sizes, numpy_add, label="NumPy", marker='o')
    plt.plot(sizes, serial_add, label="Serial", marker='o')
    plt.plot(sizes, numba_add, label="Numba", marker='o')
    plt.plot(sizes, pycuda_add, label="PyCUDA", marker='o')
    plt.xlabel('Grid resolution n')
    plt.ylabel('Runtime (s)')
    plt.title('Array Addition Comparison')
    plt.legend()

    # График для умножения матриц
    plt.subplot(1, 2, 2)
    plt.plot(sizes, numpy_mult, label="NumPy", marker='o')
    plt.plot(sizes, serial_mult, label="Serial", marker='o')
    plt.plot(sizes, numba_mult, label="Numba", marker='o')
    plt.plot(sizes, pycuda_mult, label="PyCUDA", marker='o')
    plt.xlabel('Grid resolution n')
    plt.ylabel('Runtime (s)')
    plt.title('Matrix Multiplication Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_comparison()
