import matplotlib.pyplot as plt

# Основная функция для построения графиков
def plot_comparison_all():
    # Массивы с временем выполнения для сложения массивов (время для каждого размера n)
    add_sizes = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

    # Для сложения массивов
    serial_add = [0.275297, 2.780581, 28.461764, 325.099969]  # значения для последовательного
    numpy_add = [0.000998, 0.010971, 0.122988, 33.766590]  # значения времени для NumPy
    numba_add = [0.000965, 0.001993, 0.014928, 34.454805]  # значения для Numba
    pycuda_add = [0.000703, 0.000997, 0.001964, 3.445480]  # значения для PyCUDA

    # Массивы с временем выполнения для умножения матриц (время для каждого размера n)
    mult_sizes = [100, 300, 500, 700, 1000]

    # Для умножения матриц
    serial_mult = [0.484680, 14.283885, 62.531973, 176.968623, 537.071679]  # значения для последовательного
    numpy_mult = [0.000997, 0.000994, 0.001971, 0.003976, 0.007946]  # значения для NumPy
    numba_mult = [0.0, 0.203690, 0.206571, 0.219358, 0.236562]  # значения для Numba
    pycuda_mult = [0.0, 0.000964, 0.000998, 0.002992, 0.005981]  # значения для PyCUDA

    # График для сложения массивов
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(add_sizes, serial_add, label="Serial", marker='o')
    plt.plot(add_sizes, numpy_add, label="NumPy", marker='o')
    plt.plot(add_sizes, numba_add, label="Numba", marker='o')
    plt.plot(add_sizes, pycuda_add, label="PyCUDA", marker='o')
    plt.xlabel('Grid resolution n')
    plt.ylabel('Runtime (s)')
    plt.title('Array Addition Comparison')
    plt.legend()

    # График для умножения матриц
    plt.subplot(1, 2, 2)
    plt.plot(mult_sizes, serial_mult, label="Serial", marker='o')
    plt.plot(mult_sizes, numpy_mult, label="NumPy", marker='o')
    plt.plot(mult_sizes, numba_mult, label="Numba", marker='o')
    plt.plot(mult_sizes, pycuda_mult, label="PyCUDA", marker='o')
    plt.xlabel('Grid resolution n')
    plt.ylabel('Runtime (s)')
    plt.title('Matrix Multiplication Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_comparison_parallel():
    # Массивы с временем выполнения для сложения массивов (время для каждого размера n)
    add_sizes = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

    # Для сложения массивов
    #serial_add = [0.275297, 2.780581, 28.461764, 325.099969] # значения для последовательного
    numpy_add = [0.000998, 0.010971, 0.122988, 33.766590]  # значения времени для NumPy
    numba_add = [0.000965, 0.001993, 0.014928, 34.454805]  # значения для Numba
    pycuda_add = [0.000703, 0.000997, 0.001964, 3.445480]  # значения для PyCUDA

    # Массивы с временем выполнения для умножения матриц (время для каждого размера n)
    mult_sizes = [100, 300, 500, 700, 1000, 10_000]

    # Для умножения матриц
    #serial_mult = [0.484680, 14.283885, 62.531973, 176.968623, 537.071679, 1611.210504] # значения для последовательного
    numpy_mult = [0.000997, 0.000994, 0.001971, 0.003976, 0.007946, 7.711419]  # значения для NumPy
    numba_mult = [0.0, 0.203690, 0.206571, 0.219358, 0.236562, 25.868758]  # значения для Numba
    pycuda_mult = [0.0, 0.000964, 0.000998, 0.002992, 0.005981, 9.718075]  # значения для PyCUDA

    # График для сложения массивов
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    #plt.plot(add_sizes, serial_add, label="Serial", marker='o')
    plt.plot(add_sizes, numpy_add, label="NumPy", marker='o')
    plt.plot(add_sizes, numba_add, label="Numba", marker='o')
    plt.plot(add_sizes, pycuda_add, label="PyCUDA", marker='o')
    plt.xlabel('Grid resolution n')
    plt.ylabel('Runtime (s)')
    plt.title('Array Addition Comparison')
    plt.legend()

    # График для умножения матриц
    plt.subplot(1, 2, 2)
    #plt.plot(mult_sizes, serial_mult, label="Serial", marker='o')
    plt.plot(mult_sizes, numpy_mult, label="NumPy", marker='o')
    plt.plot(mult_sizes, numba_mult, label="Numba", marker='o')
    plt.plot(mult_sizes, pycuda_mult, label="PyCUDA", marker='o')
    plt.xlabel('Grid resolution n')
    plt.ylabel('Runtime (s)')
    plt.title('Matrix Multiplication Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # plot_comparison_all()
    plot_comparison_parallel()
