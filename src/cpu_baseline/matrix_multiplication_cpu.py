import numpy as np
import time

def multiply_matrices(a, b):
    """
    Умножение матриц с использованием встроенных функций NumPy.
    """
    return np.dot(a, b)  # матричное умножение через NumPy

def multiply_matrices_sequential(a, b, c, n):
    """
    Последовательное умножение матриц.
    """
    for i in range(n):
        for j in range(n):
            c[i, j] = 0  # обнуляем текущий элемент результирующей матрицы
            for k in range(n):
                c[i, j] += a[i, k] * b[k, j]  # вычисляем элемент в строке i и столбце j
    return c

def main():
    n = 10_000  # размерность квадратных матриц (например, 500x500)
    a = np.random.rand(n, n).astype(np.float32)  # случайная матрица a
    b = np.random.rand(n, n).astype(np.float32)  # случайная матрица b
    c = np.zeros((n, n), dtype=np.float32)  # результирующая матрица c
    print(f"matrix size: {n}x{n}")

    # Умножение матриц с использованием NumPy
    start = time.time()
    result = multiply_matrices(a, b)
    end = time.time()

    print(f"Matrix multiplication result (NumPy): {result}")
    print(f"Performing matrix multiplication with NumPy on the CPU in: {end - start:.6f} seconds\n")

    # Последовательное умножение матриц
    #start = time.time()
    #result = multiply_matrices_sequential(a, b, c, n)
    #end = time.time()

    #print(f"Matrix multiplication result (sequential): {result}")
    #print(f"Performing matrix multiplication sequentially on the CPU in: {end - start:.6f} seconds")

if __name__ == "__main__":
    main()
