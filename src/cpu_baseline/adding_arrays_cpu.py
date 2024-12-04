import numpy as np
import time

def adding_vectors(a, b):
    return a + b  # vector addition with numpy

def adding_arrays (a, b, c, n):
    for i in range(n):
        c[i] = a[i] + b[i]
    return c

def main():
    n = 10_000_000
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    print(f"input size: {n}")

    start = time.time()
    result = adding_vectors(a, b)
    end = time.time()

    print(f"adding vectors result: {result}")
    print(f"performing adding vectors on the CPU in: {end - start:.6f} seconds\n")

    start = time.time()
    result = adding_arrays(a, b, c, n)
    end = time.time()

    print(f"adding arrays result: {result}")
    print(f"performing adding arrays on the CPU in: {end - start:.6f} seconds")

if __name__ == "__main__":
    main()
