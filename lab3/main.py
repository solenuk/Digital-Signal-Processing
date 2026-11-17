import numpy as np
import time
from fractions import Fraction
from lab2 import calc_Ck

n = 22
N1 = 10 + n
# N1 = 2**15
np.random.seed(N1)
f = (np.random.random((1, N1)) * 10)[0]

def fft(x: np.ndarray, N: int, s: int = 1) -> np.ndarray:
    if N == 1:
        return np.array([x[0]], dtype=complex)

    X = np.empty(N, dtype=complex)
    X[0 : N // 2] = fft(x, N // 2, 2 * s)
    X[N // 2 : N] = fft(x[s:], N // 2, 2 * s)

    for k in range(N // 2):
        p = X[k]
        q = np.exp(-2j * np.pi * k / N) * X[k + N // 2]
        X[k] = p + q
        X[k + N // 2] = p - q

    return X

def run_fft(N: int):
    start = time.perf_counter()
    Ck = fft(f, N)
    end = time.perf_counter()
    exec_time = end - start

    stages = int(np.log2(N))
    c_mult = (N // 2) * stages
    c_add = N * stages

    r_mult = 2 * N * stages
    r_add = 3 * N * stages

    print(f"N: {N}")
    print(f"\nЧас обчислення ШПФ: {exec_time:.6f} s")
    print("--- Кількість операцій ШПФ ---")
    print(f"Комплексні операції -> Множення: {c_mult}, Додавання: {c_add}")
    print(f"Дійсні операції     -> Множення: {r_mult}, Додавання: {r_add}")
    
    print("Коефіцієнти розкладу (перші п'ять):")
    for i in range(0, 5):
        print(f"C{i} = {Ck[i]:3f}; |C{i}| = {np.abs(Ck[i]):3f}; argC{i} = {Fraction(np.angle(Ck[i]) / np.pi).limit_denominator(10)}*pi")
    
    dft_mult = 8 * N**2
    dft_div = 2 * N**2

    start = time.perf_counter()
    Ck_lab2 = calc_Ck(f)
    end = time.perf_counter()
    exec_time = end - start

    print("\n--- Порівняння з ДПФ ---")
    print(f"Час обчислення ДПФ: {exec_time:.6f} s")
    print(f"ДПФ множення: {dft_mult} (менше в {dft_mult / r_mult:.1f}x разів)")
    print(f"ДПФ ділення:  {dft_div}")

    print("Коефіцієнти розкладу (перші п'ять):")
    for i in range(0, 5):
        print(f"C{i} = {Ck_lab2[i]:3f}; |C{i}| = {np.abs(Ck_lab2[i]):3f}; argC{i} = {Fraction(np.angle(Ck_lab2[i]) / np.pi).limit_denominator(10)}*pi")

    return Ck

if __name__ == "__main__":
    run_fft(N1)
