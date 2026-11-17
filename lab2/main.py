import numpy as np
from matplotlib import pyplot as plt
import time
from fractions import Fraction

n = 22
N1 = 10+n
N2 = 96+n

np.random.seed(N1)

f = (np.random.random((1,N1)) * 10)[0]

s = format(N2, "07b")
bit8 = "1" if n % 2 != 0 else "0"
s = bit8 + s
s = np.array(list(s), float)

def calc_Ak(f: np.ndarray, k: int):
    sum = 0
    N = len(f)
    # 4N multiplications and N divisions 
    for i in range(N):
        sum += f[i]*np.cos((2*np.pi*k*i)/N)

    return sum

def calc_Bk(f: np.ndarray, k: int):
    sum = 0
    N = len(f)
    # 4N multiplications and N divisions 
    for i in range(N):
        sum += f[i]*np.sin((2*np.pi*k*i)/N)

    return -sum

def calc_Ck(f: np.ndarray):
    # N harmonics * 4N multmultiplications and N divisions 
    # 4N^2 multmultiplications and N^2 divisions
    Ak = np.array([calc_Ak(f, k) for k in range(len(f))]) 

    # N harmonics * 4N multmultiplications and N divisions
    # 4N^2 multmultiplications and N^2 divisions
    Bk = np.array([calc_Bk(f, k) for k in range(len(f))]) 

    return Ak + Bk*1j

def calc_fi(Ck: np.ndarray):
    N = len(Ck)
    fi = []

    for i in range(N):
        sum = 0
        for k in range(N):
            sum += Ck[k]*(np.cos((2*np.pi*k*i)/N) + 1j*np.sin((2*np.pi*k*i)/N))
        fi.append(sum/N)
    
    return fi
    
def plot_spectrums(Ck: np.ndarray):
    k = np.arange(len(Ck))
    amplitudes = [np.abs(Ck[i]) for i in k]
    phases = [np.angle(Ck[i]) for i in k]

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.stem(k, amplitudes)
    plt.title("Спектр амплітуд")
    plt.xlabel("k")
    plt.ylabel("Амлітуда")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.stem(k, phases)
    plt.title("Спектр фаз")
    plt.xlabel("k")
    plt.ylabel("Фаза (радіани)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def part1(N: int):
    start = time.perf_counter()
    Ck = calc_Ck(f)
    end = time.perf_counter()
    exec_time = end - start
    print(f"Час обчислення ЗПФ: {exec_time:.6f} s")

    print(f)

    print("Коефіцієнти розкладу C_k k=(0, N):")
    for i in range(0, N):
        print(f"C{i} = {Ck[i]:3f}; |C{i}| = {np.abs(Ck[i]):3f}; argC{i} = {Fraction(np.angle(Ck[i]) / np.pi).limit_denominator(10)}*pi")

    mult_code = 8 * N**2
    div_code = 2 * N**2

    print(f"К-сть операцій -> Множення: {mult_code}, Ділення: {div_code}")
    plot_spectrums(Ck)

def s_t(t: np.ndarray, Ck: np.ndarray, Tc: float):
    N = len(Ck)
        
    s_t = Ck[0].real * np.ones_like(t)
        
    for i in range(1, N // 2):
        mag = np.abs(Ck[i])
        phi = np.angle(Ck[i])
        s_t += 2 * mag * np.cos((2 * np.pi * i * t) / Tc + phi)
            

    half = N // 2
    mag_half = np.abs(Ck[half])
    phi_half = np.angle(Ck[half])
    s_t += mag_half * np.cos((N * np.pi * t) / Tc + phi_half)
        
    return s_t

def part2():
    print(f"s(nTδ) = {s}")
    N = len(s)
    Tc = float(N)

    Ck = calc_Ck(s)/N

    print("Коефіцієнти розкладу C_k k=(0, N):")
    for i in range(0, N):
        print(f"C{i} = {Ck[i]:3f}; |C{i}| = {np.abs(Ck[i]):3f}; argC{i} = {Fraction(np.angle(Ck[i]) / np.pi).limit_denominator(10)}*pi")

    t = np.linspace(0, 2*Tc, 1000)
    st = s_t(t, Ck, Tc)

    plt.figure(figsize=(10, 5))
    plt.plot(t, st, label=r"$s(t)$", color="blue", linewidth=1.5)
    plt.stem(np.arange(N), s, linefmt="r--", markerfmt="ro", basefmt="k", label=r"Відліки $s(nT_\delta)$")
    plt.title("Відновлений аналоговий сигнал $s(t)$")
    plt.xlabel("t")
    plt.ylabel("s(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

def part3():
    N = len(s)
    Ck = calc_Ck(s)
    sn = calc_fi(Ck)

    print("Значення відліків s(nTδ) n=(0, N-1):")
    for i in range(N):
        print(f"s({i}Tδ) = {sn[i].real:5f}")

part1(N1)
# part2()
# part3()