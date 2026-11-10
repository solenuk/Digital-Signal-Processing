import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def f(x):
    return (x ** 22) * np.exp(-(x**2) / 22)

def trapezoidal_rule(f, a, b, N, I_N = None):
    h = (b-a)/N
    
    if I_N is not None:
        Sum = sum(f(a + i * h) for i in range(1, N, 2))
        
        result = I_N / 2 + h * Sum
    else: 
        Sum = sum(f(a + i * h) for i in range(1, N))

        result = h * (0.5 * (f(a) + f(b)) + Sum)

    return result

def evaluate(f, a, b):
    N = 200
    I_N = trapezoidal_rule(f, a, b, N) 
    eps = 1e-6

    while True:
        N *= 2
        
        I_2N = trapezoidal_rule(f, a, b, N, I_N)

        error = np.abs(I_2N - I_N) / 3

        if error <= eps:
            return I_2N
        
        I_N = I_2N

def calc_a_k(k):
    func = lambda x: f(x)*np.cos(x*k*np.pi)
    return (2)*(evaluate(func, 0, 1))

def calc_b_k(k):
    func = lambda x: f(x)*np.sin(x*k*np.pi)
    return 0

def approx(x, a_k_coeffs, b_k_coeffs, N):
    sum = a_k_coeffs[0] / 2
    for k in range(1, N + 1):
        if k < len(a_k_coeffs):
            sum += a_k_coeffs[k]*np.cos(k*x*np.pi)
        if k < len(b_k_coeffs):
            sum += b_k_coeffs[k]*np.sin(k*x*np.pi)
    return sum

def calc_relative_error(actual_vals, approx_vals):
    rmse = np.sqrt(np.mean((actual_vals - approx_vals)**2))
    max_val = np.max(np.abs(actual_vals))
    
    return (rmse / max_val) * 100

def plot_harmonics_and_spectrum(k_values, a_coeffs, b_coeffs, x_vals, y_actual, y_approx):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(x_vals, y_actual, 'k--', label='Original f(x)', linewidth=2)
    axs[0].plot(x_vals, y_approx, 'r-', label=f'Fourier Series (N={N_max})', alpha=0.7)
    axs[0].set_title("Time Domain: Function vs Approximation")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].stem(k_values - 0.1, a_coeffs, linefmt='b-', markerfmt='bo', label='a(k)')
    axs[1].stem(k_values + 0.1, b_coeffs, linefmt='r-', markerfmt='ro', label='b(k)')
    axs[1].set_title("Frequency Domain: Fourier Coefficients")
    axs[1].set_xlabel("Harmonic (k)")
    axs[1].set_ylabel("Coefficient Value")
    axs[1].set_xticks(k_values)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_harmonics(a_coeffs, b_coeffs, x_vals, y_actual):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals, 
        y=y_actual, 
        mode='lines', 
        name='Original f(x)', 
        line=dict(color='black', width=3, dash='dash')
    ))

    N_values_to_show = [i for i in range(0, len(a_coeffs), 2)]

    for N_part in N_values_to_show:
        y_approx_part = [approx(x, a_coeffs, b_coeffs, N_part) for x in x_vals]
        
        fig.add_trace(go.Scatter(
            x=x_vals, 
            y=y_approx_part, 
            mode='lines', 
            name=f'N = {N_part}'
        ))

    fig.update_layout(
        title="Накладення гармонік",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.show()

def save_results(N, a_coeffs, b_coeffs, error, filename="results.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Order N: {N}\n")
        f.write("Coefficients a_k:\n")
        for k, a in enumerate(a_coeffs):
            f.write(f"a_{k} = {a:.6f}\n")
        f.write("Coefficients b_k:\n")
        for k, b in enumerate(b_coeffs):
            f.write(f"b_{k} = {b:.6f}\n")
        f.write(f"Approximation Error: {error:.3f}%\n")

N_max = 50

k_values = np.arange(0, N_max + 1)
a_coeffs = [calc_a_k(k) for k in k_values]
b_coeffs = [calc_b_k(k) for k in k_values]

x_vals = np.linspace(-1, 1, 1000)
y_actual = f(x_vals)
y_approx = [approx(x, a_coeffs, b_coeffs, N_max) for x in x_vals]

error = calc_relative_error(y_actual, y_approx)

print(f"Relative error of approximation: {error:.3f}%")
save_results(N_max, a_coeffs, b_coeffs, error)
plot_harmonics(a_coeffs,b_coeffs, x_vals, y_actual)
plot_harmonics_and_spectrum(k_values, a_coeffs, b_coeffs, x_vals, y_actual, y_approx)