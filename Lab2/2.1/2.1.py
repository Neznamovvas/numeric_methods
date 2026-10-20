import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def f(x):
    return x * np.exp(x) + x ** 2 - 1


def df(x):
    return np.exp(x) * (1 + x) + 2 * x


def find_x0_graphically():
    x_grid = np.linspace(-1.5, 1.0, 400)
    y_grid = f(x_grid)
    a, b = None, None
    for i in range(len(x_grid) - 1):
        if y_grid[i] * y_grid[i + 1] < 0 and x_grid[i] > 0:
            a, b = x_grid[i], x_grid[i + 1]
            break
    x0 = 0.5 * (a + b)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, y_grid, label=r'\(f(x) = x e^x + x^2 - 1\)', color='blue')
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(a, f(a), 'ro')
    plt.plot(b, f(b), 'ro')
    plt.plot(x0, f(x0), 'g*', markersize=12, label=f'x0 ≈ {x0:.4f}')
    plt.xlim(-1.5, 1.0)
    plt.ylim(-2, 3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Графическое определение x0')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('initial_guess_1d.png', dpi=150)
    plt.show()
    
    return x0


def simple_iteration(f, x0, eps, c, max_iter=1000, x_star=None):
    def phi(x):
        return x + c * f(x)
        
    x = x0
    history = [x]
    errors = []
    
    for k in range(1, max_iter + 1):
        x_new = phi(x)
        history.append(x_new)
        
        if x_star is not None:
            errors.append(abs(x_new - x_star))
            
        if abs(x_new - x) < eps:
            return x_new, k, errors, history
            
        x = x_new
        
    raise RuntimeError("Метод простой итерации не сошёлся")


def newton(f, df, x0, eps, max_iter=1000, x_star=None):
    x = x0
    history = [x]
    errors = []
    
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-15:
            raise ZeroDivisionError(f"Нулевая производная в x = {x}")
            
        x_new = x - fx / dfx
        history.append(x_new)
        
        if x_star is not None:
            errors.append(abs(x_new - x_star))
            
        if abs(x_new - x) < eps:
            return x_new, k, errors, history
            
        x = x_new
        
    raise RuntimeError("Метод Ньютона не сошёлся")


if __name__ == "__main__":
    eps = 1e-6
    
    print("=" * 65)
    print("ШАГ 1. ГРАФИЧЕСКОЕ ОПРЕДЕЛЕНИЕ НАЧАЛЬНОГО ПРИБЛИЖЕНИЯ")
    print("=" * 65)
    x0 = find_x0_graphically()
    print(f"Найденное по графику x0 = {x0:.4f}")

    x_star = brentq(f, 0, 1)
    print(f"Эталонный корень (brentq): x* = {x_star:.10f}")
    print("-" * 65)

    print("\nШАГ 2. ЧИСЛЕННОЕ РЕШЕНИЕ")
    print("=" * 65)

    c = -1.0 / df(x0)
    
    print(f"\n[МЕТОД ПРОСТОЙ ИТЕРАЦИИ]")
    print(f"  Вид: phi(x) = x + c*f(x), где c = {c:.6f}")
    x_si, iters_si, err_si, hist_si = simple_iteration(f, x0, eps, c, x_star=x_star)
    print(f"  Корень:   {x_si:.10f}")
    print(f"  Итераций: {iters_si}")
    print(f"  |x - x*|: {abs(x_si - x_star):.3e}")

    print("\n[МЕТОД НЬЮТОНА]")
    x_nt, iters_nt, err_nt, hist_nt = newton(f, df, x0, eps, x_star=x_star)
    print(f"  Корень:   {x_nt:.10f}")
    print(f"  Итераций: {iters_nt}")
    print(f"  |x - x*|: {abs(x_nt - x_star):.3e}")

    print("\n" + "=" * 65)
    print("АНАЛИЗ: ПОГРЕШНОСТЬ ОТ ЧИСЛА ИТЕРАЦИЙ")
    print("=" * 65)

    print("\n[Метод простой итерации]")
    print(f"{'k':>3} {'x_k':>14} {'|x_k - x*|':>14} {'|Δx|':>12}")
    for k in range(len(hist_si)):
        xk = hist_si[k]
        err = abs(xk - x_star)
        diff_str = "—" if k == 0 else f"{abs(hist_si[k] - hist_si[k-1]):.3e}"
        print(f"{k:>3} {xk:>14.10f} {err:>14.3e} {diff_str:>12}")

    print("\n[Метод Ньютона]")
    print(f"{'k':>3} {'x_k':>14} {'|x_k - x*|':>14} {'|Δx|':>12}")
    for k in range(len(hist_nt)):
        xk = hist_nt[k]
        err = abs(xk - x_star)
        diff_str = "—" if k == 0 else f"{abs(hist_nt[k] - hist_nt[k-1]):.3e}"
        print(f"{k:>3} {xk:>14.10f} {err:>14.3e} {diff_str:>12}")

    plt.figure(figsize=(10, 6))
    if err_si:
        plt.semilogy(range(1, len(err_si) + 1), err_si, 'o-', label='МПИ')
    if err_nt:
        plt.semilogy(range(1, len(err_nt) + 1), err_nt, 's-', label='Ньютон')
    plt.xlabel('Итерация k')
    plt.ylabel(r'\(|x_k - x^*|\)')
    plt.title('Сходимость методов')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.savefig('convergence_1d.png', dpi=150)
    plt.show()