import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


def F(X):
    x1, x2 = X
    return np.array([
        2 * x1 - np.cos(x2),
        2 * x2 - np.exp(x1)
    ])


def J(X):
    x1, x2 = X
    return np.array([
        [2.0, np.sin(x2)],
        [-np.exp(x1), 2.0]
    ])


def Phi(X, C):
    return X + C @ F(X)


def find_x0_graphically():
    x1_c1 = np.linspace(0, 0.5, 200)
    x2_c1 = np.arccos(2 * x1_c1)
    
    x1_c2 = np.linspace(0, 1, 200)
    x2_c2 = 0.5 * np.exp(x1_c2)

    diff = x2_c1 - 0.5 * np.exp(x1_c1)
    idx = np.where(np.diff(np.sign(diff)))[0]
    
    if len(idx) > 0:
        i = idx[0]
        x0_x1 = 0.5 * (x1_c1[i] + x1_c1[i + 1])
        x0_x2 = 0.5 * (x2_c1[i] + x2_c1[i + 1])
        x0 = np.array([x0_x1, x0_x2])
    else:
        x0 = np.array([0.4, 0.6])

    plt.figure(figsize=(7, 5))
    plt.plot(x1_c1, x2_c1, label=r'\(2x_1 - \cos(x_2) = 0\)')
    plt.plot(x1_c2, x2_c2, label=r'\(2x_2 - e^{x_1} = 0\)')
    plt.plot(x0[0], x0[1], 'ro', markersize=10, label=f'x0 ≈ ({x0[0]:.3f}, {x0[1]:.3f})')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r'\(x_1\)')
    plt.ylabel(r'\(x_2\)')
    plt.title('Графическое определение x0')
    plt.grid(True)
    plt.legend()
    plt.savefig('initial_guess.png', dpi=150)
    plt.show()

    return x0


def simple_iteration(F, Phi, C, X0, eps, max_iter=1000, x_star=None):
    X = X0.astype(float).copy()
    history = [X.copy()]
    errors = []
    
    for k in range(1, max_iter + 1):
        X_new = Phi(X, C)
        diff = np.max(np.abs(X_new - X))
        history.append(X_new.copy())
        
        if x_star is not None:
            errors.append(np.max(np.abs(X_new - x_star)))
            
        if diff < eps:
            return X_new, k, errors, history
            
        X = X_new
        
    raise RuntimeError("МПИ не сошёлся")


def newton(F, J, X0, eps, max_iter=1000, x_star=None):
    X = X0.astype(float).copy()
    history = [X.copy()]
    errors = []
    
    for k in range(1, max_iter + 1):
        Fx = F(X)
        Jx = J(X)
        
        dX = np.linalg.solve(Jx, -Fx)
        X_new = X + dX
        diff = np.max(np.abs(dX))
        
        history.append(X_new.copy())
        
        if x_star is not None:
            errors.append(np.max(np.abs(X_new - x_star)))
            
        if diff < eps:
            return X_new, k, errors, history
            
        X = X_new
        
    raise RuntimeError("Метод Ньютона не сошёлся")


if __name__ == "__main__":
    eps = 1e-6
    
    print("=" * 65)
    print("ШАГ 1. ГРАФИЧЕСКОЕ ОПРЕДЕЛЕНИЕ НАЧАЛЬНОГО ПРИБЛИЖЕНИЯ")
    print("=" * 65)
    X0 = find_x0_graphically()
    print(f"Найденное по графику x0 = ({X0[0]:.4f}, {X0[1]:.4f})")

    sol = root(F, X0)
    X_star = sol.x
    print(f"Эталонный корень (scipy): x* = ({X_star[0]:.8f}, {X_star[1]:.8f})")
    print("-" * 65)

    print("\nШАГ 2. ЧИСЛЕННОЕ РЕШЕНИЕ")
    print("=" * 65)

    C = -np.linalg.inv(J(X0))

    print(f"\n[МЕТОД ПРОСТОЙ ИТЕРАЦИИ]")
    print("  Вид: Phi(X) = X + C*F(X), где C = -J^(-1)(x0)")
    x_si, iters_si, err_si, hist_si = simple_iteration(F, Phi, C, X0, eps, x_star=X_star)
    print(f"  Корень:   ({x_si[0]:.8f}, {x_si[1]:.8f})")
    print(f"  Итераций: {iters_si}")
    print(f"  |x - x*|: {np.max(np.abs(x_si - X_star)):.3e}")

    print("\n[МЕТОД НЬЮТОНА]")
    x_nt, iters_nt, err_nt, hist_nt = newton(F, J, X0, eps, x_star=X_star)
    print(f"  Корень:   ({x_nt[0]:.8f}, {x_nt[1]:.8f})")
    print(f"  Итераций: {iters_nt}")
    print(f"  |x - x*|: {np.max(np.abs(x_nt - X_star)):.3e}")

    print("\n" + "=" * 65)
    print("АНАЛИЗ: ПОГРЕШНОСТЬ ОТ ЧИСЛА ИТЕРАЦИЙ")
    print("=" * 65)

    print("\n[Метод простой итерации]")
    print(f"{'k':>3} {'x1':>12} {'x2':>12} {'||x_k - x*||':>15} {'||Δx||':>12}")
    for k in range(len(hist_si)):
        xk = hist_si[k]
        err = np.max(np.abs(xk - X_star))
        diff_str = "—" if k == 0 else f"{np.max(np.abs(hist_si[k] - hist_si[k-1])):.3e}"
        print(f"{k:>3} {xk[0]:>12.8f} {xk[1]:>12.8f} {err:>15.3e} {diff_str:>12}")

    print("\n[Метод Ньютона]")
    print(f"{'k':>3} {'x1':>12} {'x2':>12} {'||x_k - x*||':>15} {'||Δx||':>12}")
    for k in range(len(hist_nt)):
        xk = hist_nt[k]
        err = np.max(np.abs(xk - X_star))
        diff_str = "—" if k == 0 else f"{np.max(np.abs(hist_nt[k] - hist_nt[k-1])):.3e}"
        print(f"{k:>3} {xk[0]:>12.8f} {xk[1]:>12.8f} {err:>15.3e} {diff_str:>12}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if err_si:
        plt.semilogy(range(1, len(err_si) + 1), err_si, 'o-', label='МПИ')
    if err_nt:
        plt.semilogy(range(1, len(err_nt) + 1), err_nt, 's-', label='Ньютон')
    plt.xlabel('Итерация k')
    plt.ylabel(r'\(\|x_k - x^*\|_\infty\)')
    plt.title('Сходимость методов')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    x1_vals = np.linspace(0, 1, 100)
    plt.plot(x1_vals, 0.5 * np.exp(x1_vals), label=r'\(x_2 = \frac{1}{2}e^{x_1}\)')
    
    x2_vals = np.linspace(0, 1, 100)
    plt.plot(0.5 * np.cos(x2_vals), x2_vals, label=r'\(x_1 = \frac{1}{2}\cos(x_2)\)')
    
    plt.plot(X_star[0], X_star[1], 'r*', markersize=12, label='Корень')
    plt.plot(X0[0], X0[1], 'go', markersize=8, label='x0')
    plt.xlabel(r'\(x_1\)')
    plt.ylabel(r'\(x_2\)')
    plt.title('Графическое отделение корня')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('sys_convergence.png', dpi=150)
    plt.show()