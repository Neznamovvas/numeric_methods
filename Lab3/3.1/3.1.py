import numpy as np
import matplotlib.pyplot as plt
from math import log


def read_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def lagrange_poly(x_nodes, y_nodes, x):
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        li = 1.0
        for j in range(n):
            if i != j:
                li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * li
    return result


def build_divided_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    table = np.zeros((n, n))
    table[:, 0] = y_nodes
    for k in range(1, n):
        for i in range(n - k):
            table[i, k] = (table[i + 1, k - 1] - table[i, k - 1]) / (x_nodes[i + k] - x_nodes[i])
    return table


def newton_poly(x_nodes, table, x):
    n = len(x_nodes)
    result = table[0, 0]
    prod = 1.0
    for k in range(1, n):
        prod *= (x - x_nodes[k - 1])
        result += table[0, k] * prod
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("ЗАДАЧА 3.1. Многочлены Лагранжа и Ньютона")
    print("=" * 70)

    lines = read_file("input.txt")
    n = int(lines[0])
    x_nodes = [float(v) for v in lines[1].split()]
    y_nodes = [float(v) for v in lines[2].split()]
    x_star = float(lines[3])

    print("\nУзлы интерполяции:")
    print(f"{'i':>3} {'x_i':>12} {'y_i':>14}")
    for i in range(len(x_nodes)):
        print(f"{i:>3} {x_nodes[i]:>12.6f} {y_nodes[i]:>14.6f}")

    table = build_divided_differences(x_nodes, y_nodes)
    print("\nТаблица разделённых разностей:")
    header = "      " + "".join(f"{'k=' + str(k):>16}" for k in range(len(x_nodes)))
    print(header)
    for i in range(len(x_nodes)):
        row = f"x_{i:<3} "
        for k in range(len(x_nodes) - i):
            row += f"{table[i, k]:>16.6f}"
        print(row)

    print("\nМногочлен Лагранжа (вид в виде суммы слагаемых):")
    terms = []
    for i in range(len(x_nodes)):
        denom = 1.0
        factors = []
        for j in range(len(x_nodes)):
            if i != j:
                denom *= (x_nodes[i] - x_nodes[j])
                factors.append(f"(x - {x_nodes[j]:.4f})")
        coeff = y_nodes[i] / denom
        terms.append(f"{coeff:+.6f} * " + " * ".join(factors))
    print("  L(x) = " + "\n         ".join(terms))

    print("\nМногочлен Ньютона (вид в виде суммы слагаемых):")
    terms = [f"{table[0, 0]:+.6f}"]
    for k in range(1, len(x_nodes)):
        factors = " * ".join(f"(x - {x_nodes[j]:.4f})" for j in range(k))
        terms.append(f"{table[0, k]:+.6f} * {factors}")
    print("  N(x) = " + "\n         ".join(terms))

    L_star = lagrange_poly(x_nodes, y_nodes, x_star)
    N_star = newton_poly(x_nodes, table, x_star)

    print(f"\nЗначение в точке x* = {x_star}:")
    print(f"  L(x*) = {L_star:.10f}")
    print(f"  N(x*) = {N_star:.10f}")
    print(f"  |L(x*) - N(x*)| = {abs(L_star - N_star):.3e}")

    exact_func = lambda x: log(x) + x
    func_name = 'ln(x) + x'

    if exact_func is not None:
        exact = exact_func(x_star)
        print(f"\nТочная функция: f(x) = {func_name}")
        print(f"Точное значение f(x*) = {exact:.10f}")
        print(f"Абсолютная погрешность Лагранжа: {abs(L_star - exact):.3e}")
        print(f"Абсолютная погрешность Ньютона:  {abs(N_star - exact):.3e}")
    else:
        print("\nТочная функция не определена автоматически (не совпала ни с одним шаблоном).")
        print("Если известна аналитика, добавьте её в код.")

    x_min, x_max = min(x_nodes) - 0.1, max(x_nodes) + 0.1
    xs = np.linspace(x_min, x_max, 400)
    L_vals = [lagrange_poly(x_nodes, y_nodes, x) for x in xs]
    N_vals = [newton_poly(x_nodes, table, x) for x in xs]

    plt.figure(figsize=(9, 5))
    plt.plot(xs, L_vals, 'b-', label='Лагранж', linewidth=2)
    plt.plot(xs, N_vals, 'r--', label='Ньютон', linewidth=2)
    plt.plot(x_nodes, y_nodes, 'ko', markersize=8, label='Узлы')
    plt.plot(x_star, L_star, 'g*', markersize=15,
             label=f'x* = {x_star:.3f}, y = {L_star:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Интерполяционные многочлены Лагранжа и Ньютона')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()