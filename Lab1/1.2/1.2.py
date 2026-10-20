import numpy as np


def read_tridiagonal(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    n = int(lines[0])

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    a_vals = [float(x) for x in lines[1].split()]
    if len(a_vals) != n - 1:
        raise ValueError(f"Поддиагональ: ожидалось {n-1} чисел, получено {len(a_vals)}")
    a[1:] = a_vals

    b_vals = [float(x) for x in lines[2].split()]
    if len(b_vals) != n:
        raise ValueError(f"Главная диагональ: ожидалось {n} чисел, получено {len(b_vals)}")
    b[:] = b_vals

    c_vals = [float(x) for x in lines[3].split()]
    if len(c_vals) != n - 1:
        raise ValueError(f"Наддиагональ: ожидалось {n-1} чисел, получено {len(c_vals)}")
    c[:-1] = c_vals

    d_lines = lines[4:4 + n]
    if len(d_lines) != n:
        raise ValueError(f"Правая часть: ожидалось {n} строк, получено {len(d_lines)}")
    for i, s in enumerate(d_lines):
        d[i] = float(s)

    return a, b, c, d


def sweep_method(a, b, c, d, n):
    p = np.zeros(n)
    q = np.zeros(n)

    if abs(b[0]) < 1e-15:
        raise ZeroDivisionError("Нулевой элемент b[0] — прогонка невозможна")

    p[0] = -c[0] / b[0]
    q[0]  =  d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * p[i - 1]
        if abs(denom) < 1e-15:
            raise ZeroDivisionError(
                f"Нулевой знаменатель прогонки на шаге i={i} — "
                f"возможно, требуется выбор ведущего элемента"
            )
        p[i] = -c[i] / denom
        q[i]  = (d[i] - a[i] * q[i - 1]) / denom

    x = np.zeros(n)
    x[n - 1] = q[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x, p, q


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "input.txt")
    a, b, c, d = read_tridiagonal(filename)
    n = len(b)

    print("Исходные данные:")
    print("n =", n)
    print("a =", a)
    print("b =", b)
    print("c =", c)
    print("d =", d)

    x, p, q = sweep_method(a, b, c, d, n)

    print("\nПрогоночные коэффициенты:")
    print(f"{'i':>3} {'p[i]':>15} {'q[i]':>15}")
    for i in range(n):
        print(f"{i:>3} {p[i]:>15.10f} {q[i]:>15.10f}")

    print("\nРешение системы:")
    for i in range(n):
        print(f"x[{i}] = {x[i]:.10f}")
