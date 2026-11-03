import numpy as np
import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def build_spline(x_nodes, y_nodes):
    n = len(x_nodes) - 1
    h = [x_nodes[i + 1] - x_nodes[i] for i in range(n)]

    m = n - 1
    A = np.zeros((m, m))
    rhs = np.zeros(m)

    for i in range(m):
        k = i + 1
        if i - 1 >= 0:
            A[i, i - 1] = h[k - 1]
        A[i, i] = 2 * (h[k - 1] + h[k])
        if i + 1 < m:
            A[i, i + 1] = h[k]
        slope_right = (y_nodes[k + 1] - y_nodes[k]) / h[k]
        slope_left = (y_nodes[k] - y_nodes[k - 1]) / h[k - 1]
        rhs[i] = 3 * (slope_right - slope_left)

    c_inner = np.linalg.solve(A, rhs) if m > 0 else np.array([])
    c = np.concatenate(([0.0], c_inner, [0.0]))

    a = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        a[i] = y_nodes[i]
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3.0

    return [(a[i], b[i], c[i], d[i]) for i in range(n)]


def spline_eval(coeffs, x_nodes, x):
    n = len(coeffs)
    for i in range(n):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            a, b, c, d = coeffs[i]
            dx = x - x_nodes[i]
            return a + b * dx + c * dx ** 2 + d * dx ** 3, i
    if x < x_nodes[0]:
        i = 0
    else:
        i = n - 1
    a, b, c, d = coeffs[i]
    dx = x - x_nodes[i]
    return a + b * dx + c * dx ** 2 + d * dx ** 3, i


if __name__ == "__main__":
    print("=" * 70)
    print("ЗАДАЧА 3.2. Кубический сплайн")
    print("=" * 70)

    lines = read_file("input.txt")
    n_points = int(lines[0])
    x_nodes = [float(v) for v in lines[1].split()]
    y_nodes = [float(v) for v in lines[2].split()]
    x_star = float(lines[3])

    print("\nИсходные данные:")
    print(f"{'i':>3} {'x_i':>10} {'y_i':>12}")
    for i in range(len(x_nodes)):
        print(f"{i:>3} {x_nodes[i]:>10.4f} {y_nodes[i]:>12.6f}")

    coeffs = build_spline(x_nodes, y_nodes)

    print("\nКоэффициенты сплайна S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3:")
    print(f"{'i':>3} {'[x_i, x_i+1]':>20} {'a_i':>12} {'b_i':>12} {'c_i':>12} {'d_i':>12}")
    for i, (a, b, c, d) in enumerate(coeffs):
        seg = f"[{x_nodes[i]:.2f}, {x_nodes[i + 1]:.2f}]"
        print(f"{i:>3} {seg:>20} {a:>12.6f} {b:>12.6f} {c:>12.6f} {d:>12.6f}")

    S_star, seg_idx = spline_eval(coeffs, x_nodes, x_star)
    print(f"\nЗначение S(x*) в точке x* = {x_star}:")
    print(f"  S(x*) = {S_star:.10f}")
    print(f"  Номер интервала: {seg_idx} "
          f"(отрезок [{x_nodes[seg_idx]:.4f}, {x_nodes[seg_idx + 1]:.4f}])")

    print("\nПроверка прохождения через исходные узлы:")
    ok = True
    for i in range(len(x_nodes)):
        S_i, _ = spline_eval(coeffs, x_nodes, x_nodes[i])
        err = abs(S_i - y_nodes[i])
        status = "OK" if err < 1e-8 else "ОШИБКА"
        print(f"  x = {x_nodes[i]:.4f}: S(x) = {S_i:.8f}, "
              f"y = {y_nodes[i]:.8f}, err = {err:.2e}  [{status}]")
        if err > 1e-8:
            ok = False
    print(f"  Все узлы пройдены точно: {'да' if ok else 'НЕТ'}")

    print("\nПроверка граничных условий (S''(x0) = S''(xn) = 0):")
    S2_left = 2 * coeffs[0][2]
    a, b, c, d = coeffs[-1]
    h_n = x_nodes[-1] - x_nodes[-2]
    S2_right = 2 * c + 6 * d * h_n
    print(f"  S''(x0) = {S2_left:.3e}")
    print(f"  S''(xn) = {S2_right:.3e}")

    xs = np.linspace(x_nodes[0], x_nodes[-1], 500)
    S_vals = [spline_eval(coeffs, x_nodes, x)[0] for x in xs]

    plt.figure(figsize=(9, 5))
    plt.plot(xs, S_vals, 'b-', label='Кубический сплайн', linewidth=2)
    plt.plot(x_nodes, y_nodes, 'ko', markersize=9, label='Исходные узлы')
    plt.plot(x_star, S_star, 'g*', markersize=15,
             label=f'x* = {x_star:.3f}, S = {S_star:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Кубический сплайн')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()