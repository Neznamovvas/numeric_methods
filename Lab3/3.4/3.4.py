def read_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def compute_derivatives(x_nodes, y_nodes, x_star):
    n = len(x_nodes)
    
    h = x_nodes[1] - x_nodes[0]
    for i in range(1, n - 1):
        if abs((x_nodes[i + 1] - x_nodes[i]) - h) > 1e-9:
            raise ValueError("Ошибка: таблица должна быть равноотстоящей")

    k = min(range(n), key=lambda i: abs(x_nodes[i] - x_star))
    
    if k == 0 or k == n - 1:
        raise ValueError("Ошибка: точка на краю таблицы, центральные/двусторонние разности неприменимы")

    d1_left    = (y_nodes[k] - y_nodes[k - 1]) / h
    d1_central = (y_nodes[k + 1] - y_nodes[k - 1]) / (2.0 * h)
    d1_right   = (y_nodes[k + 1] - y_nodes[k]) / h
    d2         = (y_nodes[k + 1] - 2.0 * y_nodes[k] + y_nodes[k - 1]) / (h ** 2)
    
    return d1_left, d1_central, d1_right, d2, k, h


if __name__ == "__main__":
    print("=" * 70)
    print("ЗАДАЧА 3.4. Численное дифференцирование")
    print("=" * 70)

    lines = read_file("input.txt")
    n_points = int(lines[0])
    x_nodes = [float(v) for v in lines[1].split()]
    y_nodes = [float(v) for v in lines[2].split()]
    x_star = float(lines[3])

    print("\n1. Исходная таблица значений функции:")
    print(f"{'i':>3} | {'x_i':>8} | {'y_i':>10}")
    print("-" * 27)
    for i in range(len(x_nodes)):
        print(f"{i:>3} | {x_nodes[i]:>8.4f} | {y_nodes[i]:>10.4f}")

    print(f"\n2. Точка, в которой выполняется вычисление: X* = {x_star}")

    d1_left, d1_central, d1_right, d2, k, h = compute_derivatives(x_nodes, y_nodes, x_star)

    print("\n3. Узлы таблицы, использованные при вычислении производных:")
    print(f"{'i':>3} | {'x_i':>8} | {'y_i':>10}")
    print("-" * 27)
    for i in range(k - 1, k + 2):
        print(f"{i:>3} | {x_nodes[i]:>8.4f} | {y_nodes[i]:>10.4f}")

    print("\n4. Значения первой производной:")
    print(f"   Левосторонняя:  f'-(x*) ≈ {d1_left:.6f}")
    print(f"   Центральная:    f'(x*)  ≈ {d1_central:.6f}")
    print(f"   Правосторонняя: f'+(x*) ≈ {d1_right:.6f}")

    print(f"\n5. Значение второй производной:  f''(x*) ≈ {d2:.6f}")

    print(f"\n(Шаг таблицы h = {h:.1f})")
    print("=" * 70)