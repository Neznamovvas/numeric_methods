import numpy as np
import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def mls_coeffs(x, y, degree):
    n = len(x)
    A = np.zeros((degree + 1, degree + 1))
    b = np.zeros(degree + 1)
    for k in range(degree + 1):
        for i in range(degree + 1):
            A[k, i] = sum(xj ** (k + i) for xj in x)
        b[k] = sum(yj * (xj ** k) for xj, yj in zip(x, y))
    return np.linalg.solve(A, b)


def poly_str(coeffs, var='x'):
    terms = []
    for i, a in enumerate(coeffs):
        if i == 0:
            terms.append(f"{a:+.6f}")
        elif i == 1:
            terms.append(f"{a:+.6f}*{var}")
        else:
            terms.append(f"{a:+.6f}*{var}^{i}")
    return " ".join(terms)


if __name__ == "__main__":
    print("=" * 70)
    print("ЗАДАЧА 3.3. Метод наименьших квадратов")
    print("=" * 70)

    lines = read_file("input.txt")
    N = int(lines[0])
    x = np.array([float(v) for v in lines[1].split()])
    y = np.array([float(v) for v in lines[2].split()])

    print("\nИсходные данные:")
    print(f"{'i':>3} {'x_i':>10} {'y_i':>12}")
    for i in range(len(x)):
        print(f"{i:>3} {x[i]:>10.4f} {y[i]:>12.6f}")

    degs = [1, 2, 3]
    results = {}

    for deg in degs:
        coeffs = mls_coeffs(x, y, deg)
        P_vals = np.array([sum(c * xi ** k for k, c in enumerate(coeffs)) for xi in x])
        mse = np.mean((P_vals - y) ** 2)
        results[deg] = (coeffs, mse)

        print(f"\nМногочлен степени {deg}:")
        print(f"  Коэффициенты (a_0 ... a_{deg}): {coeffs}")
        print(f"  P_{deg}(x) = {poly_str(coeffs)}")
        print(f"  MSE_{deg} = {mse:.6e}")

    best = min(results.keys(), key=lambda d: results[d][1])
    print("\n" + "-" * 70)
    print("Сравнение ошибок:")
    for d in degs:
        print(f"  MSE_{d} = {results[d][1]:.6e}")
    print(f"Наилучшее приближение: многочлен степени {best}")

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, 'ko', markersize=9, label='Исходные данные', zorder=5)

    xs = np.linspace(min(x), max(x), 400)
    colors = {1: 'r-', 2: 'g-', 3: 'b-'}
    for deg in degs:
        coeffs, _ = results[deg]
        P_vals = [sum(c * xi ** k for k, c in enumerate(coeffs)) for xi in xs]
        plt.plot(xs, P_vals, colors[deg], linewidth=2,
                 label=f'P_{deg}(x), MSE = {results[deg][1]:.4e}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Приближение методом наименьших квадратов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()