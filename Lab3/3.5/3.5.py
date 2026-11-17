import sympy as sp


def read_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def f(x):
    return x**2 / (x**4 + 256)


def integrate_rect_left(f, a, b, h):
    n = int(round((b - a) / h))
    s = sum(f(a + i * h) for i in range(n))
    return s * h


def integrate_rect_right(f, a, b, h):
    n = int(round((b - a) / h))
    s = sum(f(a + i * h) for i in range(1, n + 1))
    return s * h


def integrate_rect_mid(f, a, b, h):
    n = int(round((b - a) / h))
    s = sum(f(a + (i + 0.5) * h) for i in range(n))
    return s * h


def integrate_trapezoid(f, a, b, h):
    n = int(round((b - a) / h))
    s = (f(a) + f(b)) / 2.0 + sum(f(a + i * h) for i in range(1, n))
    return s * h


def integrate_simpson(f, a, b, h):
    n = int(round((b - a) / h))
    if n % 2 != 0:
        raise ValueError("Для метода Симпсона число шагов должно быть чётным")
    s = f(a) + f(b)
    for i in range(1, n):
        coef = 4 if i % 2 != 0 else 2
        s += coef * f(a + i * h)
    return s * h / 3.0


def runge_romberg(I_h1, I_h2, p):
    return I_h2 + (I_h2 - I_h1) / (2**p - 1)


def error_runge(I_h1, I_h2, p):
    return abs(I_h2 - I_h1) / (2**p - 1)


if __name__ == "__main__":
    print("=" * 90)
    print("ЗАДАЧА 3.5. Численное интегрирование")
    print("Функция: y = x^2 / (x^4 + 256)")
    print("=" * 90)

    lines = read_file("input.txt")
    a, b = map(float, lines[0].split())
    h1, h2 = map(float, lines[1].split())

    print(f"\nОтрезок интегрирования: [{a}, {b}]")
    print(f"Шаг h1 = {h1}, Шаг h2 = {h2}")

    x_sym = sp.Symbol('x')
    f_sym = x_sym**2 / (x_sym**4 + 256)
    
    integral_expr = sp.integrate(f_sym, (x_sym, a, b))
    
    exact_val = float(integral_expr.evalf())
    
    print(f"Точное значение интеграла (вычислено SymPy): {exact_val:.8f}")

    methods = [
        ("Левых прямоугольников",    1, integrate_rect_left),
        ("Правых прямоугольников",   1, integrate_rect_right),
        ("Средних прямоугольников",  2, integrate_rect_mid),
        ("Трапеций",                 2, integrate_trapezoid),
        ("Симпсона",                 4, integrate_simpson),
    ]

    header = (f"{'Метод':>25} | {'I(h1)':>10} | {'I(h2)':>10} | {'p':>2} | "
              f"{'Рунге-Ромберг':>13} | {'Погр.(h2)':>10} | {'Абс.погр.(h2)':>13}")
    print("\n" + header)
    print("-" * len(header))

    best_method = None
    best_err = float('inf')
    best_val = None

    for name, p, func in methods:
        I_h1 = func(f, a, b, h1)
        I_h2 = func(f, a, b, h2)
        I_rr = runge_romberg(I_h1, I_h2, p)
        err_h2 = error_runge(I_h1, I_h2, p)
        abs_err = abs(I_h2 - exact_val)

        if abs_err < best_err:
            best_err = abs_err
            best_method = name
            best_val = I_h2

        print(f"{name:>25} | {I_h1:>10.6f} | {I_h2:>10.6f} | {p:>2} | "
              f"{I_rr:>13.6f} | {err_h2:>10.2e} | {abs_err:>13.2e}")

    print("-" * len(header))
    print(f"\nНаиболее точный метод (по абсолютной погрешности с h2): {best_method}")
    print(f"Значение интеграла: {best_val:.6f}")
    print(f"Погрешность этого метода: {best_err:.2e}")
    print("=" * 90)