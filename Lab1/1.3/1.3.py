import numpy as np


def read_system(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    eps = float(lines[1])

    A = np.zeros((n, n))
    for i in range(n):
        row = lines[2 + i].split()
        if len(row) != n:
            raise ValueError(
                f"Строка {2 + i + 1} матрицы: ожидалось {n} чисел, получено {len(row)}"
            )
        A[i] = [float(x) for x in row]

    b = np.zeros(n)
    for i in range(n):
        b[i] = float(lines[2 + n + i])

    return A, b, eps, n


def check_diagonal_dominance(A, n):
    for i in range(n):
        off = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= off:
            return False
    return True


def check_zero_diagonal(A, n, method_name="МПИ"):
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ZeroDivisionError(
                f"Нулевой диагональный элемент a[{i}][{i}] — метод {method_name} неприменим"
            )


def solve_lower_triangular(M, Bmat, n):
    k = Bmat.shape[1]
    X = np.zeros((n, k))
    for i in range(n):
        if abs(M[i, i]) < 1e-15:
            raise ZeroDivisionError(f"Нулевой элемент M[{i}][{i}]")
        for j in range(k):
            s = sum(M[i, p] * X[p, j] for p in range(i))
            X[i, j] = (Bmat[i, j] - s) / M[i, i]
    return X



def jacobi(A, b, eps, n, max_iter=10000, verbose=True):
    check_zero_diagonal(A, n, "Якоби")

    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    for i in range(n):
        beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]

    norm_alpha = np.max(np.sum(np.abs(alpha), axis=1))
    if norm_alpha >= 1:
        print(f"⚠  ||alpha_J||_inf = {norm_alpha:.6f} >= 1 — сходимость не гарантирована")
    else:
        print(f"   ||alpha_J||_inf = {norm_alpha:.6f} < 1 — метод сходится")

    x = beta.copy()
    history = [x.copy()]

    for k in range(1, max_iter + 1):
        x_new = beta + alpha @ x

        diff = np.max(np.abs(x_new - x))
        history.append(x_new.copy())

        if verbose:
            print(f"   k = {k:>3}   x = {x_new}"
                  f"   ||Δx|| = {diff:.3e}")

        x = x_new

        if diff < eps:
            return x, k, history

    raise RuntimeError(f"Метод Якоби не сошёлся за {max_iter} итераций")


def seidel(A, b, eps, n, max_iter=10000, verbose=True):
    check_zero_diagonal(A, n, "Зейделя")

    D_inv = np.diag(1.0 / np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    B = -D_inv @ L
    C = -D_inv @ U
    E_minus_B = np.eye(n) - B

    alpha_S = solve_lower_triangular(E_minus_B, C, n)

    norm_alpha = np.max(np.sum(np.abs(alpha_S), axis=1))
    norm_C = np.max(np.sum(np.abs(C), axis=1))

    if norm_alpha >= 1:
        print(f"  ||alpha_S||_inf = {norm_alpha:.6f} >= 1 — сходимость не гарантирована")
    else:
        print(f"   ||alpha_S||_inf = {norm_alpha:.6f} < 1 — метод сходится")
    print(f"   ||C||_inf = {norm_C:.6f}   "
          f"множитель оценки погрешности = {norm_C / (1 - norm_alpha):.6f}")

    beta = np.zeros(n)
    for i in range(n):
        beta[i] = b[i] / A[i, i]
    x = beta.copy()
    history = [x.copy()]

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            s_new = sum(A[i, j] * x[j] for j in range(i))
            s_old = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s_new - s_old) / A[i, i]

        diff = np.max(np.abs(x - x_old))
        history.append(x.copy())

        if norm_alpha < 1:
            eps_k = norm_C / (1 - norm_alpha) * diff
        else:
            eps_k = diff

        if verbose:
            print(f"   k = {k:>3}   x = {x}"
                  f"   ||Δx|| = {diff:.3e}   ε^(k) = {eps_k:.3e}")

        if diff < eps:
            return x, k, history

    raise RuntimeError(f"Метод Зейделя не сошёлся за {max_iter} итераций")


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "input.txt")
    A, b, eps, n = read_system(filename)

    print("=" * 60)
    print("Исходные данные:")
    print(f"  n   = {n}")
    print(f"  eps = {eps}")
    print("  A =")
    print(A)
    print("  b =", b)

    if check_diagonal_dominance(A, n):
        print("\n Строгое диагональное преобладание выполнено — сходимость гарантирована")
    else:
        print("\n Диагонального преобладания нет — сходимость под вопросом")

    print("\n" + "=" * 60)
    print("МЕТОД ЯКОБИ")
    print("=" * 60)
    try:
        x_jac, iters_jac, _ = jacobi(A, b, eps, n, verbose=True)
        print(f"\nРешение (Якоби): {x_jac}")
        print(f"Итераций:        {iters_jac}")
    except RuntimeError as e:
        print(f"Ошибка: {e}")
        x_jac, iters_jac = None, None

    print("\n" + "=" * 60)
    print("МЕТОД ЗЕЙДЕЛЯ")
    print("=" * 60)
    try:
        x_sei, iters_sei, _ = seidel(A, b, eps, n, verbose=True)
        print(f"\nРешение (Зейдель): {x_sei}")
        print(f"Итераций:          {iters_sei}")
    except RuntimeError as e:
        print(f"Ошибка: {e}")
        x_sei, iters_sei = None, None

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ")
    print("=" * 60)
    try:
        x_exact = np.linalg.solve(A, b)
        print(f"numpy:              {x_exact}")

        if x_jac is not None:
            print(f"Ошибка (Якоби):     ||x - x*||_inf = {np.max(np.abs(x_jac - x_exact)):.3e}")
            print(f"Невязка (Якоби):    ||Ax - b||_inf = {np.max(np.abs(A @ x_jac - b)):.3e}")

        if x_sei is not None:
            print(f"Ошибка (Зейдель):   ||x - x*||_inf = {np.max(np.abs(x_sei - x_exact)):.3e}")
            print(f"Невязка (Зейдель):  ||Ax - b||_inf = {np.max(np.abs(A @ x_sei - b)):.3e}")

        if x_jac is not None and x_sei is not None:
            print(f"\nОтношение итераций: Зейдель/Якоби = {iters_sei}/{iters_jac} "
                  f"≈ {iters_sei / iters_jac:.2f}")

    except np.linalg.LinAlgError:
        print("Матрица вырождена — сравнение с numpy невозможно")
        if x_jac is not None:
            print(f"Невязка (Якоби):    ||Ax - b||_inf = {np.max(np.abs(A @ x_jac - b)):.3e}")
        if x_sei is not None:
            print(f"Невязка (Зейдель):  ||Ax - b||_inf = {np.max(np.abs(A @ x_sei - b)):.3e}")