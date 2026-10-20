import numpy as np


def read_symmetric_matrix(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    eps = float(lines[1])

    A = np.zeros((n, n))
    for i in range(n):
        row = lines[2 + i].split()
        if len(row) != n:
            raise ValueError(
                f"Строка {2 + i + 1}: ожидалось {n} чисел, получено {len(row)}"
            )
        A[i] = [float(x) for x in row]

    return A, eps, n


def check_symmetric(A, n, tol=1e-10):
    return np.max(np.abs(A - A.T)) < tol



def jacobi_rotation_method(A, eps, n, max_iter=10000, verbose=True):
    A = A.astype(float, copy=True)
    V = np.eye(n)

    def off_diag_norm():
        return np.sqrt(np.sum(A ** 2) - np.sum(np.diag(A) ** 2))

    k = 0
    while off_diag_norm() >= eps and k < max_iter:
        k += 1

        max_val = 0.0
        p, q = 0, 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j


        if abs(A[p, p] - A[q, q]) < 1e-15:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2.0 * A[p, q] / (A[p, p] - A[q, q]))

        c = np.cos(phi)
        s = np.sin(phi)

        for i in range(n):
            aip = A[i, p]
            aiq = A[i, q]
            A[i, p] = c * aip + s * aiq
            A[i, q] = -s * aip + c * aiq

        for j in range(n):
            apj = A[p, j]
            aqj = A[q, j]
            A[p, j] = c * apj + s * aqj
            A[q, j] = -s * apj + c * aqj  

        for i in range(n):
            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c * vip + s * viq
            V[i, q] = -s * vip + c * viq

        if verbose and (k <= 5 or k % 50 == 0):
            print(f"  итерация {k:>4}:  off(A) = {off_diag_norm():.3e},  "
                  f"поворот ({p}, {q}), φ = {phi:+.6f}")

    eigenvalues = np.diag(A).copy()
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = V[:, order]

    return eigenvalues, eigenvectors, k


if __name__ == "__main__":
    A, eps, n = read_symmetric_matrix("D:/learning/numeric_methods/Lab1/1.4/input.txt")

    print("=" * 60)
    print("Исходные данные:")
    print(f"  n    = {n}")
    print(f"  eps  = {eps}")
    print("  A =")
    for row in A:
        print("   ", "  ".join(f"{x:>10.6f}" for x in row))

    if not check_symmetric(A, n):
        print("\n Матрица не симметрична! Метод Якоби для вращений требует A = A^T.")
        raise SystemExit(1)
    print("\n Матрица симметрична — метод вращений Якоби применим")

    print("\n" + "=" * 60)
    print("МЕТОД ВРАЩЕНИЙ ЯКОБИ")
    print("=" * 60)

    eigenvalues, eigenvectors, sweeps = jacobi_rotation_method(A, eps, n, verbose=True)

    print("\n" + "=" * 60)
    print("СОБСТВЕННЫЕ ЗНАЧЕНИЯ")
    print("=" * 60)
    for i, lam in enumerate(eigenvalues):
        print(f"  λ[{i}] = {lam:.10f}")

    print("\n" + "=" * 60)
    print("СОБСТВЕННЫЕ ВЕКТОРЫ (по столбцам матрицы V)")
    print("=" * 60)
    for j in range(n):
        print(f"\n  v[{j}] (для λ = {eigenvalues[j]:.6f}):")
        for i in range(n):
            print(f"    v[{i}] = {eigenvectors[i, j]:>14.10f}")

    print("\n  Матрица собственных векторов V:")
    for row in eigenvectors:
        print("   ", "  ".join(f"{x:>12.8f}" for x in row))

    print("\n" + "=" * 60)
    print("ПРОВЕРКА  A · V = V · Λ")
    print("=" * 60)

    Lambda = np.diag(eigenvalues)
    AV = A @ eigenvectors
    VLambda = eigenvectors @ Lambda

    print("  A · V:")
    for row in AV:
        print("   ", "  ".join(f"{x:>12.8f}" for x in row))

    print("\n  V · Λ:")
    for row in VLambda:
        print("   ", "  ".join(f"{x:>12.8f}" for x in row))
