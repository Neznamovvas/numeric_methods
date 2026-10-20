import numpy as np


def read_system_from_file(filename):
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            row = f.readline().split()
            if len(row) != n:
                raise ValueError(f"Строка {i+1}: ожидалось {n} чисел, получено {len(row)}")
            A[i] = [float(x) for x in row]
        b = np.zeros((n, 1), dtype=float)
        for i in range(n):
            b[i, 0] = float(f.readline().strip())
    return A, b


def LUP(A, n):
    U = A.astype(float, copy=True)
    P = np.eye(n)
    L = np.zeros((n, n))
    perm = 0

    for i in range(n):
        pivotValue = -1
        pivot = 0

        for k in range(i, n):
            if abs(U[k, i]) > pivotValue:
                pivotValue = abs(U[k, i])
                pivot = k

        assert pivotValue > 1e-15, "Матрица вырождена"

        if pivot != i:
            P[[i, pivot]] = P[[pivot, i]]
            U[[i, pivot]] = U[[pivot, i]]
            L[[i, pivot]] = L[[pivot, i]]
            perm += 1

        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            for k in range(i, n):
                U[j, k] -= L[j, i] * U[i, k]

    for i in range(n):
        L[i, i] = 1.0

    return L, U, P, perm


def solve(A, b, n):
    L, U, P, _ = LUP(A, n)
    b1 = (P @ b).ravel()
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = b1[i] - sum(L[i, k] * y[k] for k in range(i))

    for i in range(n - 1, -1, -1):
        x[i] = 1 / U[i, i] * (y[i] - sum(U[i, k] * x[k] for k in range(i + 1, n)))

    return x


def inverse(A, n):
    A_inv = np.zeros((n, n))

    for j in range(n):
        e_j = np.zeros((n, 1))
        e_j[j, 0] = 1.0
        x = solve(A, e_j, n)
        A_inv[:, j] = x

    return A_inv


def triangular_determinant(M, n):
    return np.prod([M[i, i] for i in range(n)])


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "input.txt")
    A, b = read_system_from_file(filename)
    n = A.shape[0]

    L, U, P, perm = LUP(A, n)

    print("A:\n", A)
    print("\nL:\n", L)
    print("\nU:\n", U)
    print("\nL @ U\n", L @ U)
    print("\nP @ A:\n", P @ A)

    det = (-1) ** perm * triangular_determinant(U, n)
    print("\nОпределитель (LU):", det)
    print("Определитель (numpy):", np.linalg.det(A))

    x = solve(A, b, n)
    print("\nРешение (LU):", x)
    print("Решение (numpy):", np.linalg.solve(A, b).ravel())

    A_inv = inverse(A, n)
    print("\nОбратная матрица:\n", A_inv)
    print("Обратная матрица (numpy):\n", np.linalg.inv(A))