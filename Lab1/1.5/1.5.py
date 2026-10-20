import numpy as np


def read_matrix(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    eps = float(lines[1])

    A = np.zeros((n, n))
    for i in range(n):
        row = lines[2 + i].split()
        if len(row) != n:
            raise ValueError(f"Строка {2 + i + 1}: ожидалось {n} чисел, получено {len(row)}")
        A[i] = [float(x) for x in row]

    return A, eps, n


def householder_vector(b):
    v = b.copy()
    norm_b = np.sqrt(np.sum(b ** 2))
    if b[0] >= 0:
        v[0] = b[0] + norm_b
    else:
        v[0] = b[0] - norm_b
    return v



def qr_householder(A, n):
    R = A.astype(float, copy=True)
    Q = np.eye(n)
    
    for k in range(n - 1):
        b = R[k:, k].copy()
        v = householder_vector(b)
        vTv = v @ v
        if vTv < 1e-15:
            continue
        
        R[k:, :] -= (2.0 / vTv) * np.outer(v, v @ R[k:, :])
        Q[:, k:] -= (2.0 / vTv) * np.outer(Q[:, k:] @ v, v)
    
    return Q, R


def get_eigenvalues_from_quasi_triangular(Ak, n, eps=1e-8):
    eigenvalues = []
    i = 0
    while i < n:
        if i < n - 1 and abs(Ak[i + 1, i]) > eps:
            a = Ak[i, i]
            b = Ak[i, i + 1]
            c = Ak[i + 1, i]
            d = Ak[i + 1, i + 1]

            tr = a + d
            det = a * d - b * c
            disc = tr * tr - 4 * det

            if disc >= 0:
                l1 = (tr + np.sqrt(disc)) / 2
                l2 = (tr - np.sqrt(disc)) / 2
                eigenvalues.append(complex(l1, 0))
                eigenvalues.append(complex(l2, 0))
            else:
                re = tr / 2
                im = np.sqrt(-disc) / 2
                eigenvalues.append(complex(re,  im))
                eigenvalues.append(complex(re, -im))
            i += 2
        else:
            eigenvalues.append(complex(Ak[i, i], 0))
            i += 1
    return np.array(eigenvalues)


def qr_algorithm(A, eps, n, max_iter=10000, verbose=True):
    Ak = A.astype(float, copy=True)
    prev_lambdas = None
    k = 0

    for k in range(1, max_iter + 1):
        Q, R = qr_householder(Ak, n)
        Ak = R @ Q

        cur_lambdas = get_eigenvalues_from_quasi_triangular(Ak, n, eps=eps)

        if verbose and (k <= 5 or k % 20 == 0):
            lambdas_str = ", ".join(f"{l.real:>7.3f}{l.imag:+.3f}j" for l in cur_lambdas)
            print(f"   итерация {k:>4}: λ = [{lambdas_str}]")

        if prev_lambdas is not None:
            sorted_cur = np.sort_complex(cur_lambdas)
            sorted_prev = np.sort_complex(prev_lambdas)
            diff = np.max(np.abs(sorted_cur - sorted_prev))
            
            if diff < eps:
                if verbose:
                    print(f"   Сошлось на итерации {k} (изменение λ < {eps})")
                break
        prev_lambdas = cur_lambdas
    else:
        if verbose:
            print(f"   Достигнут лимит итераций ({max_iter})")

    eigenvalues = get_eigenvalues_from_quasi_triangular(Ak, n, eps=eps)
    return eigenvalues, Ak, k


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "input.txt")
    A, eps, n = read_matrix(filename)

    print("=" * 60)
    print("ИСХОДНАЯ МАТРИЦА A")
    print("=" * 60)
    for row in A:
        print("   ", "  ".join(f"{x:>12.6f}" for x in row))

    print("\n" + "=" * 60)
    print("QR-РАЗЛОЖЕНИЕ (метод Хаусхолдера)")
    print("=" * 60)

    Q, R = qr_householder(A, n)

    print("\nМатрица Q:")
    for row in Q:
        print("   ", "  ".join(f"{x:>12.6f}" for x in row))

    print("\nМатрица R:")
    for row in R:
        print("   ", "  ".join(f"{x:>12.6f}" for x in row))

    QR = Q @ R
    err_QR = np.max(np.abs(QR - A))

    print("\nРезультат умножения Q · R:")
    for row in QR:
        print("   ", "  ".join(f"{x:>12.6f}" for x in row))

    print(f"\n‖Q·R − A‖_∞ = {err_QR:.3e}")
    err_orth = np.max(np.abs(Q.T @ Q - np.eye(n)))
    print(f"‖Qᵀ·Q − E‖_∞ = {err_orth:.3e}")

    print("\n" + "=" * 60)
    print("QR-АЛГОРИТМ ПОИСКА СОБСТВЕННЫХ ЗНАЧЕНИЙ")
    print("=" * 60)
    print("Ход итераций:")

    eigenvalues, Ak, iters = qr_algorithm(A, eps, n, verbose=True)

    print(f"\nВсего итераций: {iters}")
    print("\nФинальная матрица A_k:")
    for row in Ak:
        print("   ", "  ".join(f"{x:>12.6f}" for x in row))

    print("\n" + "=" * 60)
    print("СОБСТВЕННЫЕ ЗНАЧЕНИЯ (QR-алгоритм)")
    print("=" * 60)
    for i, lam in enumerate(eigenvalues):
        if abs(lam.imag) < 1e-10:
            print(f"  λ[{i}] = {lam.real:>14.10f}")
        else:
            print(f"  λ[{i}] = {lam.real:>14.10f} {lam.imag:+.10f}j")

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ С numpy.linalg.eigvals")
    print("=" * 60)
    eigvals_np = np.linalg.eigvals(A)
    for i, lam in enumerate(eigvals_np):
        if abs(lam.imag) < 1e-10:
            print(f"  λ[{i}] = {lam.real:>14.10f}")
        else:
            print(f"  λ[{i}] = {lam.real:>14.10f} {lam.imag:+.10f}j")