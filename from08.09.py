import numpy as np
import random
from scipy.linalg import solve
import sys

class Matrix:
    def __init__(self, data, vector_type='row'):
        if not isinstance(data[0], list):
            if vector_type == 'row':
                data = [data]
            elif vector_type == 'col':
                data = [[x] for x in data]
            else:
                raise ValueError("vector_type должен быть 'row' или 'col'")

        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("Все строки должны иметь одинаковую длину")

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __add__(self, other):
        if self.rows == other.rows and self.cols == other.cols:
            return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
        else:
            raise ValueError("Несовместимые размеры матриц")

    def __mul__(self, other):
        if self.cols == other.rows:
            other_T = other.transpose()
            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    dot_product = sum(self.data[i][k] * other_T.data[j][k] for k in range(self.cols))
                    row.append(dot_product)
                result.append(row)
            return Matrix(result)
        else:
            raise ValueError("Несовместимые размеры матриц")

    def __sub__(self, other):
        if self.rows == other.rows and self.cols == other.cols:
            return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
        else:
            raise ValueError("Несовместимые размеры матриц")

    def transpose(self):
        transposed_data = [[self.data[i][j] for i in range(self.rows)] for j in range(self.cols)]
        return Matrix(transposed_data)

    def normL1(self):
        if self.rows == 1:
            return sum(abs(self.data[0][i]) for i in range(self.cols))
        else:
            return max(sum(abs(self.data[i][j]) for i in range(self.rows)) for j in range(self.cols))

    def normL2(self):
        if self.rows == 1:
            return np.sqrt((self * self.transpose()).data[0][0])
        elif self.rows == self.cols:
            mat = self.transpose() * self
            mat = np.array(mat.data)
            eigenvalues = np.linalg.eigvals(mat)
            return np.sqrt(max(eigenvalues))
        else:
            raise ValueError("Недопустимые размеры матриц")

    def norm_infty(self):
        if self.rows == 1:
            return max(abs(x) for x in self.data[0])
        else:
            return max(sum(abs(self.data[i][j]) for j in range(self.rows)) for i in range(self.cols))

    def print(self):
        for i in self.data:
            for j in i:
                print(j, end=" ")
            print()

    def cond(self, type_norm=1):
        if type_norm == 1:
            cond = self.norm_1() * self.inverse(1).norm_1()
        elif type_norm == 'infty':
            cond = self.norm_infty() * self.inverse(1).norm_infty()
        elif type_norm == 2:
            cond = self.norm_2() * self.inverse(1).norm_2()
        else:
            raise ValueError("type_norm должен быть 1, 'infty' или 2")
        return cond

    def det(self):
        if self.rows != self.cols:
            raise ValueError("Матрица должна быть квадратной")
        return np.linalg.det(np.array(self.data))

    def is_positive_definite(self):
        if self.rows != self.cols:
            raise ValueError("Матрица должна быть квадратной")
        A = np.array(self.data)
        n = self.rows
        for i in range(n):
            if np.linalg.det(A[:i + 1, :i + 1]) <= 0:
                return False
        return True


    def inverse(self, pivot=1):
        if self.rows != self.cols:
            raise ValueError("Матрица должна быть квадратной")

        n = self.rows
        A = [row[:] for row in self.data]
        I = [[float(i == j) for j in range(n)] for i in range(n)]

        for k in range(n):

            if pivot == 1:

                max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
                if A[max_row][k] == 0:
                    raise ValueError("Матрица вырождена")
                A[k], A[max_row] = A[max_row], A[k]
                I[k], I[max_row] = I[max_row], I[k]

            elif pivot == 2:

                max_col = max(range(k, n), key=lambda j: abs(A[k][j]))
                if A[k][max_col] == 0:
                    raise ValueError("Матрица вырождена")
                for row in A:
                    row[k], row[max_col] = row[max_col], row[k]
                for row in I:
                    row[k], row[max_col] = row[max_col], row[k]

            elif pivot == 3:
                i_max, j_max = max(
                    ((i, j) for i in range(k, n) for j in range(k, n)),
                    key=lambda x: abs(A[x[0]][x[1]])
                )
                if A[i_max][j_max] == 0:
                    raise ValueError("Матрица вырождена")

                A[k], A[i_max] = A[i_max], A[k]
                I[k], I[i_max] = I[i_max], I[k]

                for rowA, rowI in zip(A, I):
                    rowA[k], rowA[j_max] = rowA[j_max], rowA[k]
                    rowI[k], rowI[j_max] = rowI[j_max], rowI[k]

            else:
                raise ValueError("pivot должен быть 1, 2 или 3")

            pivot_val = A[k][k]
            for j in range(n):
                A[k][j] /= pivot_val
                I[k][j] /= pivot_val
            for i in range(n):
                if i != k:
                    factor = A[i][k]
                    for j in range(n):
                        A[i][j] -= factor * A[k][j]
                        I[i][j] -= factor * I[k][j]

        return Matrix(I)


def LUP(matrix, b, pivot_strategy=1):
    if matrix.rows != matrix.cols:
        raise ValueError("Матрица должна быть квадратной")
    if pivot_strategy not in [1, 2, 3]:
        raise ValueError("Стратегия должна быть 1, 2 или 3")

    A = [row[:] for row in matrix.data]
    b = [row[0] for row in b.data]
    n = len(A)

    counter = 0
    col_history = list(range(n))

    for i in range(n):
        if pivot_strategy == 1:  # Выбор по строкам
            max_row = i
            max_val = abs(A[i][i])

            for k in range(i + 1, n):
                if abs(A[k][i]) > max_val:
                    max_val = abs(A[k][i])
                    max_row = k

            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                b[i], b[max_row] = b[max_row], b[i]

        elif pivot_strategy == 2:  # Выбор по столбцам
            max_col = i
            max_val = abs(A[i][i])
            for k in range(i + 1, n):
                if abs(A[i][k]) > max_val:
                    max_val = abs(A[i][k])
                    max_col = k

            if max_col != i:
                for k in range(n):
                    A[k][i], A[k][max_col] = A[k][max_col], A[k][i]
                col_history[i], col_history[max_col] = col_history[max_col], col_history[i]

        elif pivot_strategy == 3:  # Полный выбор
            max_row, max_col = i, i
            max_val = abs(A[i][i])
            for k in range(i, n):
                for l in range(i, n):
                    if abs(A[k][l]) > max_val:
                        max_val = abs(A[k][l])
                        max_row, max_col = k, l

            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                b[i], b[max_row] = b[max_row], b[i]

            if max_col != i:
                for k in range(n):
                    A[k][i], A[k][max_col] = A[k][max_col], A[k][i]
                col_history[i], col_history[max_col] = col_history[max_col], col_history[i]

        for j in range(i + 1, n):
            # L
            A[j][i] = A[j][i] / A[i][i]
            counter += 1

            # U
            for k in range(i + 1, n):
                A[j][k] = A[j][k] - A[j][i] * A[i][k]
                counter += 2
    y = [0.0] * n
    for i in range(n):
        y[col_history[i]] = b[i]
        for j in range(i):
            y[col_history[i]] -= A[i][j] * y[col_history[j]]
            counter += 1

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[col_history[i]] = y[col_history[i]]
        for j in range(i + 1, n):
            x[col_history[i]] -= A[i][j] * x[col_history[j]]
            counter += 1
        x[col_history[i]] = x[col_history[i]] / A[i][i]
        counter += 1

    return x, counter

def gauss(n, A, x_original, pivot_strategy=1):
    b = [0] * n
    for i in range(n):
        for j in range(n):
            b[i] += A[i][j] * x_original[j]

    if pivot_strategy not in [1, 2, 3]:
        raise ValueError("Стратегия должна быть 1, 2 или 3")

    A_work = [row[:] for row in A]
    b_work = b[:]

    col_history = list(range(n))
    counter = 0
    det = 1

    for i in range(n):
        if pivot_strategy == 1:
            max_row = i
            max_val = abs(A_work[i][i])
            for k in range(i + 1, n):
                if abs(A_work[k][i]) > max_val:
                    max_val = abs(A_work[k][i])
                    max_row = k
                    #counter += 1

            if max_row != i:
                A_work[i], A_work[max_row] = A_work[max_row], A_work[i]
                b_work[i], b_work[max_row] = b_work[max_row], b_work[i]
                #counter += 3 * n

        elif pivot_strategy == 2:
            max_col = i
            max_val = abs(A_work[i][i])
            for k in range(i + 1, n):
                if abs(A_work[i][k]) > max_val:
                    max_val = abs(A_work[i][k])
                    max_col = k
                    #counter += 1

            if max_col != i:
                for k in range(n):
                    A_work[k][i], A_work[k][max_col] = A_work[k][max_col], A_work[k][i]
                col_history[i], col_history[max_col] = col_history[max_col], col_history[i]
                #counter += 3 * n
        elif pivot_strategy == 3:
            max_row, max_col = i, i
            max_val = abs(A_work[i][i])
            for k in range(i, n):
                for l in range(i, n):
                    if abs(A_work[k][l]) > max_val:
                        max_val = abs(A_work[k][l])
                        max_row, max_col = k, l
                    #counter += 1

            if max_row != i:
                A_work[i], A_work[max_row] = A_work[max_row], A_work[i]
                b_work[i], b_work[max_row] = b_work[max_row], b_work[i]
                #counter += 3 * n

            if max_col != i:
                for k in range(n):
                    A_work[k][i], A_work[k][max_col] = A_work[k][max_col], A_work[k][i]
                col_history[i], col_history[max_col] = col_history[max_col], col_history[i]
                #counter += 3 * n

        for k in range(i + 1, n):
            factor = A_work[k][i] / A_work[i][i]
            for j in range(i, n):
                A_work[k][j] -= factor * A_work[i][j]
                counter += 2
            b_work[k] -= factor * b_work[i]
            counter += 2

        det *= A_work[i][i]

    x_calculated = [0] * n
    for i in range(n - 1, -1, -1):
        x_calculated[col_history[i]] = b_work[i]
        for j in range(i + 1, n):
            x_calculated[col_history[i]] -= A_work[i][j] * x_calculated[col_history[j]]
            counter += 2
        x_calculated[col_history[i]] /= A_work[i][i]
        counter += 1

    residual = 0
    for i in range(n):
        sum_ax = 0
        for j in range(n):
            sum_ax += A[i][j] * x_calculated[j]
        residual += abs(b[i] - sum_ax)

    return b, x_calculated, x_original, det, counter, residual

def test_inverse(n):
    A_np = np.random.randint(-5, 5, (n, n)).astype(float)
    for i in range(n):
        A_np[i, i] = np.sum(np.abs(A_np[i])) + 1

    matrix_data = A_np.tolist()
    matrix = Matrix(matrix_data)

    try:
        inv_matrix = matrix.inverse(1)
        inv_matrix_np = np.linalg.inv(A_np)

        return matrix_data, inv_matrix.data, inv_matrix_np.tolist()

    except Exception as e:
        print(f"Ошибка при вычислении обратной матрицы: {e}")
        return matrix_data, None, None

def test1(n):
    A_data = [[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)]

    for i in range(n):
        A_data[i][i] += 0.0000001

    A = Matrix(A_data)

    x_true_data = np.random.uniform(-10, 10, n)
    x_true = Matrix(x_true_data.tolist(), 'col')

    b = A * x_true

    x_gauss = gauss(n, A_data, x_true_data)[1]
    counter = gauss(n, A_data, x_true_data)[4]

    A_np = np.array(A_data)
    b_np = np.array(b.data).flatten()
    x_np = solve(A_np, b_np)

    temp_x_true = Matrix([x_true.data[i][0] for i in range(n)])

    difference_gauss = (temp_x_true - Matrix(x_gauss)).normL2()

    cond = A.normL1() * A.inverse(1).normL1()

    return x_gauss, x_np, x_true_data, difference_gauss, cond, counter


def test2(n):
    A_data = [[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)]

    for i in range(n):
        A_data[i][i] += 0.0000001

    A = Matrix(A_data)


    x_true_data = np.random.uniform(-10, 10, n)
    x_true = Matrix(x_true_data.tolist(), 'col')

    b = A * x_true

    x_lup_list, counter = LUP(A, b, 3)

    x_lup = Matrix([x_lup_list])

    x_true_row = Matrix([[x_true.data[i][0] for i in range(n)]])

    difference = x_true_row - x_lup

    difference_lup = difference.normL2()

    A_np = np.array(A_data)
    b_np = np.array([row[0] for row in b.data])
    x_np = np.linalg.solve(A_np, b_np)

    A_inv = A.inverse(1)
    cond = A.normL2() * A_inv.normL2()

    return x_lup_list, x_np, x_true_data, difference_lup, cond, counter

def th_number(n):
    return 2 / 3 * n ** 3 + 3 / 2 * n ** 2 - 7 / 6 * n

# _______________________________________________________________________________________________
def is_symmetric(matrix, tol=1e-10):
    if matrix.rows != matrix.cols:
        return False
    for i in range(matrix.rows):
        for j in range(i + 1, matrix.cols):
            if abs(matrix.data[i][j] - matrix.data[j][i]) > tol:
                return False
    return True

def square(matrix, b):
    if matrix.rows != matrix.cols:
        raise ValueError("Матрица должна быть квадратной")
    # if not matrix.is_positive_definite():
    #     raise ValueError("Матрица должна быть положительно определенной")

    A = [row[:] for row in matrix.data]
    b = [row[0] for row in b.data]
    n = len(A)

    counter = 0

    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                L[i][i] = np.sqrt(A[i][i] - sum(L[k][i] ** 2 for k in range(i)))
                counter+=2* i+1 + 1
            else:
                L[i][j] = (A[i][j] - sum(L[k][i] * L[k][j] for k in range(i))) / L[i][i]
                counter+=2* i+1 + 1

    ## Решение Ly = b
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[j][i] * y[j]
            counter += 1
        y[i] /= L[i][i]
        counter += 1

    ## Решение L^Tx = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= L[i][j] * x[j]
            counter += 1
        x[i] /= L[i][i]
        counter += 1

    return x, counter


def Tomas(matrix, b):
    if matrix.rows != matrix.cols:
        raise ValueError("Матрица должна быть квадратной")
    n = matrix.rows
    if not all(matrix.data[i][j] == 0 for i in range(n) for j in range(n) if abs(i - j) > 1):
        raise ValueError("Матрица должна быть трехдиагональной")

    A = [[float(x) for x in row] for row in matrix.data]
    b = [float(row[0]) for row in b.data]

    counter = 0

    alpha = [0.0] * (n - 1)
    beta = [0.0] * n

    alpha[0] = A[0][1] / A[0][0]
    beta[0] = b[0] / A[0][0]
    counter += 3

    for i in range(1, n - 1):
        denom = A[i][i] - A[i][i - 1] * alpha[i - 1]
        alpha[i] = A[i][i + 1] / denom
        beta[i] = (b[i] - A[i][i - 1] * beta[i - 1]) / denom
        counter += 5

    denom = A[n - 1][n - 1] - A[n - 1][n - 2] * alpha[n - 2]
    beta[n - 1] = (b[n - 1] - A[n - 1][n - 2] * beta[n - 2]) / denom

    x = [0.0] * n
    x[-1] = beta[-1]
    counter += 1
    for i in range(n - 2, -1, -1):
        x[i] = beta[i] - alpha[i] * x[i + 1]
        counter += 2

    return x, counter

def generate_test_cases(n):

    L_data = np.tril(np.random.randint(1, 5, (n, n)))
    while np.linalg.det(L_data) == 0:
        L_data = np.tril(np.random.randint(1, 5, (n, n)))
    L_data = L_data.tolist()
    LT_data = Matrix(L_data).transpose().data
    A_good = (Matrix(L_data) * Matrix(LT_data)).data

    A_non_symmetric = np.random.randint(-5, 5, (n, n)).astype(float)
    A_non_symmetric = A_non_symmetric.tolist()

    A_symmetric_bad = np.random.randint(-5, 5, (n, n)).astype(float)
    A_symmetric_bad = (A_symmetric_bad + A_symmetric_bad.T) / 2
    # Делаем матрицу отрицательно определенной
    for i in range(n):
        A_symmetric_bad[i][i] = -np.sum(np.abs(A_symmetric_bad[i])) - 1
    A_symmetric_bad = A_symmetric_bad.tolist()

    return A_good, A_non_symmetric, A_symmetric_bad

# def task7_3(n):
#     L_data = np.tril(np.random.randint(1,5, (n, n)))
#     while np.linalg.det(L_data) == 0:
#         L_data = np.tril(np.random.randint(1, 5, (n, n)))
#     L_data = L_data.tolist()
#     LT_data = Matrix(L_data).transpose().data
#     A_data = (Matrix(L_data) * Matrix(LT_data)).data
#     A = Matrix(A_data)
#
#     ##A_right = A_data.copy()
#     ##A_wrong = A_data.copy()
#     ##A_wrong[0][0] = -1*n
#
#     x_true_data = np.random.uniform(-10, 10, n)
#     x_true = Matrix(x_true_data.tolist(), 'col')
#
#     b = A * x_true
#
#     x_square, counter1 = square(A, b)
#     counter2 = LUP(A, b, 1)[1]
#
#     x_square = Matrix(x_square, 'col')
#
#     diff = (x_square - x_true).normL1()
#     return x_square, x_true_data, counter1, counter2, diff

def task7_4(n):
    print("ТЕСТ МЕТОДА ТОМАСА\n")

    A_data = [[0.0]*n for _ in range(n)]
    for i in range(n):
        A_data[i][i] = np.random.uniform(4.0, 6.0)
        if i > 0:
            A_data[i][i - 1] = np.random.uniform(0.5, 2.0)
        if i < n - 1:
            A_data[i][i + 1] = np.random.uniform(0.5, 2.0)

    A = Matrix(A_data)

    print("Матрица A:")
    Matrix(A_data).print()

    x_true_data = np.random.uniform(-10, 10, n)
    x_true = Matrix(x_true_data.tolist(), 'col')

    print("\nВектор x_true:")
    for i in range(n):
        print(f"{x_true_data[i]:.4f}", end=" ")
    print()

    b = A * x_true

    print("\nВектор b:")
    for i in range(n):
        print(f"{b.data[i][0]:.4f}", end=" ")
    print("\n")

    x_tomas, counter1 = Tomas(A, b)
    counter2 = LUP(A, b, 1)[1]

    x_tomas_vec = Matrix([[x] for x in x_tomas])

    diff  = (x_tomas_vec - x_true).normL1()

    print(type(diff))
    print(sys.getsizeof(diff))

    print("Решение методом Томаса:")
    for i in range(n):
        print(f"{x_tomas[i]:.4f}", end=" ")
    print(f"\n\nОшибка: {diff:.2e}")
    print(f"Операций Томаса: {counter1}")
    print(f"Операций LUP: {counter2}")
    print("Погрешность:", diff)

    return x_tomas, x_true, counter1, counter2, diff


def test_cholesky_cases(n=20):
    A_good, A_non_sym, A_sym_bad = generate_test_cases(n)

    print("ТЕСТ МЕТОДА ХОЛЕЦКОГО\n")
    print("1. Матрица:")
    Matrix(A_good).print()

    A = Matrix(A_good)
    x_true = Matrix(np.random.uniform(-10, 10, n).tolist(), 'col')
    b = A * x_true

    symmetric = is_symmetric(A)
    positive_def = A.is_positive_definite()

    if symmetric and positive_def:
        x_cholesky, ops_cholesky = square(A, b)
        x_lup, ops_lup = LUP(A, b, 1)

        x_cholesky_vec = Matrix([[x] for x in x_cholesky])
        x_lup_vec = Matrix([[x] for x in x_lup])

        diff1 = (x_cholesky_vec - x_true).normL1()
        diff2 = (x_lup_vec - x_true).normL1()

        print(f"Успешно решена")
        print(f"Операций Холецкого: {ops_cholesky}")
        print(f"Операций LUP: {ops_lup}")
        print("Погрешность Holec:", diff1)
        print("Погрешность LUP:", diff2)
        print("")
    else:
        print(f"Ошибка: матрица не удовлетворяет условиям\n")

    print("2. Несимметричная матрица:")
    Matrix(A_non_sym).print()

    A = Matrix(A_non_sym)
    symmetric = is_symmetric(A)

    if not symmetric:
        print("Неприменим: матрица не симметрична")
        print("A ≠ A_transpose\n")
    else:
        print("Ошибка: матрица симметрична, но не должна быть\n")

    print("3. Симметричная, но не положительно определенная:")
    Matrix(A_sym_bad).print()

    A = Matrix(A_sym_bad)
    symmetric = is_symmetric(A)
    # positive_def = A.is_positive_definite()

    x_cholesky, ops_cholesky = square(A, b)
    x_lup, ops_lup = LUP(A, b, 1)

    x_cholesky_vec = Matrix([[x] for x in x_cholesky])
    x_lup_vec = Matrix([[x] for x in x_lup])

    for i in range(len(x_cholesky)):
        print(f"{x_cholesky[i]:.6f}", end=" ")
    print()

    print("LUP:")
    for i in range(len(x_lup)):
        print(f"{x_lup[i]:.6f}", end=" ")
    print()

    diff = (x_cholesky_vec - x_true).normL1()

    print(f"Операций Холецкого: {ops_cholesky}")
    print(f"Операций LUP: {ops_lup}")
    print("Погрешность:", diff)

    # if symmetric and not positive_def:
    #     print("Неприменим: матрица не положительно определена")
    #     print("Cуществуют главные миноры ≤ 0\n")
    # else:
    #     print("Ошибка: матрица либо не симметрична, либо положительно определена\n")

# Тест метода Гаусса
# print("введите n для теста Гаусса")
# n = int(input())
#
# A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
# x = [random.uniform(-10, 10) for _ in range(n)]
# b, x_calculated, x_original, det, counter, residual = gauss(n, A, x)
# err = 0
# for i in range(n):
#     err += (x_calculated[i] - x_original[i]) ** 2
# print("x_calculated = ", x_calculated)
# print("x_original = ", x_original)
# print("|err| = ", err ** 0.5)
# print("det = ", det)
# print("counter =", counter)
# print("|counter - theoretical counter| = ", abs(counter - th_number(n)))
# print("residual = ", residual)
#
# A_matrix = Matrix(A)
# A_inv = A_matrix.inverse(1)
# cond = A_matrix.normL2() * A_inv.normL2()
# print("cond =", cond)

#
# test_1 = test1(15)
# print("Gauss:")
# print("x_calculated = ",test_1[0])
# print("x_exact = ",test_1[2])
# print("diff = ",test_1[3])
# print("cond = ",test_1[4])
# print("counter = ", test_1[5])
# print(" ")
#
# test_2 = test2(15)
# print("LUP:")
# print("x_calculated = ",test_2[0])
# print("x_exact = ",test_2[2])
# print("diff = ",test_2[3])
# print("cond = ",test_2[4])
# print("counter = ", test_2[5])
#
# print("counter: Gauss - LUP =", test_1[5]-test_2[5])

# test = test_inverse(3)
# Matrix(test[0]).print()
# print()
# Matrix(test[1]).print()
# print()
# Matrix(test[2]).print()
# print()

# ТУТ МЕТОД ХОЛЕЦКОГО
test_cholesky_cases(20)

#ТУТ МЕТОД ТОМАСА
# test = task7_4(10)