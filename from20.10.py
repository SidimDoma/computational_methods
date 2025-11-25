import numpy as np


class PowerMethod:
    def __init__(self, A, epsilon=1e-8, max_iter=1000):
        self.A = A.astype(float)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n = A.shape[0]

    def basic_power_method(self, x0=None):
        if x0 is None:
            x0 = np.random.rand(self.n)

        x_prev = x0.copy()
        lambda_history = []
        residual_history = []

        for k in range(self.max_iter):
            # Формула (2)
            x_current = self.A @ x_prev

            # Формула (3):
            lambda_k = np.dot(x_current, x_prev) / np.dot(x_prev, x_prev)

            residual = np.linalg.norm(x_current - lambda_k * x_prev) / np.linalg.norm(x_prev)

            lambda_history.append(lambda_k)
            residual_history.append(residual)

            if residual < self.epsilon and k > 0:
                break

            x_prev = x_current.copy()

        eigenvector = x_current / np.linalg.norm(x_current)

        return lambda_k, eigenvector, {
            'lambda_history': lambda_history,
            'residual_history': residual_history,
            'iterations': k + 1
        }

    def modified_power_method(self, x0=None):
        if x0 is None:
            x0 = np.random.rand(self.n)
            x0 = x0 / np.linalg.norm(x0)

        x_prev = x0.copy()
        lambda_history = []
        residual_history = []

        for k in range(self.max_iter):
            # Формула (12)
            y_current = self.A @ x_prev
            # Формула (13)
            lambda_k = np.dot(y_current, x_prev)

            x_current = y_current / np.linalg.norm(y_current)

            residual = np.linalg.norm(y_current - lambda_k * x_prev) / np.linalg.norm(x_prev)

            lambda_history.append(lambda_k)
            residual_history.append(residual)

            if residual < self.epsilon and k > 0:
                break

            x_prev = x_current.copy()

        return lambda_k, x_current, {
            'lambda_history': lambda_history,
            'residual_history': residual_history,
            'iterations': k + 1
        }

    def inverse_iteration(self, lambda_approx, x0=None, max_iter=10):
        """Метод обратных итераций по формулам (6)"""
        if x0 is None:
            x0 = np.ones(self.n)  # Начальный вектор (1, 1, ..., 1)^T
            x0 = x0 / np.linalg.norm(x0)

        x_prev = x0.copy()
        lambda_history = []
        residual_history = []

        # Матрица для решения систем
        B = self.A - lambda_approx * np.eye(self.n)

        for k in range(max_iter):
            try:
                # ФОРМУЛА (6): (A - λ*E)y^(k+1) = x^(k)
                y_current = np.linalg.solve(B, x_prev)

                # Нормировка: x^(k+1) = y^(k+1) / ‖y^(k+1)‖₂
                x_current = y_current / np.linalg.norm(y_current)

                # Отношение Рэлея для уточнения собственного значения
                lambda_k = np.dot(self.A @ x_current, x_current)

                # Невязка
                residual = np.linalg.norm(self.A @ x_current - lambda_k * x_current)

                lambda_history.append(lambda_k)
                residual_history.append(residual)

                if residual < self.epsilon and k > 0:
                    break

                x_prev = x_current.copy()

            except np.linalg.LinAlgError:
                print("Матрица вырождена, попытка использовать псевдообратную")
                y_current = np.linalg.lstsq(B, x_prev, rcond=None)[0]
                x_current = y_current / np.linalg.norm(y_current)
                lambda_k = np.dot(self.A @ x_current, x_current)
                break

        return lambda_k, x_current, {
            'lambda_history': lambda_history,
            'residual_history': residual_history,
            'iterations': k + 1
        }

    def rayleigh_quotient_iteration(self, x0=None, max_iter=10):
        """Метод обратных итераций с отношением Рэлея по формулам (13)-(15)"""
        if x0 is None:
            x0 = np.random.rand(self.n)
            x0 = x0 / np.linalg.norm(x0)

        x_prev = x0.copy()
        lambda_history = []
        residual_history = []

        # Начальное приближение для собственного значения
        lambda_k = np.dot(self.A @ x_prev, x_prev)
        lambda_history.append(lambda_k)

        for k in range(max_iter):
            try:
                # ФОРМУЛА (13): λ^(k+1) = ρ(x^(k))
                lambda_k = np.dot(self.A @ x_prev, x_prev)

                # ФОРМУЛА: (A - λ^(k+1)E)y^(k+1) = x^(k)
                B = self.A - lambda_k * np.eye(self.n)
                y_current = np.linalg.solve(B, x_prev)

                # ФОРМУЛА (15): x^(k+1) = y^(k+1) / ‖y^(k+1)‖₂
                x_current = y_current / np.linalg.norm(y_current)

                # Невязка
                residual = np.linalg.norm(self.A @ x_current - lambda_k * x_current)

                lambda_history.append(lambda_k)
                residual_history.append(residual)

                if residual < self.epsilon and k > 0:
                    break

                x_prev = x_current.copy()

            except np.linalg.LinAlgError:
                print("Матрица вырождена, попытка использовать псевдообратную")
                y_current = np.linalg.lstsq(B, x_prev, rcond=None)[0]
                x_current = y_current / np.linalg.norm(y_current)
                break

        return lambda_k, x_current, {
            'lambda_history': lambda_history,
            'residual_history': residual_history,
            'iterations': k + 1
        }

    def rayleigh_quotient(self, x):
        # Формула (6)
        return np.dot(self.A @ x, x) / np.dot(x, x)

    def a_posteriori_error(self, lambda_approx, x_approx):
        residual_vector = self.A @ x_approx - lambda_approx * x_approx
        return np.linalg.norm(residual_vector) / np.linalg.norm(x_approx)


def generate_random_matrix(size, matrix_type='symmetric'):
    if matrix_type == 'symmetric':
        A = np.random.randn(size, size)
        return (A + A.T) / 2
    elif matrix_type == 'diagonal_dominant':
        A = np.random.randn(size, size)
        for i in range(size):
            A[i, i] = np.sum(np.abs(A[i, :])) + 1
        return A
    else:
        return np.random.randn(size, size)


def input_matrix_manually():
    print("\nВвод матрицы:")
    size = int(input("Введите размер матрицы: "))

    print("Введите элементы матрицы построчно (через пробел):")
    A = []
    for i in range(size):
        while True:
            row = input(f"Строка {i + 1}: ").split()
            if len(row) == size:
                try:
                    A.append([float(x) for x in row])
                    break
                except ValueError:
                    print("Ошибка: введите числа!")
            else:
                print(f"Ошибка: нужно ввести {size} чисел!")

    return np.array(A)


def select_test_matrix():
    test_matrices = {
        '1': np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]),
        '2': np.array([[2.5, 1.0, 0.3], [0.8, 1.8, 0.1], [0.2, 0.1, 0.9]]),
        '3': np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.05], [0.05, 0.1, 0.5]]),
        '4': np.array([[1.0, 100.0], [0.01, 1.0]])
    }

    print("\nТестовые матрицы:")
    print("1. Симметричная с доминирующим собственным значением")
    print("2. Несимметричная с |λ1| > 1")
    print("3. С |λ1| < 1")
    print("4. Плохо обусловленная")

    choice = input("Выберите тестовую матрицу (1-4): ")
    return test_matrices.get(choice, test_matrices['1'])


def select_matrix():
    print("\n" + "=" * 50)
    print("ВЫБОР МАТРИЦЫ")
    print("=" * 50)
    print("1. Случайная симметричная матрица")
    print("2. Случайная матрица с диагональным преобладанием")
    print("3. Случайная общая матрица")
    print("4. Ввод матрицы вручную")
    print("5. Использовать тестовые матрицы")

    choice = input("\nВыберите тип матрицы (1-5): ")

    if choice == '1':
        size = int(input("Введите размер матрицы: "))
        return generate_random_matrix(size, 'symmetric'), "Случайная симметричная"

    elif choice == '2':
        size = int(input("Введите размер матрицы: "))
        return generate_random_matrix(size, 'diagonal_dominant'), "Случайная с диаг. преобладанием"

    elif choice == '3':
        size = int(input("Введите размер матрицы: "))
        return generate_random_matrix(size, 'general'), "Случайная общая"

    elif choice == '4':
        A = input_matrix_manually()
        return A, "Пользовательская матрица"

    elif choice == '5':
        return select_test_matrix(), "Тестовая матрица"

    else:
        print("Неверный выбор, используется случайная симметричная матрица")
        size = int(input("Введите размер матрицы: "))
        return generate_random_matrix(size, 'symmetric'), "Случайная симметричная"


def get_parameters():
    print("\n" + "=" * 50)
    print("ПАРАМЕТРЫ МЕТОДА")
    print("=" * 50)

    try:
        epsilon = float(input("Точность ε (по умолчанию 1e-8): ") or "1e-8")
        max_iter = int(input("Максимальное число итераций (по умолчанию 1000): ") or "1000")
    except ValueError:
        print("Неверный ввод, используются значения по умолчанию")
        epsilon = 1e-8
        max_iter = 1000

    return epsilon, max_iter


def select_task():
    print("\n" + "=" * 60)
    print("ВЫБОР ЗАДАНИЯ")
    print("=" * 60)
    print("1. Практическое занятие 9.1 - Степенной метод")
    print("2. Практическое занятие 9.2 - Метод обратных итераций")
    print("3. Выполнить оба задания последовательно")

    choice = input("\nВыберите задание (1-3): ")
    return choice


def task_9_1():
    """Выполнение задания 9.1 - Степенной метод"""
    np.random.seed(42)

    print("=" * 70)
    print("ВЫЧИСЛИТЕЛЬНЫЙ ПРАКТИКУМ - СТЕПЕННОЙ МЕТОД")
    print("=" * 70)

    A, matrix_name = select_matrix()
    epsilon, max_iter = get_parameters()

    print(f"\n{'-' * 60}")
    print(f"Матрица: {matrix_name}")
    print(f"Размер: {A.shape}")
    print(f"Матрица A:\n{A}")

    pm = PowerMethod(A, epsilon=epsilon, max_iter=max_iter)
    x0 = np.random.rand(A.shape[0])

    lambda_basic, eigenvector_basic, history_basic = pm.basic_power_method(x0.copy())
    lambda_modified, eigenvector_modified, history_modified = pm.modified_power_method(x0.copy())

    exact_eigenvalues, exact_eigenvectors = np.linalg.eig(A)
    idx_max = np.argmax(np.abs(exact_eigenvalues))
    lambda_exact = exact_eigenvalues[idx_max]
    eigenvector_exact = exact_eigenvectors[:, idx_max]

    print(f"\nТочное максимальное собственное значение: {lambda_exact:.10f}")

    print(f"\nБазовый метод:")
    print(f"  λ₁ = {lambda_basic:.10f}")
    print(f"  Погрешность: {abs(lambda_basic - lambda_exact):.2e}")
    print(f"  Итераций: {history_basic['iterations']}")
    print(f"  Апостериорная оценка: {pm.a_posteriori_error(lambda_basic, eigenvector_basic):.2e}")

    print(f"\nМодифицированный метод:")
    print(f"  λ₁ = {lambda_modified:.10f}")
    print(f"  Погрешность: {abs(lambda_modified - lambda_exact):.2e}")
    print(f"  Итераций: {history_modified['iterations']}")
    print(f"  Апостериорная оценка: {pm.a_posteriori_error(lambda_modified, eigenvector_modified):.2e}")

    cos_angle_basic = abs(np.dot(eigenvector_basic, eigenvector_exact))
    cos_angle_modified = abs(np.dot(eigenvector_modified, eigenvector_exact))

    print(f"\nКачество собственного вектора:")
    print(f"  Базовый метод: cos(φ) = {cos_angle_basic:.6f}")
    print(f"  Модифицированный метод: cos(φ) = {cos_angle_modified:.6f}")

    print(f"\n{'=' * 50}")
    print("СРАВНЕНИЕ МЕТОДОВ")
    print(f"{'=' * 50}")
    print(f"Разница в λ: {abs(lambda_basic - lambda_modified):.2e}")
    print(f"Разница в итерациях: {history_basic['iterations'] - history_modified['iterations']}")
    print(f"Норма базового вектора: {np.linalg.norm(eigenvector_basic):.6f}")
    print(f"Норма модифицированного вектора: {np.linalg.norm(eigenvector_modified):.6f}")

    return A, lambda_basic, eigenvector_basic


def task_9_2(A=None, lambda_approx=None):
    """Выполнение задания 9.2 - Метод обратных итераций"""
    if A is None:
        A, matrix_name = select_matrix()
    else:
        matrix_name = "Предыдущая матрица"

    epsilon, max_iter = get_parameters()

    print(f"\n{'-' * 60}")
    print(f"Матрица: {matrix_name}")
    print(f"Размер: {A.shape}")
    print(f"Матрица A:\n{A}")

    pm = PowerMethod(A, epsilon=epsilon, max_iter=max_iter)

    if lambda_approx is None:
        print("Получение начального приближения степенным методом...")
        lambda_approx, eigenvector_approx, _ = pm.modified_power_method()
        print(f"Начальное приближение λ* = {lambda_approx:.10f}")

    print(f"\nИспользуемое приближение: λ* = {lambda_approx:.10f}")

    print("\n1. МЕТОД ОБРАТНЫХ ИТЕРАЦИЙ:")
    lambda_inv, eigenvector_inv, inv_hist = pm.inverse_iteration(lambda_approx)

    print("\n2. МЕТОД ОБРАТНЫХ ИТЕРАЦИЙ С ОТНОШЕНИЕМ РЭЛЕЯ:")
    lambda_rayleigh, eigenvector_rayleigh, ray_hist = pm.rayleigh_quotient_iteration()

    exact_eigenvalues, exact_eigenvectors = np.linalg.eig(A)
    idx_max = np.argmax(np.abs(exact_eigenvalues))
    lambda_exact = exact_eigenvalues[idx_max]
    eigenvector_exact = exact_eigenvectors[:, idx_max]

    print(f"\nТочное максимальное собственное значение: {lambda_exact:.10f}")

    print(f"\nМетод обратных итераций:")
    print(f"  λ₁ = {lambda_inv:.10f}")
    print(f"  Погрешность: {abs(lambda_inv - lambda_exact):.2e}")
    print(f"  Итераций: {inv_hist['iterations']}")
    print(f"  Апостериорная оценка: {pm.a_posteriori_error(lambda_inv, eigenvector_inv):.2e}")

    print(f"\nМетод с отношением Рэлея:")
    print(f"  λ₁ = {lambda_rayleigh:.10f}")
    print(f"  Погрешность: {abs(lambda_rayleigh - lambda_exact):.2e}")
    print(f"  Итераций: {ray_hist['iterations']}")
    print(f"  Апостериорная оценка: {pm.a_posteriori_error(lambda_rayleigh, eigenvector_rayleigh):.2e}")

    cos_angle_inv = abs(np.dot(eigenvector_inv, eigenvector_exact))
    cos_angle_rayleigh = abs(np.dot(eigenvector_rayleigh, eigenvector_exact))

    print(f"\nКачество собственного вектора:")
    print(f"  Обратные итерации: cos(φ) = {cos_angle_inv:.6f}")
    print(f"  С отношением Рэлея: cos(φ) = {cos_angle_rayleigh:.6f}")

    print(f"\n{'=' * 50}")
    print("СРАВНЕНИЕ МЕТОДОВ")
    print(f"{'=' * 50}")
    print(f"Разница в λ: {abs(lambda_inv - lambda_rayleigh):.2e}")
    print(f"Разница в итерациях: {inv_hist['iterations'] - ray_hist['iterations']}")
    print(f"Норма вектора (обратные итерации): {np.linalg.norm(eigenvector_inv):.6f}")
    print(f"Норма вектора (Рэлей): {np.linalg.norm(eigenvector_rayleigh):.6f}")

np.random.seed(42)

print("=" * 70)
print("ВЫЧИСЛИТЕЛЬНЫЙ ПРАКТИКУМ")
print("ЧАСТИЧНАЯ ПРОБЛЕМА СОБСТВЕННЫХ ЗНАЧЕНИЙ")
print("=" * 70)

task_choice = select_task()

if task_choice == '1':
    task_9_1()
elif task_choice == '2':
    task_9_2()
elif task_choice == '3':
    A, lambda_approx, eigenvector_approx = task_9_1()
    print("\n" + "=" * 70)
    print("ПЕРЕХОД К ЗАДАНИЮ 9.2")
    print("Используем результат степенного метода как начальное приближение")
    print("=" * 70)
    task_9_2(A, lambda_approx)
else:
    print("Неверный выбор, выполняется задание 9.1")
    task_9_1()
