import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Захаров Тимофей, 23Б06-мм. Решение задачи 11.1 вариант 7
a, b = -1.0, 1.0
n_values = [10, 20, 100]

# Коэффициенты уравнения
def p(x):
    return 1/(2 + x)


def q(x):
    return 1 / (x * x + 4 * x + 4)


def r(x):
    return np.cos(x)


def f(x):
    return 1 + x


def ode(x, y):
    return np.vstack((y[1], (q(x) * y[1] + r(x) * y[0] - f(x)) / p(x)))


def bc(ya, yb):
    return np.array([ya[0], yb[0]])


x_exact = np.linspace(a, b, 1001)
y_init = np.zeros((2, x_exact.size))
sol = solve_bvp(ode, bc, x_exact, y_init, max_nodes=10000)


def y_exact_func(x):
    return sol.sol(x)[0]


print("=" * 85)
print("Вариант c листочка: -(1/(2+x) u'(x))' + cos(x) u(x) = 1 + x")
print("          u(-1)=0, u(1)=0")
print("=" * 85)

all_results = []

for n in n_values:
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y_exact = y_exact_func(x)

    # Метод O(h)
    main_diag = np.zeros(n + 1)
    lower_diag = np.zeros(n)
    upper_diag = np.zeros(n)
    rhs = np.zeros(n + 1)

    main_diag[0] = 1
    upper_diag[0] = 0

    for i in range(1, n):
        xi = x[i]
        A_i = -p(xi) - (h / 2) * q(xi)
        B_i = 2 * p(xi) + h ** 2 * r(xi)
        C_i = -p(xi) + (h / 2) * q(xi)

        lower_diag[i - 1] = A_i
        main_diag[i] = B_i
        upper_diag[i] = C_i
        rhs[i] = h ** 2 * f(xi)

    lower_diag[n - 1] = 0
    main_diag[n] = 1

    A_mat = diags([main_diag, lower_diag, upper_diag], [0, -1, 1], format='csc')
    y_O1 = spsolve(A_mat, rhs)

    # Метод O(h²)
    x_shifted = a - h / 2 + np.arange(0, n + 2) * h

    main_diag2 = np.zeros(n + 2)
    lower_diag2 = np.zeros(n + 1)
    upper_diag2 = np.zeros(n + 1)
    rhs2 = np.zeros(n + 2)

    main_diag2[0] = 1
    upper_diag2[0] = 0

    for i in range(1, n + 1):
        xi = x_shifted[i]
        A_i = -p(xi) - (h / 2) * q(xi)
        B_i = 2 * p(xi) + h ** 2 * r(xi)
        C_i = -p(xi) + (h / 2) * q(xi)

        lower_diag2[i - 1] = A_i
        main_diag2[i] = B_i
        upper_diag2[i] = C_i
        rhs2[i] = h ** 2 * f(xi)

    lower_diag2[n] = 0
    main_diag2[n + 1] = 1

    A_mat2 = diags([main_diag2, lower_diag2, upper_diag2], [0, -1, 1], format='csc')
    y_shifted = spsolve(A_mat2, rhs2)

    y_O2 = np.zeros(n + 1)
    for i in range(n + 1):
        y_O2[i] = (y_shifted[i] + y_shifted[i + 1]) / 2

    err_O1_L2 = np.sqrt(h * np.sum((y_O1 - y_exact) ** 2))
    err_O1_max = np.max(np.abs(y_O1 - y_exact))
    err_O2_L2 = np.sqrt(h * np.sum((y_O2 - y_exact) ** 2))
    err_O2_max = np.max(np.abs(y_O2 - y_exact))

    all_results.append({
        'n': n, 'h': h, 'x': x, 'y_exact': y_exact,
        'y_O1': y_O1, 'y_O2': y_O2,
        'err_O1_L2': err_O1_L2, 'err_O1_max': err_O1_max,
        'err_O2_L2': err_O2_L2, 'err_O2_max': err_O2_max
    })

# Таблицы
print("\nТаблица результатов")
print("-" * 85)
print(f"{'n':<6} {'h':<8} {'Метод':<10} {'L2-погрешность':<20} {'Макс.погрешность':<20}")
print("-" * 85)

for res in all_results:
    print(f"{res['n']:<6} {res['h']:<8.4f} {'O(h)':<10} {res['err_O1_L2']:<20.6e} {res['err_O1_max']:<20.6e}")
    print(f"{' ':<6} {' ':<8} {'O(h²)':<10} {res['err_O2_L2']:<20.6e} {res['err_O2_max']:<20.6e}")
    print("-" * 85)

print("\nТаблица значений в узлах (n=10)")
print("-" * 85)
print(f"{'x':<10} {'Точное':<15} {'O(h)':<15} {'O(h²)':<15} {'|O(h²)-Точн.|':<15}")
print("-" * 85)

res_10 = all_results[0]
for i in range(0, 11):
    x_val = res_10['x'][i]
    exact = res_10['y_exact'][i]
    O1 = res_10['y_O1'][i]
    O2 = res_10['y_O2'][i]
    diff = np.abs(O2 - exact)
    print(f"{x_val:<10.2f} {exact:<15.6e} {O1:<15.6e} {O2:<15.6e} {diff:<15.6e}")

print("=" * 85)

# Графики решений
plt.figure(figsize=(15, 10))


#
# plt.plot(x_fine, y_exact_fine, 'k-', linewidth=2, label='Точное решение')
# plt.plot(res_20['x'], res_20['y_O1'], 'bo--', linewidth=1, markersize=4, label=f'O(h), n={res_20["n"]}')
# plt.plot(res_20['x'], res_20['y_O2'], 'rs--', linewidth=1, markersize=4, label=f'O(h²), n={res_20["n"]}')
#
# plt.xlabel('x')
# plt.ylabel('u(x)')
# plt.title('Сравнение решений (n=20)')
# plt.legend()
# plt.grid(True, alpha=0.3)

# Погрешности для n=20
res_20 = all_results[1]  # n=20
x_fine = np.linspace(a, b, 200)
y_exact_fine = y_exact_func(x_fine)

plt.subplot(2, 3, 1)
x_20 = res_20['x']
err_O1 = np.abs(res_20['y_O1'] - res_20['y_exact'])
err_O2 = np.abs(res_20['y_O2'] - res_20['y_exact'])

plt.plot(x_20, err_O1, 'b-o', linewidth=1, markersize=4, label=f'O(h), макс={err_O1.max():.2e}')
plt.plot(x_20, err_O2, 'r-s', linewidth=1, markersize=4, label=f'O(h²), макс={err_O2.max():.2e}')

plt.xlabel('x')
plt.ylabel('Абсолютная погрешность')
plt.title('Погрешности методов (n=20)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# График 3: Сравнение методов при n=10
plt.subplot(2, 3, 4)
res_10 = all_results[0]  # n=10
plt.plot(x_fine, y_exact_fine, 'k-', linewidth=2, label='Точное решение')
plt.plot(res_10['x'], res_10['y_O1'], 'bo-', linewidth=1, markersize=4, label=f'O(h), n={res_10["n"]}')
plt.plot(res_10['x'], res_10['y_O2'], 'rs-', linewidth=1, markersize=4, label=f'O(h²), n={res_10["n"]}')

plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(f'Сравнение методов при n={res_10["n"]}')
plt.legend()
plt.grid(True, alpha=0.3)

# График 4: Сравнение методов при n=20
plt.subplot(2, 3, 5)
res_20 = all_results[1]  # n=20
plt.plot(x_fine, y_exact_fine, 'k-', linewidth=2, label='Точное решение')
plt.plot(res_20['x'], res_20['y_O1'], 'bo-', linewidth=1, markersize=4, label=f'O(h), n={res_20["n"]}')
plt.plot(res_20['x'], res_20['y_O2'], 'rs-', linewidth=1, markersize=4, label=f'O(h²), n={res_20["n"]}')

plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(f'Сравнение методов при n={res_20["n"]}')
plt.legend()
plt.grid(True, alpha=0.3)

# График 5: Сравнение методов при n=100
plt.subplot(2, 3, 6)
res_100 = all_results[2]  # n=100
plt.plot(x_fine, y_exact_fine, 'k-', linewidth=2, label='Точное решение')
plt.plot(res_100['x'], res_100['y_O1'], 'bo-', linewidth=0.5, markersize=2, alpha=0.7, label=f'O(h), n={res_100["n"]}')
plt.plot(res_100['x'], res_100['y_O2'], 'rs-', linewidth=0.5, markersize=2, alpha=0.7, label=f'O(h²), n={res_100["n"]}')

plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(f'Сравнение методов при n={res_100["n"]}')
plt.legend()
plt.grid(True, alpha=0.3)

# График 6: Сходимость методов
plt.subplot(2, 3, 2)
n_list = [res['n'] for res in all_results]
err_O1_L2_list = [res['err_O1_L2'] for res in all_results]
err_O2_L2_list = [res['err_O2_L2'] for res in all_results]
h_list = [res['h'] for res in all_results]

plt.loglog(h_list, err_O1_L2_list, 'b-o', linewidth=2, markersize=8, label='Метод O(h)')
plt.loglog(h_list, err_O2_L2_list, 'r-s', linewidth=2, markersize=8, label='Метод O(h²)')

# Линии сходимости
plt.loglog(h_list, np.array(h_list), 'k--', alpha=0.5, label='h (первый порядок)')
plt.loglog(h_list, np.array(h_list)**2, 'k:', alpha=0.5, label='h² (второй порядок)')

plt.xlabel('Шаг сетки h')
plt.ylabel('L2-погрешность')
plt.title('Зависимость погрешности от шага')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Вывод
print("\nВвыводы:")
print(f"1. Метод O(h²) дает погрешность в {all_results[0]['err_O1_L2'] / all_results[0]['err_O2_L2']:.1f} раз")
print(f"   меньше, чем метод O(h) для n=10")
print(f"2. Метод O(h²) дает погрешность в {all_results[1]['err_O1_L2'] / all_results[1]['err_O2_L2']:.1f} раз")
print(f"   меньше, чем метод O(h) для n=20")
print("3. С уменьшением шага h погрешности уменьшаются для обоих методов")