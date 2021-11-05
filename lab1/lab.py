import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

COUNT = 1000


class DrvGenerator:
    def __init__(self, matrix_p, vector_a, vector_b):
        self.vector_a = vector_a
        self.vector_b = vector_b

        self.sum_by_row = np.sum(matrix_p, axis=1)
        self.cum_sum = np.cumsum(self.sum_by_row)
        self.cum_sum_by_row = np.cumsum(matrix_p, axis=1) / self.sum_by_row.reshape(-1, 1)

    def __next__(self):
        value = np.random.uniform(size=2)
        row = np.searchsorted(self.cum_sum, value[0])
        column = np.searchsorted(self.cum_sum_by_row[row], value[1])
        return row, column


def input_data():
    print("Input n", end=" ")
    n = int(input())

    print("Input m", end=" ")
    m = int(input())

    matrix_p = np.random.rand(n, m)
    matrix_p = np.vectorize(lambda x: x / matrix_p.sum())(matrix_p)

    print("matrix = ", matrix_p)

    vector_a = np.array([i for i in range(0, n)])
    vector_b = np.array([i for i in range(0, m)])

    print("vector_a = ", vector_a)
    print("vector_b = ", vector_b)

    return matrix_p, vector_a, vector_b


def create_matrix_emp(matrix_p, vector_a, vector_b, need_print=False):
    matrix_drv = DrvGenerator(matrix_p, vector_a, vector_b)
    matrix_emp = np.zeros(matrix_p.shape)

    for i in range(COUNT):
        matrix_emp[next(matrix_drv)] += 1
    matrix_emp /= np.sum(matrix_emp)

    if need_print:
        print("matrix_emp = \n", matrix_emp)

    return matrix_emp


def create_histograms(matrix_p, matrix_emp):
    values_a_p = np.sum(matrix_p, axis=1)
    values_b_p = np.sum(matrix_p, axis=0)

    values_a_emp = np.sum(matrix_emp, axis=1)
    values_b_emp = np.sum(matrix_emp, axis=0)

    data = {'теоретическая': values_a_p.tolist(), 'эмпирическая': values_a_emp.tolist()}
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.title("Гистограммы А")
    plt.show()

    data = {'теоретическая': values_b_p.tolist(), 'эмпирическая': values_b_emp.tolist()}
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.title("Гистограммы B")
    plt.show()


def calculate_expected_values(matrix_p, vector_a, vector_b):
    matrix_emp = create_matrix_emp(matrix_p, vector_a, vector_b)

    theoretical = vector_a @ np.sum(matrix_p, axis=1), vector_b @ np.sum(matrix_p, axis=0)
    empirical = vector_a @ np.sum(matrix_emp, axis=1), vector_b @ np.sum(matrix_emp, axis=0)

    print("\nМатематическое ожидание")
    print(f"Теоретическое:\nM[A] = {theoretical[0]}, M[B] = {theoretical[1]}")
    print(f"Эмпирическое:\nM[A] = {empirical[0]}, M[B] = {empirical[1]}")


def calculate_intervals(values):
    normal_quantile = stats.norm.ppf(1.95 / 2)

    values_mean = np.mean(values)
    values_var = np.var(values, ddof=1)

    return values_mean - np.sqrt(values_var / COUNT) * normal_quantile, values_mean + np.sqrt(
        values_var / COUNT) * normal_quantile


def calculate_intervals_for_expected_values(matrix_p, vector_a, vector_b):
    matrix_drv = DrvGenerator(matrix_p, vector_a, vector_b)
    values = np.array([next(matrix_drv) for _ in range(COUNT)])

    values_a = [vector_a[value[0]] for value in values]
    values_b = [vector_b[value[1]] for value in values]

    interval_x = calculate_intervals(values_a)
    print("\nДоверительный интервал M[A]: ", interval_x)
    interval_y = calculate_intervals(values_b)
    print("Доверительный интервал M[B]: ", interval_y)


def dispersion(matrix_p, vector_a, vector_b):
    values_a = vector_a @ np.sum(matrix_p, axis=1)
    values_b = vector_b @ np.sum(matrix_p, axis=0)

    return np.square(vector_a) @ np.sum(matrix_p, axis=1) - values_a ** 2, np.square(vector_b) @ np.sum(matrix_p, axis=0) - values_b ** 2


def calculate_dispersion(matrix_p, vector_a, vector_b):
    matrix_emp = create_matrix_emp(matrix_p, vector_a, vector_b)

    theoretical = dispersion(matrix_p, vector_a, vector_b)
    empirical = dispersion(matrix_emp, vector_a, vector_b)

    print("\nДисперсия")
    print(f"Теоретическая:\nD[A] = {theoretical[0]}, D[B] = {theoretical[1]}")
    print(f"Эмпирическая:\nD[A] = {empirical[0]}, D[B] = {empirical[1]}")


def calculate_interval_dispersion(values, n):
    rv_var = np.var(values, ddof=1)
    chi = stats.chi2(n - 1)
    array = chi.rvs(100000)
    q = stats.mstats.mquantiles(array, prob=[0.05 / 2, 1.95 / 2])

    xi_plus = q[0]
    xi_minus = q[1]

    return (n - 1) * rv_var / xi_minus, (n - 1) * rv_var / xi_plus


def calculate_intervals_for_dispersion(matrix_p, vector_a, vector_b):
    matrix_drv = DrvGenerator(matrix_p, vector_a, vector_b)
    values = np.array([next(matrix_drv) for _ in range(COUNT)])

    values_a = [vector_a[value[0]] for value in values]
    values_b = [vector_b[value[1]] for value in values]

    interval_a = calculate_interval_dispersion(values_a, COUNT)
    print("Доверительный интервал D[A]: ", interval_a)

    interval_b = calculate_interval_dispersion(values_b, COUNT)
    print("Доверительный интервал D[B]: ", interval_b)


def covariance(matrix_p, vector_a, vector_b):
    values = vector_a @ np.sum(matrix_p, axis=1), vector_b @ np.sum(matrix_p, axis=0)

    return vector_a @ matrix_p @ vector_b - np.prod(values)


def correlation(matrix_p, vector_a, vector_b):
    matrix_drv = DrvGenerator(matrix_p, vector_a, vector_b)
    values = np.array([next(matrix_drv) for _ in range(COUNT)])

    values_a = np.var([vector_a[x[0]] for x in values], ddof=1)
    values_b = np.var([vector_b[y[1]] for y in values], ddof=1)

    return covariance(matrix_p, vector_a, vector_b) / np.sqrt(np.prod((values_a, values_b)))


def calculate_correlation(matrix_p, vector_a, vector_b):
    matrix_emp = create_matrix_emp(matrix_p, vector_a, vector_b)

    print("Теоретическая корреляция = ", correlation(matrix_p, vector_a, vector_b))
    print("Эмпирическая корреляция = ", correlation(matrix_emp, vector_a, vector_b))


def check_chi_square(matrix_p, matrix_emp):
    chi_square = COUNT * np.sum(np.square(matrix_emp - matrix_p) / matrix_p)
    chi_square_from_stats = stats.chi2.ppf(0.95, matrix_p.size - 1)

    print(f"Критерий Пирсона = {chi_square}, ", end="")
    if chi_square_from_stats > chi_square:
        print("эмпирическое распределение сооответствует теоретичскому")
    else:
        print("эмпирическое распределение не сооответствует теоретическому")


def main():
    matrix_p, vector_a, vector_b = input_data()
    matrix_emp = create_matrix_emp(matrix_p, vector_a, vector_b, True)

    create_histograms(matrix_p, matrix_emp)
    calculate_expected_values(matrix_p, vector_a, vector_b)
    calculate_intervals_for_expected_values(matrix_p, vector_a, vector_b)
    calculate_dispersion(matrix_p, vector_a, vector_b)
    calculate_intervals_for_dispersion(matrix_p, vector_a, vector_b)
    calculate_correlation(matrix_p, vector_a, vector_b)
    check_chi_square(matrix_p, matrix_emp)


if __name__ == '__main__':
    main()
