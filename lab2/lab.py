from math import factorial
import matplotlib.pyplot as plt
import numpy as np


def generate_p(n, m, ro, betta):
    p = []

    p0 = sum([(ro ** i) / factorial(i) for i in range(0, n + 1)])
    summa = 0
    for i in range(1, m + 1):
        summa += ro ** i / np.prod([n + l * betta for l in range(1, i + 1)])
    p0 += summa * ((ro ** n) / factorial(n))
    p0 = 1 / p0
    p.append(p0)

    for k in range(1, n + 1):
        p.append(ro ** k / factorial(k) * p0)

    for i in range(1, m + 1):
        p_i = p[n] * (ro ** i / np.prod([n + l * betta for l in range(1, i + 1)]))
        p.append(p_i)

    return p


def generate_p_otk(p, n, m, ro, betta):
    p_otk = p[n] * ro ** m / np.prod([n + l * betta for l in range(1, m + 1)])
    return p_otk


def generate_absolute_capacity(p_otk, lamba):
    return (1 - p_otk) * lamba


def generate_average_in_queue(p, n, m):
    return sum([i * p[n + i] for i in range(1, m)])


def generate_average_making_in_smo(p, n, m):
    return sum([k * p[k] for k in range(1, n + 1)]) + sum([n * p[n + i] for i in range(1, m + 1)])


def generate_average_time_in_queue(L_q, lamba):
    return L_q / lamba


def generate_average_time_in_smo(L_ch, mu, t_q):
    return L_ch / mu + t_q


def generate_average_in_smo(L_q, L_ch):
    return L_q + L_ch


def generate_requests(time_max, lamba):
    requests = []
    t = 0
    while t < time_max:
        t += np.random.exponential(1 / lamba)
        requests.append(t)
    return requests


def generate_p_emp(n, m, mu, v, lamba, max_time):
    system_states = [0 for i in range(0, max_time)]
    system_states_time = [0 for i in range(0, max_time)]
    current_time = 0
    queue = []
    smo = []
    requests = generate_requests(max_time, lamba)
    p_emp = [0 for i in range(0, n + m + 1)]
    requests_count = len(requests)
    unprocessed_count = 0

    while current_time < max_time:
        request_min = min(requests)
        queue_min = -1 if len(queue) == 0 else min(queue)
        smo_min = -1 if len(smo) == 0 else min(smo)

        min_value = min([value for value in [request_min, queue_min, smo_min] if value != -1])

        if min_value == request_min:
            p_emp[len(smo) + len(queue)] += min_value - current_time
            requests.remove(min_value)
            if len(smo) < n:
                smo.append(min_value + np.random.exponential(1 / mu))
            elif len(queue) < m:
                queue.append(min_value + np.random.exponential(1 / v))
            else:
                unprocessed_count += 1
                pass

        if min_value == smo_min:
            p_emp[len(smo) + len(queue)] += min_value - current_time
            smo.remove(min_value)
            if len(queue) != 0:
                queue.pop(0)
                smo.append(min_value + np.random.exponential(1 / mu))

        if min_value == queue_min:
            p_emp[len(smo) + len(queue)] += min_value - current_time
            queue.remove(min_value)

        current_time = min_value
        system_states_time.append(current_time)
        system_states.append(len(smo) + len(queue))

    A_emp = (requests_count - unprocessed_count) / max_time
    p_emp_n = [p / max_time for p in p_emp]

    plt.plot(list(system_states_time), system_states)
    plt.xlabel("time")
    plt.ylabel("state")
    plt.show()

    return p_emp_n, A_emp


def main(n, m, lamba, mu):
    # n = 5  # число каналов
    # lamba = 25  # интенсивность поступления
    # m = 10  # места в очереди
    # mu = 2  # интенсивность обслуживания заявок
    ro = lamba / mu  # коэф загрузки
    v = 6  # интенсивность ухода потока заявок
    betta = v / mu
    max_time = 100

    # финальные вероятности состояний
    p = generate_p(n, m, ro, betta)
    p_emp, A_emp = generate_p_emp(n, m, mu, v, lamba, max_time)

    # вероятность отказа
    p_otk = generate_p_otk(p, n, m, ro, betta)
    p_otk_emp = generate_p_otk(p_emp, n, m, ro, betta)

    # абсолютная пропускная способность
    A = generate_absolute_capacity(p_otk, lamba)

    # среднее число заявок в очереди
    L_q = generate_average_in_queue(p, n, m)
    L_q_emp = generate_average_in_queue(p_emp, n, m)

    # среднее число занятых каналов
    L_ch = generate_average_making_in_smo(p, n, m)
    L_ch_emp = generate_average_making_in_smo(p_emp, n, m)

    # средние число заявок в СМО
    L_smo = generate_average_in_smo(L_q, L_ch)
    L_smo_emp = generate_average_in_smo(L_q_emp, L_ch_emp)

    # среднее время пребывания заявки в очереди
    t_q = generate_average_time_in_queue(L_q, lamba)
    t_q_emp = generate_average_time_in_queue(L_q_emp, lamba)

    # среднее время пребывания заявки в СМО
    t_smo = generate_average_time_in_smo(L_ch, mu, t_q)
    t_smo_emp = generate_average_time_in_smo(L_ch_emp, mu, t_q_emp)

    print(f"Число каналов: {n}, мест в очереди: {m}, интенсивность потока заявок: {lamba}, интенсивность потока обслуживания: {mu}, параметр v: {v}, ограничение пребывания заявки в очереди: {max_time}")

    print("Теоретические характеристики:")
    print("Финальные вероятности состояний: ", p)
    print("Абсолютная пропускная способность: ", A)
    print("Вероятность отказа: ", p_otk)
    print("Средние число заявок в СМО: ", L_smo)
    print("Среднее число заявок в очереди: ", L_q)
    print("Среднее время пребывания заявки в СМО: ", t_smo)
    print("Среднее время пребывания заявки в очереди: ", t_q)
    print("Среднее число занятых каналов: ", L_ch)

    print("\nЭмпирические характеристики:")
    print("Финальные вероятности состояний: ", p_emp)
    print("Абсолютная пропускная способность: ", A_emp)
    print("Вероятность отказа: ", p_otk_emp)
    print("Средние число заявок в СМО: ", L_smo_emp)
    print("Среднее число заявок в очереди: ", L_q_emp)
    print("Среднее время пребывания заявки в СМО: ", t_smo_emp)
    print("Среднее время пребывания заявки в очереди: ", t_q_emp)
    print("Среднее число занятых каналов: ", L_ch_emp)

    plt.plot(list(range(n + m + 1)), p)
    plt.plot(list(range(n + m + 1)), p_emp)
    plt.legend(['theor', 'emp'])
    plt.title(f"n = {n}, m = {m}, lamba = {lamba}, mu = {mu}")
    plt.show()


if __name__ == '__main__':
    print("CHANGE N")
    main(2, 4, 8, 2)
    main(4, 4, 8, 2)
    main(8, 4, 8, 2)

    print("CHANGE M")
    main(2, 4, 8, 2)
    main(2, 8, 8, 2)
    main(2, 16, 8, 2)

    print("CHANGE LAMBA")
    main(2, 4, 4, 2)
    main(2, 4, 8, 2)
    main(2, 4, 16, 2)

    print("CHANGE MU")
    main(2, 4, 8, 2)
    main(2, 4, 8, 4)
    main(2, 4, 8, 8)
