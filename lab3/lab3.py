from math import factorial
import numpy as np
import matplotlib.pyplot as plt


def generate_average_making_in_smo(p, n, m, ro):
    return ro * (1 - ro ** (n + m) / (n ** m) / factorial(n) * p[0])


def generate_average_time_in_queue(L_q, lamba):
    return L_q / lamba


def generate_average_time_in_smo(L_q, lamba, Q, mu):
    return L_q / lamba + Q / mu


def generate_average_in_queue(p, n, m):
    return n ** n / factorial(n) * m * (m + 1) / 2 * p[0]


def generate_p_otk(p):
    return p[-1]


def generate_absolute_capacity(p_otk, lamba):
    return (1 - p_otk) * lamba


def generate_p(n, m, ro):
    p = []

    p0 = sum([n ** i / factorial(i) for i in range(0, n + 1)])
    p0 += n ** n * m / factorial(n)

    p0 = 1 / p0
    p.append(p0)

    for k in range(1, n + 1):
        p.append(ro ** k / factorial(k) * p0)

    for i in range(1, m + 1):
        p_i = p[0] * (ro ** (n + i) / n ** i / factorial(n))
        p.append(p_i)

    return p


def generate_requests(time_max, lamba):
    requests = []
    t = 0
    while t < time_max:
        t += np.random.exponential(1 / lamba)
        requests.append(t)
    return requests


def generate_p_emp(n, m, mu, lamba, max_time):
    current_time = 0
    queue = []
    smo = []
    requests = generate_requests(max_time, lamba)
    p_emp = [0 for i in range(0, n + m + 1)]
    requests_count = len(requests)
    unprocessed_count = 0

    while current_time < max_time:
        request_min = min(requests)
        smo_min = -1 if len(smo) == 0 else min(smo)

        min_value = min([value for value in [request_min, smo_min] if value != -1])

        if min_value == request_min:
            p_emp[len(smo) + len(queue)] += min_value - current_time
            requests.remove(min_value)
            if len(smo) < n:
                smo.append(min_value + np.random.exponential(1 / mu))
            elif len(queue) < m:
                queue.append(min_value + 1000)
            else:
                unprocessed_count += 1
                pass

        if min_value == smo_min:
            p_emp[len(smo) + len(queue)] += min_value - current_time
            smo.remove(min_value)
            if len(queue) != 0:
                queue.pop(0)
                smo.append(min_value + np.random.exponential(1 / mu))

        current_time = min_value

    A_emp = (requests_count - unprocessed_count) / max_time
    p_emp_n = [p / max_time for p in p_emp]

    return p_emp_n, A_emp


def main():
    n = 2  # ?????????? ??????????????
    lamba = 1  # ?????????????????????????? ??????????????????????
    m = 4  # ?????????? ?? ??????????????
    t_serv = 2
    mu = 1 / t_serv  # ?????????????????????????? ???????????????????????? ????????????
    ro = lamba / mu  # ???????? ????????????????
    max_time = 10000

    # ?????????????????? ?????????????????????? ??????????????????
    p = generate_p(n, m, ro)
    p_emp, A_emp = generate_p_emp(n, m, mu, lamba, max_time)

    # ?????????????????????? ????????????
    p_otk = generate_p_otk(p)
    p_otk_emp = generate_p_otk(p_emp)

    # ???????????????????? ???????????????????? ??????????????????????
    A = generate_absolute_capacity(p_otk, lamba)

    # ?????????????????????????? ???????????????????? ??????????????????????
    Q = 1 - p_otk
    Q_emp = 1 - p_otk_emp

    # ?????????????? ?????????? ???????????? ?? ??????????????
    L_q = generate_average_in_queue(p, n, m)
    L_q_emp = generate_average_in_queue(p_emp, n, m)

    # ?????????????? ?????????? ???????????????????? ???????????? ?? ??????
    t_smo = generate_average_time_in_smo(L_q, lamba, Q, mu)
    t_smo_emp = generate_average_time_in_smo(L_q_emp, lamba, Q_emp, mu)

    print(
        f"?????????? ??????????????: {n}, ???????? ?? ??????????????: {m}, ?????????????????????????? ???????????? ????????????: {lamba}, ?????????????????????????? ???????????? ????????????????????????: {mu}")

    print("?????????????????????????? ????????????????????????????:")
    print(sum(p))
    print("?????????????????? ?????????????????????? ??????????????????: ", p)
    print("???????????????????? ???????????????????? ??????????????????????: ", A)
    print("?????????????????????????? ???????????????????? ??????????????????????: ", Q)
    print("?????????????????????? ????????????: ", p_otk)
    print("?????????????? ?????????? ???????????? ?? ??????????????: ", L_q)
    print("?????????????? ?????????? ???????????????????? ???????????? ?? ??????: ", t_smo)

    print("\n???????????????????????? ????????????????????????????:")
    print(sum(p_emp))
    print("?????????????????? ?????????????????????? ??????????????????: ", p_emp)
    print("???????????????????? ???????????????????? ??????????????????????: ", A_emp)
    print("?????????????????????????? ???????????????????? ??????????????????????: ", Q_emp)
    print("?????????????????????? ????????????: ", p_otk_emp)
    print("?????????????? ?????????? ???????????? ?? ??????????????: ", L_q_emp)
    print("?????????????? ?????????? ???????????????????? ???????????? ?? ??????: ", t_smo_emp)

    plt.plot(list(range(n + m + 1)), p)
    plt.plot(list(range(n + m + 1)), p_emp)
    plt.legend(['theor', 'emp'])
    plt.show()


if __name__ == '__main__':
    main()
