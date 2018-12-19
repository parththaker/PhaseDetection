import numpy as np
import math


def generate_random_quant(n,m):

    A = np.random.randn(m, n)
    x_star = np.random.randn(n)
    x = np.random.randn(n)

    psi = np.zeros(m)
    for i in range(m):
        psi[i] = np.abs(np.dot(A[i], x_star))

    return A, x_star, x, psi


def check_error(A, x, x_star, m, option = 1):

    if option == 1:
        return sum([np.abs(np.abs(np.dot(A[i], x)) - np.abs(np.dot(A[i], x_star))) for i in range(m)])
    elif option == 2:
        s = 0
        for i in range(len(x)):
            s += min((x[i] - x_star[i])**2, (x[i] + x_star[i])**2)
        return math.sqrt(s)/np.linalg.norm(x_star)