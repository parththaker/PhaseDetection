import numpy as np
import math
import scipy

def generate_random_quant(n,m):

    A = np.random.randn(m, n)
    x_star = np.random.randn(n)
    x = np.random.randn(n)

    psi = np.zeros(m)
    for i in range(m):
        psi[i] = np.abs(np.dot(A[i], x_star))

    return A, x_star, x, psi


def check_error(A, x, x_star, m, option = 1, order = 1):

    if option == 1:
        f = np.array([np.abs(np.abs(np.dot(A[i], x))**2 - np.abs(np.dot(A[i], x_star))**2) for i in range(m)])
        return np.linalg.norm(f, ord=order)
    elif option == 2:
        s = 0
        for i in range(len(x)):
            s += min((x[i] - x_star[i])**2, (x[i] + x_star[i])**2)
        return math.sqrt(s)/np.linalg.norm(x_star)


def TWFinitialization(m, n, A, psi):
    ini_matrix = 0.0
    for i in range(m):
        ini_matrix += psi[i]**2*np.outer(A[i], A[i])
    ini_matrix = (1.0/m)*ini_matrix

    eig_values , eig_vec = np.linalg.eig(ini_matrix)
    index = np.argmax(eig_values)
    if isinstance(index, np.int64):
        led_eig_vec = eig_vec[:, index]
    else:
        led_eig_vec = eig_vec[:, index[0]]

    lambda_0 = math.sqrt(1.0/m*sum([psi[i]**2 for i in range(m)]))

    return math.sqrt(m*n/sum([np.linalg.norm(A[i])**2 for i in range(m)]))*lambda_0*led_eig_vec

def corrupt_readings(psi, type='gauss'):
    if type == 'gauss':
        for i in range(len(psi)):
            var = np.random.normal(0, 0.001)
            print(psi[i]**2, var)
            while psi[i]**2+var <0:
                var = np.random.normal(0, 0.001)
            psi[i] = math.sqrt(psi[i]**2 + var)
    elif type == 'out':
        for i in range(len(psi)):
            eta = np.random.random()
            if eta <0.2:
                err = np.random.random()*0.4
                psi[i] = math.sqrt(psi[i] ** 2 + err)
    return psi

def running_avg(arr):
    new_arr = []
    s = 0
    for i in arr:
        s = ((0.1)*i + 0.9*s)
        new_arr.append(s)
    return new_arr