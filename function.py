import numpy as np
import scipy.optimize as opt
import support


class NormMethods:

    def __init__(self, n, m, niter, A, xstar, x0, psi):
        self.niter = niter
        self.n = n
        self.m = m
        self.A = A
        self.xstar = xstar
        self.x0 = x0
        self.psi = psi
        self.L = np.linalg.norm(self.A)

    def run_iter(self, err_bound, order):
        trial = 0
        err_arr = []
        x_old = self.x0
        x = self.x0
        while trial < self.niter:
            trial += 1
            sol = opt.minimize(self.func, x, args=(x_old, order), method='nelder-mead',
                               options={'xtol': err_bound, 'disp': True})
            x_old = x
            x = sol.x
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr

    def func(self, x, z, order=2):
        g = np.zeros(self.m)
        for i in range(self.m):
            g[i] = np.abs(np.dot(self.A[i], z) ** 2 + 2 * np.dot(self.A[i], z) * np.dot(self.A[i], x - z) - self.psi[i] ** 2)
        f = np.linalg.norm(g, ord=order)
        f += (self.L / 2.) * (np.linalg.norm(x - z) ** 2)
        return f


class AmplitudeFlow:

    def __init__(self, n, m, niter, A, xstar, x0, psi):
        self.niter = niter
        self.n = n
        self.m = m
        self.gamma = 0.7
        self.alpha = 0.6
        self.A = A
        self.xstar = xstar
        self.x0 = x0
        self.active = 1
        self.psi = psi
        self.L = np.linalg.norm(self.A)

    def get_active_set(self, x):
        return [i for i in range(self.m) if np.abs(np.dot(self.A[i], x)) > self.psi[i] / (1. + self.gamma)]

    def update_x(self, active_set, x):
        s = 0.0
        for i in active_set:
            s += (np.dot(self.A[i], x) - self.psi[i] * (np.dot(self.A[i], x) / np.abs(np.dot(self.A[i], x)))) * self.A[i]
        return x - (1. * self.alpha / self.m) * s

    def run_iter(self, active = 1):
        self.active = active
        trial = 0
        err_arr = []
        x = self.x0
        while trial < self.niter:
            trial += 1
            active_set = range[self.m]
            if self.active:
                active_set = self.get_active_set(x)
            x = self.update_x(active_set, x)
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr