import numpy as np
import scipy.optimize as opt
import support
import math
import random


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

    def run_iter(self, err_bound, order, inc=0):
        trial = 0
        err_arr = []
        x_old = self.x0
        x = self.x0
        while trial < self.niter:
            trial += 1
            index = random.randint(0, self.m - 1)
            sol = opt.minimize(self.func, x, args=(x_old, order, inc, index), method='nelder-mead',
                               options={'xtol': err_bound, 'disp': True})
            x_old = x
            x = sol.x
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr

    def func(self, x, z, order=2, inc=0, index =0):
        g = np.zeros(self.m)
        if not inc:
            for i in range(self.m):
                g[i] = np.abs(np.dot(self.A[i], z) ** 2 + 2 * np.dot(self.A[i], z) * np.dot(self.A[i], x - z) - self.psi[i] ** 2)
        else:
            i = index
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

    def run_iter(self, active=1, inc=0):
        self.active = active
        trial = 0
        err_arr = []
        x = self.x0
        while trial < self.niter:
            trial += 1
            active_set = range(self.m)
            if self.active:
                active_set = self.get_active_set(x)
            if inc:
                index = np.random.choice(active_set)
                active_set = [index]
            x = self.update_x(active_set, x)
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr


class WirtingerFlow:

    def __init__(self, n, m, niter, A, xstar, x0, psi):
        self.niter = niter
        self.n = n
        self.m = m
        self.alpha_h = 5.0
        self.alpha_y = 3.0
        self.alpha_lb = 0.3
        self.alpha_ub = 5.0
        self.mu = 0.3
        self.A = A
        self.xstar = xstar
        self.x0 = x0
        self.active = 1
        self.psi = psi
        self.L = np.linalg.norm(self.A)

    def get_active_set(self, x):
        angles = [math.sqrt(self.n)*np.abs(np.dot(self.A[i], x))/np.linalg.norm(self.A[i])/np.linalg.norm(x) for i in range(self.m)]
        violations = [np.abs(self.psi[i]**2 - np.dot(self.A[i], x)**2) for i in range(self.m)]

        h = []
        for i in range(self.m):
            if self.alpha_lb <= angles[i] <= self.alpha_ub and violations[i] <= self.alpha_h*(sum(violations)/self.m)*angles[i]:
                h.append(i)
        return h

    def update_x(self, active_set, x):
        s = 0.0
        for i in active_set:
            s += ( ( self.psi[i]**2 - np.dot(self.A[i], x)**2 ) / np.dot(self.A[i], x) ) * self.A[i]
        return x + (2*self.mu/self.m) * s

    def run_iter(self, active=1, inc=0):
        self.active = active
        trial = 0
        err_arr = []
        x = self.x0
        while trial < self.niter:
            trial += 1
            active_set = range(self.m)
            if self.active:
                active_set = self.get_active_set(x)
            if inc:
                index = np.random.choice(active_set)
                active_set = [index]
            x = self.update_x(active_set, x)
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr


class Kaczmarc:

    def __init__(self, n, m, niter, A, xstar, x0, psi):
        self.niter = niter
        self.n = n
        self.m = m
        self.A = A
        self.xstar = xstar
        self.x0 = x0
        self.psi = psi
        self.L = np.linalg.norm(self.A)

    def update_x(self, x):
        i = random.randint(0, self.m-1)
        s = np.dot( np.eye(self.n) - ( (1./ np.linalg.norm(self.A[i])**2)*(np.abs( np.dot(self.A[i], x) ) - self.psi[i]) / np.abs( np.dot(self.A[i], x) ) ) * np.outer(self.A[i], self.A[i]), x)
        return s

    def run_iter(self):
        trial = 0
        err_arr = []
        x = self.x0
        while trial < self.niter:
            trial += 1
            x = self.update_x(x)
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr


class SAGWirtingerFlow:

    def __init__(self, n, m, niter, A, xstar, x0, psi):
        self.niter = niter
        self.n = n
        self.m = m
        self.alpha_h = 5.0
        self.alpha_y = 3.0
        self.alpha_lb = 0.3
        self.alpha_ub = 5.0
        self.mu = 0.3
        self.A = A
        self.xstar = xstar
        self.x0 = x0
        self.active = 1
        self.psi = psi
        self.xold = x0
        self.L = np.linalg.norm(self.A)

    def get_active_set(self, x):
        angles = [math.sqrt(self.n)*np.abs(np.dot(self.A[i], x))/np.linalg.norm(self.A[i])/np.linalg.norm(x) for i in range(self.m)]
        violations = [np.abs(self.psi[i]**2 - np.dot(self.A[i], x)**2) for i in range(self.m)]

        h = []
        for i in range(self.m):
            if self.alpha_lb <= angles[i] <= self.alpha_ub and violations[i] <= self.alpha_h*(sum(violations)/self.m)*angles[i]:
                h.append(i)
        return h

    def compute_old_grad(self, active_set):
        s = 0.0
        for i in active_set:
            s += ((self.psi[i] ** 2 - np.dot(self.A[i], self.xold) ** 2) / np.dot(self.A[i], self.xold)) * self.A[i]
        return (1/ self.m) * s

    def update_x(self, active_set, x):
        i = np.random.choice(active_set)
        s_new = ( ( self.psi[i]**2 - np.dot(self.A[i], x)**2 ) / np.dot(self.A[i], x) ) * self.A[i]
        s_old = ( ( self.psi[i]**2 - np.dot(self.A[i], self.xold)**2 ) / np.dot(self.A[i], self.xold) ) * self.A[i]
        return x + (2*self.mu*0.3) * (s_new - s_old + self.compute_old_grad(active_set))

    def run_iter(self, active=1):
        self.active = active
        trial = 0
        err_arr = []
        x = self.x0
        self.xold = self.x0
        while trial < self.niter:
            trial += 1
            active_set = range(self.m)
            if self.active:
                active_set = self.get_active_set(x)
            xold = x
            x = self.update_x(active_set, x)
            if trial>1:
                self.xold = xold
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr


class SAGAmplitudeFlow:

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
        self.xold = x0
        self.psi = psi
        self.L = np.linalg.norm(self.A)

    def get_active_set(self, x):
        return [i for i in range(self.m) if np.abs(np.dot(self.A[i], x)) > self.psi[i] / (1. + self.gamma)]

    def compute_old_grad(self, active_set):
        s = 0.0
        for i in active_set:
            s += (np.dot(self.A[i], self.xold) - self.psi[i] * (np.dot(self.A[i], self.xold) / np.abs(np.dot(self.A[i], self.xold)))) * self.A[i]
        return (1./self.m)*s

    def update_x(self, active_set, x):
        i = np.random.choice(active_set)
        s_new = (np.dot(self.A[i], self.xold) - self.psi[i] * (np.dot(self.A[i], self.xold) / np.abs(np.dot(self.A[i], self.xold)))) * self.A[i]
        s_old = (np.dot(self.A[i], x) - self.psi[i] * (np.dot(self.A[i], x) / np.abs(np.dot(self.A[i], x)))) * self.A[i]
        return x - (1. * self.alpha) * ( self.compute_old_grad(active_set) + s_new - s_old)

    def run_iter(self, active=1, inc=0):
        self.active = active
        trial = 0
        err_arr = []
        x = self.x0
        self.xold = self.x0
        while trial < self.niter:
            trial += 1
            active_set = range(self.m)
            if self.active:
                active_set = self.get_active_set(x)
            if inc:
                index = np.random.choice(active_set)
                active_set = [index]
            xold = x
            x = self.update_x(active_set, x)
            self.xold = xold
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr