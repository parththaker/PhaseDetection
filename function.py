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


class GaussMethods:

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
        # f += (self.L / 2.) * (np.linalg.norm(x - z) ** 2)
        return f


class TrustRegionMethods:

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
            sol = opt.minimize(self.func, x, args=(x_old, order, inc, index), method='trust-constr',
                               options={'xtol': err_bound, 'initial_tr_radius': 1.0, 'disp': True})
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
        # f += (self.L / 2.) * (np.linalg.norm(x - z) ** 2)
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


class SubgradientFlowLinf:

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
        self.tstep = 1.0
        self.L = np.linalg.norm(self.A)

    def get_active_set(self, x):
        angles = [math.sqrt(self.n) * np.abs(np.dot(self.A[i], x)) / np.linalg.norm(self.A[i]) / np.linalg.norm(x) for i
                  in range(self.m)]
        violations = [np.abs(self.psi[i] ** 2 - np.dot(self.A[i], x) ** 2) for i in range(self.m)]

        h = []
        for i in range(self.m):
            if self.alpha_lb <= angles[i] <= self.alpha_ub and violations[i] <= self.alpha_h * (
                    sum(violations) / self.m) * angles[i]:
                h.append(i)
        return h

    def update_x(self, active_set, x):
        s = 0.0
        f = 0.0
        max_arr = np.argmax([np.abs(np.dot(self.A[i], x)**2 - self.psi[i]**2) for i in range(self.m)])
        # print(max_arr, isinstance(max_arr, np.int64))
        if isinstance(max_arr, np.int64):
            max_arr = np.array([max_arr])
        for i in max_arr:
            s += 2 * np.dot(self.A[i], x) * np.sign(np.dot(self.A[i], x) ** 2 - self.psi[i] ** 2) * self.A[i]
            f += np.abs(np.dot(self.A[i], x) ** 2 - self.psi[i] ** 2)
        return x - f*(1.0/len(max_arr)) * s/(np.linalg.norm(s)**2)
        # return x - self.tstep*(1.0 / len(max_arr)) * s

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
            # self.tstep = (self.tstep)*(trial/(trial+1))
            self.tstep = self.tstep*1.0
            error = support.check_error(self.A, x, self.xstar, self.m, order = np.inf)
            err_arr.append(error)
        return err_arr

class SubgradientFlowL1:

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
        self.tstep = 1.0
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
        f = 0.0
        for i in active_set:
            s += 2*np.dot(self.A[i], x)*np.sign(np.dot(self.A[i], x)**2 - self.psi[i]**2) * self.A[i]
            f += np.abs(np.dot(self.A[i], x)**2 - self.psi[i]**2)
        # return x - f*(10.0/self.m) * s/(np.linalg.norm(s)**2)
        return x - self.tstep*(1.0/self.m) * s

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
            frac = (trial/(trial+1))
            # self.tstep = 1.0/(trial*math.log(trial+1))
            self.tstep = self.tstep*0.8
            error = support.check_error (self.A, x, self.xstar, self.m, order=1)
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
        self.grad_arr = np.zeros((self.m, self.n))
        self.L = np.linalg.norm(self.A)

    def initialize_grad(self):
        for i in range(self.m):
            self.grad_arr[i] = ((self.psi[i] ** 2 - np.dot(self.A[i], self.x0) ** 2) / np.dot(self.A[i], self.x0)) * self.A[i]

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
            # s += ((self.psi[i] ** 2 - np.dot(self.A[i], self.xold) ** 2) / np.dot(self.A[i], self.xold)) * self.A[i]
            s += self.grad_arr[i]
        return s

    def update_x(self, active_set, x):
        i = np.random.choice(active_set)
        s_new = ( ( self.psi[i]**2 - np.dot(self.A[i], x)**2 ) / np.dot(self.A[i], x) ) * self.A[i]
        # s_old = ( ( self.psi[i]**2 - np.dot(self.A[i], self.xold)**2 ) / np.dot(self.A[i], self.xold) ) * self.A[i]
        s_old = self.grad_arr[i]
        s_agg = self.compute_old_grad(active_set)
        self.grad_arr[i] = s_new
        return x + (2*self.mu*0.5/self.m) * (s_new - s_old + s_agg)

    def run_iter(self, active=1):
        self.initialize_grad()
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
        self.grad_arr = np.zeros((self.m, self.n))
        self.L = np.linalg.norm(self.A)


    def initialize_grad(self):
        for i in range(self.m):
            self.grad_arr[i] = (np.dot(self.A[i], self.x0) - self.psi[i] * (np.dot(self.A[i], self.x0) / np.abs(np.dot(self.A[i], self.x0)))) * self.A[i]

    def get_active_set(self, x):
        return [i for i in range(self.m) if np.abs(np.dot(self.A[i], x)) > self.psi[i] / (1. + self.gamma)]

    def compute_old_grad(self, active_set):
        s = 0.0
        for i in active_set:
            # s += (np.dot(self.A[i], self.xold) - self.psi[i] * (np.dot(self.A[i], self.xold) / np.abs(np.dot(self.A[i], self.xold)))) * self.A[i]
            s += self.grad_arr[i]
        return s

    def update_x(self, active_set, x):
        i = np.random.choice(active_set)
        s_new = (np.dot(self.A[i], self.xold) - self.psi[i] * (np.dot(self.A[i], self.xold) / np.abs(np.dot(self.A[i], self.xold)))) * self.A[i]
        # s_old = (np.dot(self.A[i], x) - self.psi[i] * (np.dot(self.A[i], x) / np.abs(np.dot(self.A[i], x)))) * self.A[i]
        s_old = self.grad_arr[i]
        s_agg = self.compute_old_grad(active_set)
        self.grad_arr[i]=s_new
        return x - (0.5 * self.alpha/len(active_set)) * ( s_agg + s_new - s_old)

    def run_iter(self, active=1, inc=0):
        self.initialize_grad()
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
            self.xold = xold
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr


class ExpNormMethods:

    def __init__(self, n, m, niter, A, xstar, x0, psi):
        self.niter = niter
        self.n = n
        self.m = m
        self.A = A
        self.xstar = xstar
        self.x0 = x0
        self.psi = psi
        self.xold = x0
        self.L = np.linalg.norm(self.A)

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

    def update_x(self, x):
        s = (self.L)*np.eye(self.n)
        g = 0.0
        for i in range(self.m):
            grad = np.dot(np.outer(self.A[i], self.A[i]), x)
            s += (2.0/self.n)*(np.outer(grad, grad))
            g += (2/self.n)*(np.dot(self.A[i], x)**2 - self.psi[i]**2)*grad
        return x - np.dot(np.linalg.inv(s), g)


class SAGExpNormMethods:

    def __init__(self, n, m, niter, A, xstar, x0, psi):
        self.niter = niter
        self.n = n
        self.m = m
        self.A = A
        self.xstar = xstar
        self.x0 = x0
        self.psi = psi
        self.xold = x0
        self.L = np.linalg.norm(self.A)

    def run_iter(self):
        trial = 0
        err_arr = []
        x = self.x0
        self.xold = self.x0
        while trial < self.niter:
            trial += 1
            xold = x
            x = self.update_x(x)
            self.xold = xold
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr

    def compute_old_gradient(self):
        s = (self.L)*np.eye(self.n)
        g = 0.0
        for i in range(self.m):
            grad = np.dot(np.outer(self.A[i], self.A[i]), self.xold)
            s += (2.0/self.n)*(np.outer(grad, grad))
            g += (2.0/self.n)*(np.dot(self.A[i], self.xold)**2 - self.psi[i]**2)*grad
        return np.dot(np.linalg.inv(s), g)

    def update_x(self, x):
        i = np.random.choice(range(self.m))
        grad = np.dot(np.outer(self.A[i], self.A[i]), x)
        s = (self.L)*np.eye(self.n) + 2.0*(np.outer(grad, grad))
        g = 2*(np.dot(self.A[i], x)**2 - self.psi[i]**2)*grad

        grad_old = np.dot(np.outer(self.A[i], self.A[i]), self.xold)
        s_old = (self.L)*np.eye(self.n) + 2.0*(np.outer(grad_old, grad_old))
        g_old = 2*(np.dot(self.A[i], self.xold)**2 - self.psi[i]**2)*grad_old

        return x - 0.4*(np.dot(np.linalg.inv(s), g) - np.dot(np.linalg.inv(s_old), g_old) + self.compute_old_gradient())

class SAGNormMethods:

    def __init__(self, n, m, niter, A, xstar, x0, psi):
        self.niter = niter
        self.n = n
        self.m = m
        self.A = A
        self.xstar = xstar
        self.x0 = x0
        self.psi = psi
        self.xold = self.x0
        self.L = np.linalg.norm(self.A)

    def run_iter(self):
        trial = 0
        err_arr = []
        x_old = self.x0
        x = self.x0
        while trial < self.niter:
            trial += 1
            xold = x
            x = self.update_x(x)
            self.xold = xold
            error = support.check_error(self.A, x, self.xstar, self.m)
            err_arr.append(error)
        return err_arr

    def compute_old_gradient(self):
        s = 0.0
        for i in range(self.m):
            if np.dot(self.A[i], self.xold)**2 - self.psi[i]**2 < -4*(np.dot(self.A[i], self.xold)**2)*np.linalg.norm(self.A[i])**2:
                s += 2*(self.L)*np.dot(self.A[i], self.xold)*self.A[i]
            elif np.dot(self.A[i], self.xold)**2 - self.psi[i]**2 > 4*(np.dot(self.A[i], self.xold)**2)*np.linalg.norm(self.A[i])**2:
                s -= 2 * (self.L) * np.dot(self.A[i], self.xold) * self.A[i]
            else:
                s += ((np.dot(self.A[i], self.xold)**2 - self.psi[i]**2)/(4*(np.dot(self.A[i], self.xold)**2)*np.linalg.norm(self.A[i])**2))*2 * (self.L) * np.dot(self.A[i], self.xold) * self.A[i]
        return (1./self.m)*s

    def update_x(self, x):
        i = np.random.choice(range(self.m))
        if np.dot(self.A[i], self.xold) ** 2 - self.psi[i] ** 2 < -4 * (
                np.dot(self.A[i], self.xold) ** 2) * np.linalg.norm(self.A[i]) ** 2:
            sold = 2 * (self.L) * np.dot(self.A[i], self.xold) * self.A[i]
        elif np.dot(self.A[i], self.xold) ** 2 - self.psi[i] ** 2 > 4 * (
                np.dot(self.A[i], self.xold) ** 2) * np.linalg.norm(self.A[i]) ** 2:
            sold = - 2 * (self.L) * np.dot(self.A[i], self.xold) * self.A[i]
        else:
            sold = ((np.dot(self.A[i], self.xold) ** 2 - self.psi[i] ** 2) / (
                        4 * (np.dot(self.A[i], self.xold) ** 2) * np.linalg.norm(self.A[i]) ** 2)) * 2 * (
                     self.L) * np.dot(self.A[i], self.xold) * self.A[i]

        if np.dot(self.A[i], x) ** 2 - self.psi[i] ** 2 < -4 * (
                np.dot(self.A[i], x) ** 2) * np.linalg.norm(self.A[i]) ** 2:
            snew = 2 * (self.L) * np.dot(self.A[i], x) * self.A[i]
        elif np.dot(self.A[i], x) ** 2 - self.psi[i] ** 2 > 4 * (
                np.dot(self.A[i], x) ** 2) * np.linalg.norm(self.A[i]) ** 2:
            snew = - 2 * (self.L) * np.dot(self.A[i], x) * self.A[i]
        else:
            snew = ((np.dot(self.A[i], x) ** 2 - self.psi[i] ** 2) / (
                        4 * (np.dot(self.A[i], x) ** 2) * np.linalg.norm(self.A[i]) ** 2)) * 2 * (
                     self.L) * np.dot(self.A[i], x) * self.A[i]
        return x - 0.1*(snew-sold +self.compute_old_gradient())


class SampleGrad:

    def __init__(self, n, m, niter, A, xstar, x0, psi):
        self.niter = niter
        self.n = n
        self.m = m
        self.A = A
        self.xstar = xstar
        self.x0 = x0
        self.psi = psi
        self.xold = x0
        self.grad_arr = np.zeros((self.m, self.n))
        self.L = np.linalg.norm(self.A)

    def run_iter(self):
        trial = 0
        err_arr = []
        x = self.x0
        while trial < self.niter:
            trial += 1
            x = self.update_x(x)
            error = support.check_error(self.A, x, self.xstar, self.m, option=2)
            err_arr.append(error)
        return err_arr

    def update_x(self, x):
        s = (self.L)*np.eye(self.n)
        g = 0.0
        for i in range(self.m):
            grad = np.dot(np.outer(self.A[i], self.A[i]), x)
            s += (2.0/self.n)*(np.outer(grad, grad))
            g += (2/self.n)*(np.dot(self.A[i], x)**2 - self.psi[i]**2)*grad
        return x - np.dot(np.linalg.inv(s), g)