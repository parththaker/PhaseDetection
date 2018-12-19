import numpy as np
import matplotlib.pyplot as plt
import function
import support

niter = 100
error_bound = 1e-6
n=5
m=100

A, x_star, x_0, psi = support.generate_random_quant(n, m)

norm_method = function.NormMethods(n, m, niter, A, x_star, x_0, psi)
taf = function.AmplitudeFlow(n, m, niter, A, x_star, x_0, psi)

first_arr = norm_method.run_iter(error_bound, order = 1)
second_arr = norm_method.run_iter(error_bound, order = 2)
third_arr = norm_method.run_iter(error_bound, order = np.inf)
fourth_arr = taf.run_iter(active=0)
fifth_arr = taf.run_iter(active=1)

plt.figure(1)
plt.semilogy(range(niter), fourth_arr, label='AF')
plt.semilogy(range(niter), fifth_arr, label='TAF')
plt.semilogy(range(niter), first_arr, label='Prox. l1')
plt.semilogy(range(niter), second_arr, label='Prox. l2')
plt.semilogy(range(niter), third_arr, label='Prox. linf')
plt.xlabel('Iterate no.')
plt.ylabel(r'$\sum_{i=1}^m||a_i^Tx - \psi_i||$')
plt.title('Error comparision between TAF, Prox. Linear and Augmeneted lagrangian approach')
plt.legend()
plt.show()
