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
twf = function.WirtingerFlow(n, m, niter, A, x_star, x_0, psi)
kz = function.Kaczmarc(n, m, niter, A, x_star, x_0, psi)

l1_arr = norm_method.run_iter(error_bound, order = 1)
l2_arr = norm_method.run_iter(error_bound, order = 2)
linf_arr = norm_method.run_iter(error_bound, order = np.inf)
af_arr = taf.run_iter(active=0)
taf_arr = taf.run_iter(active=1)
wf_arr = twf.run_iter(active=0)
twf_arr = twf.run_iter(active=1)
kz_arr = kz.run_iter()

plt.figure(1)
plt.semilogy(range(niter), af_arr, label='AF')
plt.semilogy(range(niter), taf_arr, label='TAF')
plt.semilogy(range(niter), wf_arr, label='WF')
plt.semilogy(range(niter), twf_arr, label='TWF')
plt.semilogy(range(niter), kz_arr, label='Kaczmarc')
plt.semilogy(range(niter), l1_arr, label='Prox. l1')
plt.semilogy(range(niter), l2_arr, label='Prox. l2')
plt.semilogy(range(niter), linf_arr, label='Prox. linf')
plt.xlabel('Iterate no.')
plt.ylabel(r'$\sum_{i=1}^m||a_i^Tx - \psi_i||$')
plt.title('Error comparision between TAF, Prox. Linear and Augmeneted lagrangian approach')
plt.legend()
plt.show()
