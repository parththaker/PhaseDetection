import numpy as np
import matplotlib.pyplot as plt
import function
import support

niter = 100
error_bound = 1e-6
n=30
m=500

A, x_star, x_0, psi = support.generate_random_quant(n, m)

x_0 = support.TWFinitialization(m, n, A, psi)
psi = support.corrupt_readings(psi, type='out')

norm_method = function.NormMethods(n, m, niter, A, x_star, x_0, psi)
gauss_method = function.GaussMethods(n, m, niter, A, x_star, x_0, psi)
trust_reg_method = function.GaussMethods(n, m, niter, A, x_star, x_0, psi)
taf = function.AmplitudeFlow(n, m, niter, A, x_star, x_0, psi)
twf = function.WirtingerFlow(n, m, niter, A, x_star, x_0, psi)
stwf = function.SAGWirtingerFlow(n, m, niter, A, x_star, x_0, psi)
staf = function.SAGAmplitudeFlow(n, m, niter, A, x_star, x_0, psi)
kz = function.Kaczmarc(n, m, niter, A, x_star, x_0, psi)
expmethod = function.ExpNormMethods(n, m, niter, A, x_star, x_0, psi)
sagexp = function.SAGExpNormMethods(n, m, niter, A, x_star, x_0, psi)
sagnorm = function.SAGNormMethods(n, m, niter, A, x_star, x_0, psi)
subL1 = function.SubgradientFlowL1(n, m, niter, A, x_star, x_0, psi)
subLinf = function.SubgradientFlowLinf(n, m, niter, A, x_star, x_0, psi)


# linf_arr = norm_method.run_iter(error_bound, order = np.inf)
# l1_arr = norm_method.run_iter(error_bound, order = 1)
# l2_arr = norm_method.run_iter(error_bound, order = 2)
#
# glinf_arr = gauss_method.run_iter(error_bound, order = np.inf)
# gl1_arr = gauss_method.run_iter(error_bound, order = 1)
# gl2_arr = gauss_method.run_iter(error_bound, order = 2)
#
# trlinf_arr = trust_reg_method.run_iter(error_bound, order = np.inf)
# trl1_arr = trust_reg_method.run_iter(error_bound, order = 1)
# trl2_arr = trust_reg_method.run_iter(error_bound, order = 2)

# l1_inc_arr = norm_method.run_iter(error_bound, order = 1, inc=1)
# l2_inc_arr = norm_method.run_iter(error_bound, order = 2, inc=1)
# taf_arr = taf.run_iter(active=1)
# af_arr = taf.run_iter(active=0)
#
# taf_inc_arr = taf.run_iter(active=1, inc=1)
wf_arr = twf.run_iter(active=0)
twf_arr = twf.run_iter(active=1)
# twf_inc_arr = twf.run_iter(active=1, inc=1)
# sag_twf_arr = stwf.run_iter(active=1)
# sag_wf_arr = stwf.run_iter(active=0)
# sag_taf_arr = staf.run_iter(active=1)
# sag_af_arr = staf.run_iter(active=0)
kz_arr = kz.run_iter()
# exp_arr = expmethod.run_iter()
# exp_sag_arr = sagexp.run_iter()
# sag_norm_arr = sagnorm.run_iter()
#
sub_L1_arr = subL1.run_iter(active=0)
sub_Linf_arr = subLinf.run_iter(active=0)


plt.figure(1)
# plt.semilogy(range(niter), af_arr, label='AF')
plt.semilogy(range(niter), wf_arr, label='WF')
#
# plt.semilogy(range(niter), taf_arr, label='TAF')
plt.semilogy(range(niter), twf_arr, label='TWF')
plt.semilogy(range(niter), sub_L1_arr, label='L1 Sub')
plt.semilogy(range(niter), sub_Linf_arr, label='Linf Sub')

# plt.semilogy(range(niter), taf_inc_arr, label='TAF Inc')
# plt.semilogy(range(niter), twf_inc_arr, label='TWF Inc')

# plt.semilogy(range(niter), sag_wf_arr, label='SAG WF')
# plt.semilogy(range(niter), sag_af_arr, label='SAG AF')
#
# plt.semilogy(range(niter), sag_twf_arr, label='SAG TWF')
# plt.semilogy(range(niter), sag_taf_arr, label='SAG TAF')
#
plt.semilogy(range(niter), kz_arr, label='Kaczmarc')
run_avg = support.running_avg(sub_L1_arr)
run_avg_inf = support.running_avg(sub_Linf_arr)

plt.semilogy(range(len(run_avg)), run_avg, label='L1 Run Sub')
plt.semilogy(range(len(run_avg)), run_avg_inf, label='Linf Run Sub')


# plt.semilogy(range(niter), l1_arr, label='Prox. l1')
# plt.semilogy(range(niter), l2_arr, label='Prox. l2')
# plt.semilogy(range(niter), linf_arr, label='Prox. linf')
#
# plt.semilogy(range(niter), gl1_arr, label='Gauss. l1')
# plt.semilogy(range(niter), gl2_arr, label='Gauss. l2')
# plt.semilogy(range(niter), glinf_arr, label='Gauss. linf')
#
# plt.semilogy(range(niter), trl1_arr, label='TR. l1')
# plt.semilogy(range(niter), trl2_arr, label='TR. l2')
# plt.semilogy(range(niter), trlinf_arr, label='TR. linf')

# plt.semilogy(range(niter), l1_inc_arr, label='Prox. Inc. l1')
# plt.semilogy(range(niter), l2_inc_arr, label='Prox. Inc. l2')
#
# plt.semilogy(range(niter), exp_arr, label='Exp')
# plt.semilogy(range(niter), exp_sag_arr, label='SAG Exp')
# plt.semilogy(range(niter), sag_norm_arr, label='SAG Prox. l1')

plt.xlabel('Iterate no.')
plt.ylabel(r'$\sum_{i=1}^m|a_i^Tx - \psi_i|$')
plt.title('Constraint violation vs iteration no. of SAG Prox variant with Kaczmarc and other flow approachs')
plt.legend()
plt.show()
