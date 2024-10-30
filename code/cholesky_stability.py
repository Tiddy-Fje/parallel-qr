import numpy as np
import matplotlib.pyplot as plt
import utils
from computation import CQR

IN_PATH = '../output/'
FIG_PATH = '../figures/'
SHOW_PLOTS = False

# Problem set up
base_M = 2000
base_N = 5 
scales = np.linspace(1, 30, 10)

err_on_norm = np.zeros(len(scales))
cond_Q = np.zeros(len(scales))
cond_A = np.zeros(len(scales))
for i,scale in enumerate(scales):
    M = int(scale * base_M)
    N = int(scale * base_N)
    print(f'M = {M}, N = {N}')
    mat = utils.get_C(M,N)
    cond_A[i] = np.linalg.cond(mat)
    mat = [mat, 'whatever']
    err_on_norm[i], cond_Q[i] = CQR(mat, stability=True)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=True)
ax.plot(cond_A, err_on_norm, label='Data')
ax.plot(cond_A,(err_on_norm[0]/cond_A[0])*cond_A**2, 'r--', label='Line w/ slope 2')
ax.set_ylabel(r'$\|I-Q^TQ|\|$')   
ax.set_xlabel(r'$\kappa(A)$')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
if SHOW_PLOTS:
    plt.show()
plt.savefig(f'{FIG_PATH}CQR_stability.png')
