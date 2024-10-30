from matplotlib import pyplot as plt
import numpy as np
import h5py
import utils
import pandas as pd

IN_PATH = '../output/'
FIG_PATH = '../figures/'
SHOW_PLOTS = False

def load_mat_metrics(file, mat_lab, data):
    with h5py.File(file, 'r') as f:
        if not mat_lab == 'sing_mat':
            data[f'CQR_err_{mat_lab}'] = f[f'err_on_norm_CQR_{mat_lab}'][()]
            data[f'CQR_cond_{mat_lab}'] = f[f'Q_cond_number_CQR_{mat_lab}'][()]
        data[f'CGS_errs_{mat_lab}'] = f[f'errs_on_norm_CGS_{mat_lab}'][:]
        data[f'CGS_conds_{mat_lab}'] = f[f'Q_cond_numbers_CGS_{mat_lab}'][:]
        data[f'TSQR_err_{mat_lab}'] = f[f'err_on_norm_TSQR_{mat_lab}'][()]
        data[f'TSQR_cond_{mat_lab}'] = f[f'Q_cond_number_TSQR_{mat_lab}'][()]

def load_mat_runtimes(file, mat_lab, CQR_data, CGS_data, TSQR_data):
    with h5py.File(file, 'r') as f:
        if not mat_lab == 'sing_mat':
            CQR_data[f'CQR_avg_t_{mat_lab}'] = f[f'max_t_avg_CQR_{mat_lab}'][()]
            CQR_data[f'CQR_std_t_{mat_lab}'] = f[f'max_t_std_CQR_{mat_lab}'][()]    
            CQR_data[f'CQR_n_rep_{mat_lab}'] = f[f'n_rep_CQR_{mat_lab}'][()]
        CGS_data[f'CGS_avg_t_{mat_lab}'] = f[f'max_t_avg_CGS_{mat_lab}'][()]
        CGS_data[f'CGS_std_t_{mat_lab}'] = f[f'max_t_std_CGS_{mat_lab}'][()]
        CGS_data[f'CGS_n_rep_{mat_lab}'] = f[f'n_rep_CGS_{mat_lab}'][()]
        TSQR_data[f'TSQR_avg_t_{mat_lab}'] = f[f'max_t_avg_TSQR_{mat_lab}'][()]
        TSQR_data[f'TSQR_std_t_{mat_lab}'] = f[f'max_t_std_TSQR_{mat_lab}'][()]
        TSQR_data[f'TSQR_n_rep_{mat_lab}'] = f[f'n_rep_TSQR_{mat_lab}'][()]

def import_data( ns_procs:np.array ):
    data_list_CQR = []
    data_list_CGS = []
    data_list_TSQR = []

    data = {}
    for n_procs in ns_procs :
        CQR_data = {}
        CGS_data = {}
        TSQR_data = {}

        file = f'{IN_PATH}results_n-procs={n_procs}.h5'
        with h5py.File(file, 'r') as f:
            #print(file,f.keys())
            CQR_data['n_procs'] = n_procs
            CGS_data['n_procs'] = n_procs
            TSQR_data['n_procs'] = n_procs

        load_mat_runtimes(file, 'mat', CQR_data, CGS_data, TSQR_data)
        load_mat_runtimes(file, 'sing_mat', CQR_data, CGS_data, TSQR_data)
        if n_procs == 1:
            load_mat_metrics(file, 'mat', data)
            load_mat_metrics(file, 'sing_mat', data)

        data_list_CQR.append(CQR_data)
        data_list_CGS.append(CGS_data)
        data_list_TSQR.append(TSQR_data)

    # create a pandas dataframe, idexed by the number of processes
    data['t_df_CQR'] = pd.DataFrame(data_list_CQR, index=ns_procs)
    data['t_df_CGS'] = pd.DataFrame(data_list_CGS, index=ns_procs)
    data['t_df_TSQR'] = pd.DataFrame(data_list_TSQR, index=ns_procs)
    
    return data


def plot_norm_error(norm_errors, n_procs):
    '''
    Plot the errors on the norm as a function of the number of included vectors in the basis. This should be done for each number of processes.
    norm_errors : np.ndarray
        The i-th row contains teh norm_errors for the n_procs[i] processors.
    '''
    fig, ax = plt.subplots()
    for i in range(len(n_procs)):
        ax.plot(norm_errors[i], label=f'{n_procs[i]} processes')
    ax.set_xlabel('Number of included vectors in the basis')
    ax.set_ylabel('Error on the norm')
    ax.legend()
    plt.show()


def CGS_plot(data): 
    '''
    Plot CGS metrics evolution as a function of included number of columns.
    data : dict
        imported data
    '''
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    labels = [r'Non-singular $A$', r'Singular $A$']
    for i, mat_lab in enumerate(['mat', 'sing_mat']):
        norm_errors = data[f'CGS_errs_{mat_lab}']
        cond_numbers = data[f'CGS_conds_{mat_lab}']
        n = len(norm_errors)
        ns = np.arange(1, n+1)
        ax[0].plot(ns, norm_errors, label=labels[i])
        ax[1].plot(ns, cond_numbers, label=labels[i])
    #ax[0].set_xlabel('Number of included vectors in the basis')
    ax[0].set_ylabel(r'$\|I-Q^TQ|\|$')
    ax[0].set_yscale('log')
    #ax[0].set_xscale('log')
    ax[0].legend()
    ax[1].set_xlabel(r'Number of vectors included in $Q$')
    ax[1].set_ylabel(r'$\kappa(Q)$')
    ax[1].set_yscale('log')
    ax[1].legend()

    plt.savefig(f'{FIG_PATH}CGS_metric_evolution.png')
    if SHOW_PLOTS:
        plt.show()
    return

def runtime_vs_nprocs(data, mat_lab='mat'):
    '''
    Plot average runtime as a function of the number of processes.
    data : dict
        imported data
    mat_lab : str
        The label of the matrix to plot the runtime for.
    '''
    fig, ax = plt.subplots()
    algos = ['CQR', 'CGS', 'TSQR']
    baseline = 0.0
    for algo in algos:
        time_df = data[f't_df_{algo}']
        std_on_mean = time_df[f'{algo}_std_t_{mat_lab}'] / np.sqrt(time_df[f'{algo}_n_rep_{mat_lab}'])
        if algo == 'CQR':
            baseline = time_df[f'{algo}_avg_t_{mat_lab}'][1]
            baseline = 1.0
        ax.errorbar(time_df['n_procs'], time_df[f'{algo}_avg_t_{mat_lab}']/baseline, yerr=std_on_mean/baseline, fmt='o', capsize=5, label=algo)
    ax.set_xlabel('Number of processors')
    ax.set_ylabel('Performance [s]')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(f'{FIG_PATH}runtime_vs_nprocs_{mat_lab}.png')

    if SHOW_PLOTS:
        plt.show()



print('Running core plots')
#data = import_data(np.array([1,2,4]))
data = import_data(np.array([1,2,4,8,16,32,64]))
CGS_plot(data)
runtime_vs_nprocs(data)
print('Running stability plot')
import cholesky_stability
