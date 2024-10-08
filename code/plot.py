from matplotlib import pyplot as plt
import numpy as np
import h5py
import pandas as pd

input_path = '../output/'

def compute_QtQ(Q):
    '''
    Compute the error on the norm of the Q[:,:j].T @ Q[:,:j] matrices to produce a sequence. The computation for Q[:,:j+1].T @ Q[:,:j+1] is done by using the previous result Q[:,:j].T @ Q[:,:j] and the new column of Q.
    Q : np.array
        The matrix Q from the QR decomposition.
    '''
    m = Q.shape[0]
    n = Q.shape[1]
    mats_squared = [np.linalg.norm(Q[:,0])]
    mat = Q[:,0:2]
    cond_numbers = [np.linalg.cond(mat)]
    mat_squared = mat.T @ mat
    mats_squared.append(mat_squared)
    for j in range(2, n):
        new_Q = Q[:,j]
        Q_j_squared = np.empty((j+1, j+1))

        Q_j_11 = mat_squared 
        Q_j_12 = mat.T @ new_Q
        Q_j_21 = new_Q.T @ mat 
        Q_j_22 = np.dot(new_Q, new_Q)

        Q_j_squared[:-1,:-1] = Q_j_11
        Q_j_squared[:-1,-1] = Q_j_12
        Q_j_squared[-1,:-1] = Q_j_21
        Q_j_squared[-1,-1] = Q_j_22

        mat = Q[:,:j+1]
        mat_squared = Q_j_squared
        cond_numbers.append(np.linalg.cond(mat))
        mats_squared.append(Q_j_squared)
        
    return mats_squared, cond_numbers

def norm_error_from_QtQ(QtQ):
    '''
    Compute the error on the norm of the Q[:,:j].T @ Q[:,:j] matrices to produce a sequence.
    QtQ : list
        List of matrices Q[:,:j].T @ Q[:,:j] for j=1,...,n.
    '''
    n = QtQ[-1].shape[0]
    assert len(QtQ) == n
    norm_errors = np.empty(n)
    for j in range(n):
        norm_errors[j] = np.linalg.norm( QtQ[j] - np.eye(j+1) )
    return norm_errors

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


def runtime_vs_nprocs(df_cholesky):
    '''
    Plot the average runtime as a function of the number of processes.
    df_cholesky : pd.DataFrame
        The dataframe containing the average runtime for the Cholesky decomposition.
    '''
    fig, ax = plt.subplots()
    ax.errorbar(df_cholesky['n_procs'], df_cholesky['avg_runtime'], yerr=df_cholesky['std_runtime'], fmt='o')
    ax.set_xlabel('Number of processes')
    ax.set_ylabel('Average runtime')
    plt.show()


def import_data(ns_procs:np.array):
    data_list_cholesky = []
    data_list_gram = []

    for n_procs in ns_procs:
        cholesky_data = {}
        gram_data = {}
        with h5py.File(f'{input_path}results_n-procs={n_procs}.h5', 'r') as f:
            print(f.keys())
            cholesky_data['avg_runtime'] = f['max_runtime_avg_cholesky_mat'][()]
            cholesky_data['std_runtime'] = f['max_runtime_std_cholesky_mat'][()]
            cholesky_data['n_rep'] = f['n_rep_cholesky_mat'][()]
            cholesky_data['err_on_norm'] = f['err_on_norm_cholesky_mat'][()]
            cholesky_data['Q_cond_number'] = f['Q_cond_number_cholesky_mat'][()]
            cholesky_data['n_procs'] = n_procs
            gram_data['Q'] = f['Q_gram_schmidt'][:]
        data_list_cholesky.append(cholesky_data)
        data_list_gram.append(gram_data)

    # create a pandas dataframe, idexed by the number of processes
    df_cholesky = pd.DataFrame(data_list_cholesky, index=ns_procs)
    return df_cholesky, data_list_gram

df_cholesky, data_list_gram = import_data(np.array([1,2,4,8]))
runtime_vs_nprocs(df_cholesky)

#print(norm_errors,'\n',norm_errors2)
#plot_norm_error(((norm_errors,norm_errors2)), [2,4])