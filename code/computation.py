from mpi4py import MPI
import numpy as np
import utils
import time 
import h5py
from scipy.linalg import solve_triangular
from scipy import sparse

# Initialize MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_PROCS = COMM.Get_size()

N_REPS = 5
PRINT_RESULTS = True

# Problem set up 
M = 32768 # 32768 2048
N = 330 # 330 20

assert M % N_PROCS == 0, 'Number of processors must divide the number of rows of the matrix'

with h5py.File(f'../output/results_n-procs={N_PROCS}.h5', 'w') as f:
    pass

def std_print_results(max_runtimes, err_on_norm, Q_cond_number, ending):
    # store the results in a h5py file
    with h5py.File(f'../output/results_n-procs={N_PROCS}.h5', 'a') as f:
        f.create_dataset(f'max_runtime_avg_{ending}', data=np.mean(max_runtimes))
        f.create_dataset(f'max_runtime_std_{ending}', data=np.std(max_runtimes))
        f.create_dataset(f'n_rep_{ending}', data=N_REPS)
        if ending[:3] == 'CGS': # store the vectors instead of the scalars
            f.create_dataset(f'errs_on_norm_{ending}', data=err_on_norm)
            f.create_dataset(f'Q_cond_numbers_{ending}', data=Q_cond_number)
            err_on_norm = err_on_norm[-1]
            Q_cond_number = Q_cond_number[-1]
        else:
            f.create_dataset(f'err_on_norm_{ending}', data=err_on_norm)
            f.create_dataset(f'Q_cond_number_{ending}', data=Q_cond_number)

    if PRINT_RESULTS == True:
        print( 'Printing Results for ', ending, ':' )
        print( 'Maximal run-times : ', max_runtimes.flatten() )
        print( 'Final Q Condition Number : ', Q_cond_number )
        print( 'Final Error on Norm : ', err_on_norm )


def cholesky(A_l):
    A_l, mat_lab = A_l
    G = np.empty((N, N))
    runtimes = np.empty(N_REPS)
    
    for i in range(N_REPS):
        start = time.perf_counter()
        if N_PROCS > 1:
            COMM.Allreduce( A_l.T@A_l, G, op=MPI.SUM )
        else:
            G = A_l.T@A_l
        R = np.linalg.cholesky(G) # cholesky returns the lower triangular matrix
        #// solve R@Q_l.T=A_l.T as we have Q_l=A_l@(R.T)^-1
        Q_l = solve_triangular(R, A_l.T, lower=True).T
        Q = np.empty((M, N), dtype=float) 

        if N_PROCS > 1:
            COMM.Allgather(Q_l, Q)
        else:
            Q = Q_l

        runtimes[i] = time.perf_counter() - start
    
    tot_runtimes = None
    if RANK == 0:
        tot_runtimes = np.empty((N_PROCS,N_REPS), dtype=float)
    COMM.Gather(runtimes, tot_runtimes, root=0)

    if RANK == 0:
        I = np.eye(N)
        max_runtimes = np.max(tot_runtimes, axis=0) # max over all processors
        err_on_norm = np.linalg.norm( I - Q.T@Q )
        Q_cond_number = np.linalg.cond(Q)
        ending = f'cholesky_{mat_lab}'
        std_print_results(max_runtimes, err_on_norm, Q_cond_number, ending)

def CGS_metrics(Q):
    '''
    Compute the error on the norm of the Q[:,:j].T @ Q[:,:j] matrices to produce a sequence. 
    Q : np.array
        The matrix Q from the QR decomposition.
    '''
    #m = Q.shape[0]
    print('Computing metrics')
    n = Q.shape[1]
    mats_squared = [np.dot(Q[:,0], Q[:,0])]
    cond_numbers = [1.0] # assuming that the condition number of a column matrix is 1
    norm_errors = np.empty(n)
    norm_errors[0] = np.abs( mats_squared[0] - 1.0 )
    for j in range(1, n):
        Q_j = Q[:,:j]
        Q_j_squared = Q_j.T @ Q_j
        norm_errors[j] = np.linalg.norm( Q_j_squared - np.eye(j) )
        cond_numbers.append(np.linalg.cond(Q_j))
        mats_squared.append(Q_j_squared)
    return norm_errors, cond_numbers

# question about time measurement 
# we care about max over the processors 
# does that mean we should store times for every processor
# and then take the max over processors of every repetition, right ?
def gram_schmidt(A_l):
    A_l, mat_lab = A_l
    Q_l = np.empty((A_l.shape[0], N), dtype=float)
    Q = np.empty((M, N), dtype=float) 

    runtimes = np.empty(N_REPS)
    for i in range(N_REPS):
        start = time.perf_counter()
        R = np.zeros((N, N), dtype=float)

        beta_bar_l = np.dot(A_l[:,0], A_l[:,0])
        beta_bar = np.array([0.0])  # Create a writable array for the scalar
        COMM.Allreduce(beta_bar_l, beta_bar, op=MPI.SUM)
        R[0,0] = np.sqrt(beta_bar[0])  # Use the result from the array
        Q_l[:,0] = A_l[:,0] / R[0,0]

        for j in range(1, N):
            r_bar_l = Q_l[:,:j].T @ A_l[:,j]
            r_bar = np.empty(j, dtype=float)  # Create a writable array for the scalar
            COMM.Allreduce(r_bar_l, r_bar, op=MPI.SUM)
            R[:j, j] = r_bar  # Assign the result from the array

            Q_l[:,j] = A_l[:,j] - Q_l[:,:j] @ R[:j,j]
            beta_bar_l = np.dot(Q_l[:,j], Q_l[:,j])
            beta_bar = np.array([0.0])  # Writable array
            COMM.Allreduce(beta_bar_l, beta_bar, op=MPI.SUM)
            R[j,j] = np.sqrt(beta_bar[0])

            Q_l[:,j] /= R[j,j]

        if i == 0:
            COMM.Gather(Q_l, Q, root=0)

        runtimes[i] = time.perf_counter() - start

    tot_runtimes = None
    if RANK == 0:
        tot_runtimes = np.empty((N_PROCS,N_REPS), dtype=float)
    COMM.Gather(runtimes, tot_runtimes, root=0)

    if RANK == 0:
        errs_on_norm, Q_cond_numbers = CGS_metrics(Q)
        max_runtimes = np.max(tot_runtimes, axis=0) # max over all processors
        ending = f'CGS_{mat_lab}'
        std_print_results(max_runtimes, errs_on_norm, Q_cond_numbers, ending)


# should we split the Y_l matrices into the sub-matrices on slide 45 ?
# should we store the matrices with each processor and then combine them in another function ? (putting them together is convoluted and would require much communication ??)
def TSQR(A):
    A_list = np.array_split(A, N_PROCS, axis=0)
    A_l = A_list[RANK]

    runtimes = np.empty(N_REPS)
    
    Y_l_kp1, R_l_kp1 = np.linalg.qr(A_l) # check if correct mode of QR
    logp = np.log2(N_PROCS)
    assert logp.is_integer()

    for k in range(int(logp)-1,-1,-1): 
        # take -1 step to get from logp-1 to 0 (as end is excluded)
        # had to adapt the comparisons with k, to the fact that python 
        # starts indexing at 0, while the slides start at 1
        if RANK > 2**(k+1)-1:
            break
        j = (RANK + 2**k) % 2**(k+1) 
        if RANK > j:
            COMM.Send(R_l_kp1, dest=j)
        else:
            R_j_kp1 = np.empty(R_l_kp1.shape, dtype=float)
            COMM.Recv(R_j_kp1, source=j)
            Y_l_k, R_l_k = np.linalg.qr(np.concatenate((R_l_kp1,R_j_kp1), axis=0)) # check if lower or upper triangular 
            R_l_kp1 = R_l_k
            Y_l_kp1 = Y_l_k
        pass
    if RANK == 0:
        #R = R_l_k
        ending = 'TSQR'
        #Q_cond_number = np.linalg.cond(Q)
        #std_print_results(tot_runtimes, Q_cond_number, ending)
        with h5py.File(f'../output/results_n-procs={N_PROCS}.h5', 'w') as f:
            f.create_dataset(f'R_{ending}', data=R_l_k)

    return 



mat = None
sing_mat = None
if RANK == 0:
    mat = sparse.load_npz(f'../data/csr_{M}_by_{N}_other_mat.npz').toarray()
    sing_mat = utils.get_C(M,N)

mat_l = np.empty((M//N_PROCS, N), dtype=float)
sing_mat_l = np.empty((M//N_PROCS, N), dtype=float)
COMM.Scatter(mat, mat_l, root=0)
COMM.Scatter(sing_mat, sing_mat_l, root=0)
mat_l = [mat_l, 'mat']
sing_mat_l = [sing_mat_l, 'sing_mat']


cholesky(mat_l)
#cholesky(sing_mat_l)
gram_schmidt(mat_l)
