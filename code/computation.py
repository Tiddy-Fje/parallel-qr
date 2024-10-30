from mpi4py import MPI
import numpy as np
from utils import *
import time 
import h5py
from scipy.linalg import solve_triangular, block_diag
from scipy import sparse

# Initialize MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_PROCS = COMM.Get_size()
LOGP_TOT = int_check(np.log2(N_PROCS))

N_REPS = 5 # 5,10 -> To average the runtimes
SAVE_RESULTS = True 

# Problem set up 
M = 32768 # 32768 2048
N = 330 # 330 20

def std_print_results(max_runtimes, err_on_norm, Q_cond_number, ending):
    with h5py.File(f'../output/results_n-procs={N_PROCS}.h5', 'a') as f:
        f.create_dataset(f'max_t_avg_{ending}', data=np.mean(max_runtimes))
        f.create_dataset(f'max_t_std_{ending}', data=np.std(max_runtimes))
        f.create_dataset(f'n_rep_{ending}', data=N_REPS)
        if ending[:3] == 'CGS' and N_PROCS == 1: # store the vectors instead of the scalars
            f.create_dataset(f'errs_on_norm_{ending}', data=err_on_norm)
            f.create_dataset(f'Q_cond_numbers_{ending}', data=Q_cond_number)
            err_on_norm = err_on_norm[-1]
            Q_cond_number = Q_cond_number[-1]
        f.create_dataset(f'err_on_norm_{ending}', data=err_on_norm)
        f.create_dataset(f'Q_cond_number_{ending}', data=Q_cond_number)

    print( 'Printing Results for', ending, ':' )
    print( 'Maximal run-times :', max_runtimes.flatten() )
    print( 'Final Q Condition Number :', Q_cond_number )
    print( 'Final Error on Norm :', err_on_norm )

def metrics_from_Q(Q):
    '''
    Compute the error on the norm of Q.T @ Q, as well as the condition number of Q.
    Q : np.array
        The matrix Q from the QR decomposition.
    '''
    norm_error = np.linalg.norm( Q.T @ Q - np.eye(Q.shape[1]) )
    cond_number = np.linalg.cond(Q)
    return norm_error, cond_number

def CQR(A_l, stability=False):
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
        max_runtimes = np.max(tot_runtimes, axis=0) # max over all processors
        err_on_norm, Q_cond_number = metrics_from_Q(Q)
        if stability:
            return err_on_norm, Q_cond_number
        ending = f'CQR_{mat_lab}'
        if SAVE_RESULTS:
            std_print_results(max_runtimes, err_on_norm, Q_cond_number, ending)
    return

def CGS_metrics(Q):
    '''
    Compute the error on the norm of the Q[:,:j].T @ Q[:,:j] matrices to produce a sequence. 
    Q : np.array
        The matrix Q from the QR decomposition.
    '''
    if N_PROCS != 1:
        norm_error, cond_number =  metrics_from_Q(Q)
        return norm_error, cond_number
    
    norm_errors = np.empty(N, dtype=float)
    cond_numbers = np.empty(N, dtype=float)
    if SAVE_RESULTS :
        print('Computing CGS metrics')
    for j in range(N):
        norm_error, cond_number =  metrics_from_Q(Q[:,:j+1])
        norm_errors[j] = norm_error
        cond_numbers[j] = cond_number
    return norm_errors, cond_numbers

def CGS(A_l):
    A_l, mat_lab = A_l
    Q_l = np.empty((A_l.shape[0], N), dtype=float)
    Q = np.empty((M, N), dtype=float) 

    runtimes = np.empty(N_REPS)
    for i in range(N_REPS):
        start = time.perf_counter()
        R = np.zeros((N, N), dtype=float)

        beta_bar_l = np.dot(A_l[:,0], A_l[:,0])
        beta_bar = np.array([0.0])  # Create a writable array for the scalar
        if N_PROCS > 1:
            COMM.Allreduce(beta_bar_l, beta_bar, op=MPI.SUM)
        else:
            beta_bar[0] = beta_bar_l
        R[0,0] = np.sqrt(beta_bar[0])  # Use the result from the array
        Q_l[:,0] = A_l[:,0] / R[0,0]

        for j in range(1, N):
            r_bar_l = Q_l[:,:j].T @ A_l[:,j]
            r_bar = np.empty(j, dtype=float)  # Create a writable array for the scalar
            if N_PROCS > 1:
                COMM.Allreduce(r_bar_l, r_bar, op=MPI.SUM)
            else:
                r_bar = r_bar_l
            R[:j, j] = r_bar  # Assign the result from the array
            Q_l[:,j] = A_l[:,j] - Q_l[:,:j] @ R[:j,j]
            beta_bar_l = np.dot(Q_l[:,j], Q_l[:,j])
            beta_bar = np.array([0.0])  # Writable array
            if N_PROCS > 1:
                COMM.Allreduce(beta_bar_l, beta_bar, op=MPI.SUM)
            else:
                beta_bar[0] = beta_bar_l
            R[j,j] = np.sqrt(beta_bar[0])
            Q_l[:,j] /= R[j,j]

        if N_PROCS > 1:
            COMM.Gather(Q_l, Q, root=0)
        else:
            Q = Q_l
        runtimes[i] = time.perf_counter() - start

    tot_runtimes = None
    if RANK == 0:
        tot_runtimes = np.empty((N_PROCS,N_REPS), dtype=float)
    COMM.Gather(runtimes, tot_runtimes, root=0)

    if RANK == 0:
        errs_on_norm, Q_cond_numbers = CGS_metrics(Q)
        max_runtimes = np.max(tot_runtimes, axis=0) # max over all processors
        ending = f'CGS_{mat_lab}'
        if SAVE_RESULTS:
            std_print_results(max_runtimes, errs_on_norm, Q_cond_numbers, ending)
    return

def get_partner_idx( rank:int, k:int ) -> int:
    idx = 0
    if rank % 2**(k+1) == 0:
        idx = rank + 2**k
    else:
        idx = rank - 2**k
    return idx


def TSQR(A_l):
    A_l, mat_lab = A_l
    runtimes = np.empty(N_REPS)
    for i in range(N_REPS):
        start = time.perf_counter()

        if N_PROCS == 1: # direct QR
            Q, R = np.linalg.qr(A_l, mode='reduced')
            runtimes[i] = time.perf_counter() - start
            continue
        
        Y_l_kp1, R_l_kp1 = np.linalg.qr(A_l, mode='reduced') 
        Ys = [Y_l_kp1]
        for q in range(LOGP_TOT): 
            # only keep needed processors 
            if not (RANK % 2**q == 0):
                continue
            j = get_partner_idx(RANK, q)
            if RANK > j:
                COMM.Send(R_l_kp1, dest=j)
            else:
                R_j_kp1 = np.empty(R_l_kp1.shape, dtype=float)
                COMM.Recv(R_j_kp1, source=j)
                Y_l_k, R_l_k = np.linalg.qr(np.concatenate((R_l_kp1,R_j_kp1), axis=0), mode='reduced')
                R_l_kp1 = R_l_k
                Ys.append(Y_l_k)

        runtimes[i] = time.perf_counter() - start
        Q = build_Q( Ys ) # just moved this out of timing

    tot_runtimes = None
    if RANK == 0:
        tot_runtimes = np.empty((N_PROCS,N_REPS), dtype=float)
    COMM.Gather(runtimes, tot_runtimes, root=0)

    if RANK == 0:
        max_runtimes = np.max(tot_runtimes, axis=0) # max over all processors
        err_on_norm, Q_cond_number = metrics_from_Q(Q)
        ending = f'TSQR_{mat_lab}'
        if SAVE_RESULTS:
            std_print_results(max_runtimes, err_on_norm, Q_cond_number, ending)
    return 

def get_right_mat_shape( p:int ):
    if p == 2 ** LOGP_TOT:
        return (int_check(M/p), N)
    m = 2 * N
    n = N
    return (m,n)

def build_Q( Y_s ):
    current_Q = None
    if RANK == 0: # set up the initial Q
        current_Q = Y_s[-1]
        Y_s.pop()
    for q in range(LOGP_TOT-1,-1,-1):
        # loop from pen-ultimate stage to first stage
        logp = LOGP_TOT - q
        color = 1
        if not (RANK % 2**q == 0):
            color = MPI.UNDEFINED
            new_comm = COMM.Split(color, key=RANK)
            continue
        new_comm = COMM.Split(color, key=RANK)

        p = 2**logp
        if new_comm is not None:
            subs_Q_k = None
            if new_comm.Get_rank() == 0:
                shape = get_right_mat_shape( p )
                subs_Q_k = np.empty((p, *shape), dtype=float) 
            sub_mat = Y_s[-1]
            Y_s.pop()
            new_comm.Gather(sub_mat, subs_Q_k, root=0)

            if new_comm.Get_rank() == 0:
                Q_k = block_diag(*subs_Q_k)
                current_Q = Q_k @ current_Q
            # Gather assembles the object by sorting received data by rank  
    return current_Q


if __name__ == '__main__':
    assert M % N_PROCS == 0, 'Number of processors must divide the number of rows of the matrix'
    if SAVE_RESULTS and RANK == 0: # create the output file
        with h5py.File(f'../output/results_n-procs={N_PROCS}.h5', 'w') as f:
            pass

    mat = None
    sing_mat = None
    if RANK == 0:
        mat = sparse.load_npz(f'../data/csr_{M}_by_{N}_other_mat.npz').toarray()
        sing_mat = get_C(M,N)

    shape = (M//N_PROCS, N)
    mat_l = np.empty(shape, dtype=float)
    sing_mat_l = np.empty(shape, dtype=float)
    COMM.Scatter(mat, mat_l, root=0)
    COMM.Scatter(sing_mat, sing_mat_l, root=0)

    mat_l = [mat_l, 'mat']
    sing_mat_l = [sing_mat_l, 'sing_mat']

    CQR(mat_l)
    CGS(mat_l)
    CGS(sing_mat_l)
    TSQR(mat_l)
    TSQR(sing_mat_l)