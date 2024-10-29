from mpi4py import MPI
import numpy as np
import utils
import time 
import h5py
from scipy.linalg import solve_triangular, block_diag, qr
from scipy import sparse

# Initialize MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_PROCS = COMM.Get_size()

LOGP_TOT = np.log2(N_PROCS)
assert LOGP_TOT.is_integer()
LOGP_TOT = int(LOGP_TOT)

N_REPS = 1 # 5,10
PRINT_RESULTS = False
MODE = 'complete' # 'reduced' or 'complete'

# Problem set up 
M = 50000 # 32768 2048
N = 600 # 330 20

assert M % N_PROCS == 0, 'Number of processors must divide the number of rows of the matrix'

if RANK == 0:
    with h5py.File(f'../output/results_n-procs={N_PROCS}.h5', 'w') as f:
        pass

def std_print_results(max_runtimes, err_on_norm, Q_cond_number, ending):
    with h5py.File(f'../output/results_n-procs={N_PROCS}.h5', 'a') as f:
        f.create_dataset(f'max_t_avg_{ending}', data=np.mean(max_runtimes))
        f.create_dataset(f'max_t_std_{ending}', data=np.std(max_runtimes))
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
        print( 'Printing Results for', ending, ':' )
        print( 'Maximal run-times :', max_runtimes.flatten() )
        print( 'Final Q Condition Number :', Q_cond_number )
        print( 'Final Error on Norm :', err_on_norm )


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
        ending = f'CQR_{mat_lab}'
        std_print_results(max_runtimes, err_on_norm, Q_cond_number, ending)

def metrics_from_Q(Q):
    '''
    Compute the error on the norm of Q.T @ Q, as well as the condition number of Q.
    Q : np.array
        The matrix Q from the QR decomposition.
    '''
    Q_squared = Q.T @ Q
    norm_error = np.linalg.norm( Q_squared - np.eye(Q.shape[1]) )
    cond_number = np.linalg.cond(Q)
    return norm_error, cond_number

def CGS_metrics(Q):
    '''
    Compute the error on the norm of the Q[:,:j].T @ Q[:,:j] matrices to produce a sequence. 
    Q : np.array
        The matrix Q from the QR decomposition.
    '''
    if PRINT_RESULTS :
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


## should we progressively build the Q^(d) matrices multiplying each time by the M x N matrix ?
# does this mess up the order of the computations ? eg. if we the first matrix as [Q^(0),...,Q^(3)], can we still mix 0 with 2 and 1 with 3 ? 

def get_partner_idx( rank:int, log_sample:int ) -> int:
    idx = 0
    if rank % 2**(log_sample+1) == 0:
        idx = rank + 2**log_sample
    else:
        idx = rank - 2**log_sample
    return idx

def TSQR(A_l):
    A_l, mat_lab = A_l

    runtimes = np.empty(N_REPS)
    for i in range(N_REPS):
        start = time.perf_counter()
        Y_l_kp1, R_l_kp1 = np.linalg.qr(A_l, mode=MODE) 
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
                Y_l_k, R_l_k = np.linalg.qr(np.concatenate((R_l_kp1[:N,:N],R_j_kp1[:N,:N]), axis=0), mode=MODE)
                R_l_kp1 = R_l_k
                Ys.append(Y_l_k)

        Q = build_Q( Ys )
        runtimes[i] = time.perf_counter() - start

    tot_runtimes = None
    if RANK == 0:
        tot_runtimes = np.empty((N_PROCS,N_REPS), dtype=float)
    COMM.Gather(runtimes, tot_runtimes, root=0)

    if RANK == 0:
        max_runtimes = np.max(tot_runtimes, axis=0) # max over all processors
        err_on_norm, Q_cond_number = metrics_from_Q(Q)
        ending = f'TSQR_{mat_lab}'
        std_print_results(max_runtimes, err_on_norm, Q_cond_number, ending)
    return 

def TSQR_Ys(A_l):
    A_l, mat_lab = A_l
    runtimes = np.empty(N_REPS)

    Y_l_kp1_Tau_kp1 , R_l_kp1 = qr(A_l, mode='raw') 
    Ys = [Y_l_kp1_Tau_kp1[0]]
    Tau = [Y_l_kp1_Tau_kp1[1]]

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
            Y_l_kp_Tau_kp, R_l_k = qr(np.concatenate((R_l_kp1,R_j_kp1), axis=0), mode=MODE)
            R_l_kp1 = R_l_k
            Ys.append(Y_l_kp_Tau_kp[0])
            Tau.append(Y_l_kp_Tau_kp[1])

    Q = build_Q_Ys( Ys )
    if RANK == 0:
        # print % of non zero elements in Q
        print('Percentage of non zero elements in Q : ', np.count_nonzero(Q) / Q.size)
        norm_error, cond_number = metrics_from_Q(Q)
        ending = 'TSQR'

        print('Printing Results for ', ending, ':' )
        print('Final Q Condition Number : ', cond_number )
        print('Final Error on Norm : ', norm_error )
    return 

def format_sub_mat( Y_hats_k, logp:int ):
    if logp == LOGP_TOT:
        return Y_hats_k # size check 
    
    p = 2**logp
    l = int(M/p) # integer check is done before call to this function
    result = np.eye(l, dtype=float)
    s = 0.5 * l # sub block size
    assert s.is_integer()
    s = int(s) 

    if MODE == 'reduced':
        assert Y_hats_k.shape == (2*N, N) # what do we take for "irrelevant" Q part ?

        result[:N,:N] = Y_hats_k[:N,:N] # top left block
        #result[:N,s:s+N] = Y_hats_k[:N,N:] # top right block
        result[s:s+N,:N] = Y_hats_k[N:,:N] # bottom left block
        #result[s:s+N,s:s+N] = Y_hats_k[N:,N:] # bottom right block
    elif MODE == 'complete':
        #print(Y_hats_k.shape)
        assert Y_hats_k.shape == (2*N, 2*N) # what do we take for "irrelevant" Q part ?
        result[:N,:N] = Y_hats_k[:N,:N] # top left block
        result[:N,s:s+N] = Y_hats_k[:N,N:] # top right block
        result[s:s+N,:N] = Y_hats_k[N:,:N] # bottom left block
        result[s:s+N,s:s+N] = Y_hats_k[N:,N:] # bottom right block
    return result

def build_Q( Y_s ):
    #print('N_procs', N_PROCS, 'Rank', RANK)
    Q_k = None
    current_Q = None
    if RANK == 0:
        Q_k = np.empty((M, M), dtype=float)
        current_Q = np.zeros((M, N), dtype=float)
        current_Q[:N,:] = np.eye(N)
    for q in range(LOGP_TOT,-1,-1):
        # start from right-most mat, so only store 1 big mat in mem
        logp = LOGP_TOT - q

        color = 1
        if not (RANK % 2**q == 0):
            color = MPI.UNDEFINED
            new_comm = COMM.Split(color, key=RANK)
            continue
        new_comm = COMM.Split(color, key=RANK)

        p = 2**logp
        block_size = M / p
        assert block_size.is_integer()
        block_size = int(block_size)

        if new_comm is not None:
            subs_Q_k = None
            if new_comm.Get_rank() == 0:
                subs_Q_k = np.empty((p, block_size, block_size), dtype=float) 

            sub_mat = format_sub_mat(Y_s[-1], logp)
            Y_s.pop()
            #print(sub_mat.size, procs)

            if p > 1:
                new_comm.Gather(sub_mat, subs_Q_k, root=0)
            else:
                subs_Q_k = [sub_mat]

            if new_comm.Get_rank() == 0:
                Q_k = block_diag(*subs_Q_k)
                current_Q = Q_k @ current_Q
            ## note : by default, Gather will assemble the object by sorting received
            #      data by rank, so we don't need to worry about the sending order.  
    return current_Q


def build_Q_Ys( Y_s ):
    Q_k = None
    current_Q = None
    if RANK == 0:
        Q_k = np.empty((M, M), dtype=float)
        current_Q = np.zeros((M, N), dtype=float)
        current_Q[:N,:] = np.eye(N)
    for q in range(LOGP_TOT,-1,-1):
        # start from right-most mat, so only store 1 big mat in mem
        logp = LOGP_TOT - q

        color = 1
        if not (RANK % 2**q == 0):
            color = MPI.UNDEFINED
            new_comm = COMM.Split(color, key=RANK)
            continue
        new_comm = COMM.Split(color, key=RANK)

        p = 2**logp
        block_size = M / p
        assert block_size.is_integer()
        block_size = int(block_size)

        if new_comm is not None:
            subs_Q_k = None
            if new_comm.Get_rank() == 0:
                subs_Q_k = np.empty((p, block_size, block_size), dtype=float) 

            sub_mat = format_sub_mat(Y_s[-1], logp)
            Y_s.pop()
            #print(sub_mat.size, procs)

            if p > 1:
                new_comm.Gather(sub_mat, subs_Q_k, root=0)
            else:
                subs_Q_k = [sub_mat]

            if new_comm.Get_rank() == 0:
                Q_k = block_diag(*subs_Q_k)
                current_Q = Q_k @ current_Q
            ## note : by default, Gather will assemble the object by sorting received
            #      data by rank, so we don't need to worry about the sending order.  
    return current_Q

mat = None
sing_mat = None
if RANK == 0:
    mat = sparse.load_npz(f'../data/csr_{M}_by_{N}_other_mat.npz').toarray()
    sing_mat = utils.get_C(M,N)
    # import qr from scipy to check the results
    #import scipy.linalg as la
    #Q, R = np.linalg.qr(mat, mode='raw')
    #QQ, RR = np.linalg.qr(np.concatenate((R,R), axis=0), mode='raw')
    #print('Q shape', Q.shape)
    #print('R shape', R.shape)

shape = (M//N_PROCS, N)
mat_l = np.empty(shape, dtype=float)
sing_mat_l = np.empty(shape, dtype=float)
COMM.Scatter(mat, mat_l, root=0)
COMM.Scatter(sing_mat, sing_mat_l, root=0)

mat_l = [mat_l, 'mat']
sing_mat_l = [sing_mat_l, 'sing_mat']


#cholesky(mat_l)
#cholesky(sing_mat_l)
#gram_schmidt(mat_l)
gram_schmidt(sing_mat_l)
#TSQR(mat_l)
#TSQR(sing_mat_l)