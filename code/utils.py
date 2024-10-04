import numpy as np
from scipy.io import mmread
from scipy.linalg import solve_triangular
import scipy.sparse as sparse

# Load Matrix Market format
def get_other_mat(m:int, n:int):
	if m>50000 or n>50000:
		raise ValueError('No matrix so large')
	sparse_matrix = mmread('../data/cvxbqp1.mtx')
	sparse_matrix = sparse_matrix.tocsr()
	sparse.save_npz(f'../data/sparse_{m}_by_{n}_matrix', sparse_matrix)
	return sparse_matrix

#save_resonable_mat()

def get_C( m:int, n:int ):
	# compute a 2 d grid, from 0 to 1, with m rows and n columns
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, m)
	# compute the meshgrid
    X, Y = np.meshgrid(x, y)
    return np.sin( 10 *(X+Y) ) / ( 1.1 + np.cos( 100 *(Y-X) ) )	
    # the function is symmetric


