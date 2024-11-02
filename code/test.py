import numpy as np

# create a random normal matrix
m = 32768
n = 330
A = np.random.randn(m, n)
# normalize the columns
A /= np.linalg.norm(A, axis=0)
I = np.eye(n)


err_on_norm = np.linalg.norm( A.T @ A - I )
print(err_on_norm)

'''

def get_partner_idx( rank:int, log_sample:int ) -> int:
    idx = 0
    if rank % 2**(log_sample+1) == 0:
        idx = rank + 2**log_sample
    else:
        idx = rank - 2**log_sample
    return idx

arange = np.arange(0, 16)
log_sample = [0,1,2,3]
partner_idx = []
for i in log_sample:
    for rank in arange:
        #print(rank % 2**i)
        if not (rank % 2**i == 0):
            continue
        partner_idx.append(get_partner_idx(rank, i))
    print(f'partner_idx={np.array(partner_idx)}')
    partner_idx = []


# Your 2x3 matrix of data
data = np.array([[4, 7, 1], [5, 2, 8]])

# Define bar width and positions
bar_width = 0.35
x = np.arange(data.shape[1])  # Positions for the groups (3 groups)

# Create the plot
fig, ax = plt.subplots()

# Plot the sub-bars
# Left bars (first row of the matrix)
ax.bar(x - bar_width/2, data[0], bar_width, label='Left Sub-Bar', color='tab:blue')

# Right bars (second row of the matrix)
ax.bar(x + bar_width/2, data[1], bar_width, label='Right Sub-Bar', color='tab:green')

# Add some labels and title
ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Grouped Bar Plot')
ax.set_xticks(x)
ax.set_xticklabels(['Group 1', 'Group 2', 'Group 3'])

# Add legend
ax.legend()

# Show the plot
plt.show()

'''
