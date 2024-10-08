import numpy as np
import matplotlib.pyplot as plt

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
