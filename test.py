from busfactorx import Project
import numpy as np
import scipy as sp

# Generating a random bipartite graph

n = 12000          # Number of tasks
m = 50000          # Number of people
k = 10             # Average degree

rng = np.random.default_rng()
rows = np.zeros(m*k, dtype=np.uint32)
cols = np.zeros(m*k, dtype=np.uint32)

start_index = 0
for i in np.arange(m, dtype=np.uint32):
    col = np.unique(rng.integers(0, n, k, np.uint32)) # generate random task adjacencies
    end_index = start_index + col.shape[0]
    cols[start_index:end_index] = col
    rows[start_index:end_index] = i
    start_index = end_index

# Preparing data for the sparse array construction
cols = cols[0:end_index]
rows = rows[0:end_index]
data = np.ones(cols.shape, dtype=np.uint8)

B = sp.sparse.coo_array((data, (rows, cols)))
B = B.tocsr()

P = Project(B)
# Compute the MRS of the project
MRS, redundant_set = P.compute_maximum_redundant_set(p=0.5)

# Compute the MCS of the project
MCS = P.compute_minimum_critical_set(p=0.5)

# Compute the bus-factor according to the Piccolo et al. definition
piccolo_bus_factor = P.compute_bus_factor_network_robustness()

print(MRS)
print(MCS)
print(piccolo_bus_factor)