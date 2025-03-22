import numpy as np
from algorithms import set_cover, partial_set_cover
import time

n = 1200000
m = 5000000
k = 100

s = time.time()

rng = np.random.default_rng()
seg_array = np.zeros(m*k, dtype=np.uint32)
idx_segments = np.zeros(m + 1, dtype=np.uint32)

offset = np.uint32(0)
for i in range(m):
    arr = np.unique(rng.integers(0,n,k,np.uint32))
    size = np.uint64(arr.shape[0])
    idx_segments[i] = offset
    seg_array[offset: offset+size] = arr
    offset += size
idx_segments[i+1] = offset

total_dim = idx_segments[-1]

e = time.time()
print(f"Generated a random set cover instance of {n} elements and {m} subsets of {k} elements each, for a total dimension on {total_dim} elements, in {e-s} seconds")

copy_seg_array = seg_array.copy()
s = time.time()
cover = set_cover(n, copy_seg_array, idx_segments)
e = time.time()
print(f"Found a set cover of size {len(cover)} in {e-s} seconds")

copy_seg_array = seg_array.copy()
s = time.time()
p_cover = partial_set_cover(n, 0.3, copy_seg_array, idx_segments)
e = time.time()
print(f"Found a partial set cover (p = 0.3) of size {len(p_cover)} in {e-s} seconds")