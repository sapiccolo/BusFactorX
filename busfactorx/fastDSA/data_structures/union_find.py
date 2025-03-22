import numpy as np
from numba import njit, uint32, typeof
from numba.experimental import jitclass
from numba.typed import List, typeddict

spec = [
    ("_N", uint32),             # Number of nodes in the union-find data structure
    ("_parent", uint32[:]),     # Parent array: contains the root of each node
    ("_size", uint32[:]),       # Size of the tree in which the node belongs
    ("_count", uint32),         # Number of connected components
    ("_size_lcc", uint32)       # Size of the largest connected component
]

type_List_uint32 = typeof(List.empty_list(uint32))

@jitclass(spec)
class UnionFind:
    def __init__(self, N):
        self._N = N
        self._parent = np.arange(N, dtype=np.uint32)
        self._size = np.ones(N, dtype=np.uint32)
        self._count = N                 # Stores the number of connected components
        self._size_lcc = 1              # Stores the size of the largest connected component

    # Iterative version: numba does not support recursion! 
    def find_set(self, v:uint32):
        root = v
        # Find  the root
        while (root != self._parent[root]):
            root = self._parent[root]
        # Compress the path
        while (v != root):
            v, self._parent[v] = self._parent[v], root
        return root

    def union_sets(self, a:uint32, b:uint32):
        a = self.find_set(a)
        b = self.find_set(b)
        if a != b:
            if self._size[a] < self._size[b]:
                a, b = b, a
            self._parent[b] = a
            self._size[a] += self._size[b]
            self._count -= 1
            if self._size[a] > self._size_lcc:
                self._size_lcc = self._size[a]

    def union(self, array):    
        roots = set([self.find_set(elem) for elem in array])
        roots = np.array(list(roots), dtype=np.uint32)
        root = roots[self._size[roots].argmax()]

        self._parent[roots] = root
        self._size[root] = self._size[roots].sum()
        self._count -= roots.shape[0] - 1

        if self._size[root] > self._size_lcc:
            self._size_lcc = self._size[root]

    def to_sets(self):
        # Compress the path for each element
        for i in np.arange(self._N, dtype=np.uint32):
            self.find_set(i)

        # Build the dictionary of sets
        sets = typeddict.Dict.empty(key_type=uint32, value_type=type_List_uint32)
        for i in np.arange(self._N, dtype=np.uint32):
            if self._parent[i] not in sets:
                sets[self._parent[i]] = List.empty_list(uint32)
            sets[self._parent[i]].append(i)
        return [set(sets[key]) for key in sets]

    def count_components(self):
        return self._count

    def size_LCC(self):
        return self._size_lcc

    def connected(self, a, b):
        return self.find_set(a) == self.find_set(b)