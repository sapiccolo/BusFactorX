import numpy as np
from numba import njit
from .fastDSA.data_structures import UnionFind
from .fastDSA.algorithms import partial_set_cover

# Function to make the indices for a segment array 
# where the size of each segment is specified in sizes
@njit()
def make_indices(sizes):
    indices = [0]
    acc = 0
    for i in range(sizes.shape[0]):
        acc += sizes[i]
        indices.append(acc)
    return np.array(indices, dtype = np.uint32)

# A function which computes the bus-factor according to the Avelino et al. definition
@njit()
def fast_avelino_bus_factor(removal_order, crit_frac, task_degs, neighbors, indices):
    count = 0
    num_isolates = (np.where(task_degs == 0)[0]).shape[0]
    while num_isolates <= crit_frac:
        person = removal_order[count]
        for task in neighbors[indices[person]: indices[person + 1]]:
            task_degs[task] -= 1
            if not task_degs[task]:
                num_isolates += 1
        count += 1
    return count

# A function which computes the decay-curve of the taskwise largest connected component, according to the Piccolo et al. definition
@njit()
def fast_piccolo_bus_factor(removal_order, num_task, neighbors, indices):
    DSU = UnionFind(num_task)
    active_tasks = np.zeros(removal_order.shape[0] + 1, dtype=np.uint32) # one more for the initial 0.
    for j in range(removal_order.shape[0]):
        person = removal_order[j]
        tasks = neighbors[indices[person]: indices[person + 1]]
        root = tasks[0]
        for i in range(1, tasks.shape[0]):
            DSU.union_sets(root, tasks[i])
        active_tasks[j+1] = DSU.size_LCC()
    return active_tasks[::-1] # Reversed, to present it as a decay curve.

# Computes the bus-factor as the area under the decay curve, as defined by Piccolo et al.
@njit()
def normalized_piccolo_bus_factor(norm_decay_curve, num_people):
    big_sum = 0
    for i in range(0, num_people):
        big_sum += norm_decay_curve[i] + norm_decay_curve[i+1]
    return big_sum/(2*num_people - 1) # Normalised Piccolo bus-factor


class Project:
    def __init__(self, biadjacency_matrix):
        self._N, self._M = biadjacency_matrix.shape
        self._B = biadjacency_matrix        # Scipy sparse matrix

    def people_degrees(self):
        """ Returns the degree of all the people in the project """
        return self._B.sum(axis = 1)
    
    def task_degrees(self):
        """ Returns the degree of all the tasks in the project """
        return self._B.sum(axis = 0)

    def get_assigned_tasks(self, person_id):
        """
        Returns the tasks assigned to a person with the id person_id.
        If person_id is out of range, it raises an excption.
        """
        if not 0 <= person_id < self._N:
            raise IndexError("person_id out of range")
        return self._B[[person_id], :].nonzero()[1]

    def get_assigned_people(self, task_id):
        """
        Returns the people assigned to a task with the id task_id.
        If task_id is out of range, it raises an excption.
        """
        if not 0 <= task_id < self._M:
            raise IndexError("task_id out of range")
        return self._B[:, [task_id]].nonzero()[0]

    def get_people_adjacencies(self):
        """
        Returns a segment array, as a couple of arrays (indices, segments), containing the tasks assigned to each person.
        The arrays are defined as follows:
            - indices: an array of N + 1 elements containing the indices of each segment.
            - segments: an array of N segments, containing the tasks assigned to each person.

        The tasks assigned to the i-th person are found in the the array segments[indices[i]: indices[i+1]]

        That's the reason the array "indices" consists on N + 1 elements and the array segments contains "only" N segments.
        """
        neighbors = self._B.nonzero()[1]
        degrees = self._B.sum(axis=1)
        indices = make_indices(degrees)
        return indices, neighbors

    def get_tasks_adjacencies(self):
        """
        Returns a segment array, as a couple of arrays (indices, segments), containing the people assigned to each task.
        The arrays are defined as follows:
            - indices: an array of M + 1 elements containing the indices of each segment.
            - segments: an array of M segments, containing the people assigned to each task.

        The people assigned to the i-th task are found in the the array segments[indices[i]: indices[i+1]]

        That's the reason the array "indices" consists on M + 1 elements and the array segments contains "only" M segments.
        """
        neighbors = self._B.nonzero()[0]
        degrees = self._B.sum(axis=0)
        indices = make_indices(degrees)
        return indices, neighbors

    def compute_maximum_redundant_set(self, p = 0.5):
        """
        Computes the bus-factor according to the definition from Zazworka et al.
        Zazworka et al. defined the bus-factor as the minimum number of people that have knowledge of X% of tasks.
        Typically, X is set to 50%. It is assumed that a person has knowledge of a task if this person is assigned to it

        This definition of bus-factor corresponds to the partial set cover and, therefore, it is NP-Hard.
        The function computes an approximation of this NP-Hard problem with a greedy algorithm that iteratively includes in the cover 
        the person which covers the highest number of yet uncovered tasks.
        
        The function returns both the bus-factor as a number of people and the actual set of people who realise the cover.
        """
        assert 0 < p <= 1, "The parameter p must be in (0,1]"
        indices, neighbors = self.get_people_adjacencies()
        cover = partial_set_cover(self._M, p, neighbors, indices)
        cover_complement = list(set(range(self._N)) - set(cover))
        return (len(cover_complement), cover_complement)
        

    def compute_minimum_critical_set(self, p = 0.5):
        """
        Computes the bus-factor according to the definition from Avelino et al.
        Avelino et al. defined the bus-factor as the minimum number of people that needs to be removed from the project
        until there are at least X% of orphan tasks. Typically, X is set to 50%. A task is said orphan when there is no person assigned to it.

        This definition of bus-factor is an NP-Hard problem which contains max clique (or the densest subgraph).
        The function computes an approximation of this NP-Hard problem by progressively removing from the project network
        the person with the highest degree. When the number of orphan tasks is >= p*M, the procedure stops.
        
        The function returns the bus-factor as the number of people removed from the project.
        """
        assert 0 < p < 1, "The parameter p must be in (0,1)"
        crit_frac = p * self._M
        deg_tasks = self.task_degrees()
        removal_order = self.people_degrees().argsort()[::-1]
        indices, neighbors = self.get_people_adjacencies()
        bus_factor = fast_avelino_bus_factor(removal_order, crit_frac, deg_tasks, neighbors, indices)
        return bus_factor

    def compute_bus_factor_network_robustness(self):
        """
        Computes the bus-factor according to the definition from Piccolo et al.
        Piccolo et al. defined the bus-factor as the normalized area under the robustness curve of the project bipartite graph.

        This definition of bus-factor is an NP-Hard problem which contains the connected set cover.
        The function computes an approximation of this NP-Hard problem by progressively removing from the project network
        the person with the highest degree. The procedure builds a decay curve ad then computes the normalized area under said curve.
        
        The function returns the bus-factor as the normalized area under the network robustnesss decay curve.
        The value is normalized in [0, 1]. The reference for the normalization is the fully connected bipartite graph.
        As such, the Piccolo et al. Bus-factor can be interpreted as the percentage of robustness of the empirical project, compared to the fully connected one.
        """
        idx, neigs = self.get_people_adjacencies()
        removal_order = self.people_degrees().argsort() # Sorted by degree: from the lowest to the bigger
        S = fast_piccolo_bus_factor(removal_order, self._M, neigs, idx)
        S = S/self._M # Normalised decay curve
        normalized_bus_factor = normalized_piccolo_bus_factor(S, self._N)
        return normalized_bus_factor