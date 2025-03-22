from numba import njit
from ..data_structures import BitField
from heapq import heapify, heappop, heappush
import numpy as np

@njit()
def check_coverage(bitfield, bits):
    l = np.zeros(bits.shape[0], dtype=np.int32)
    full_coverage = True
    pos = 0
    for i in range(bits.shape[0]):
        if not bitfield.has(bits[i]):
            l[pos] = bits[i]
            pos += 1
        else:
            full_coverage = False
    l = l[0:pos]
    return full_coverage, pos, l

@njit()
def set_cover(N, subset_array, segment_idx):
    covered = BitField(N)
    num_covered = 0
    cover = []

    Q = [(-1 * (segment_idx[i+1] - segment_idx[i]), i) for i in range(len(segment_idx))]
    heapify(Q)

    while num_covered < N:
        priority, i = heappop(Q)
        start_idx = segment_idx[i]
        S = subset_array[start_idx : start_idx+(-1 * priority)]
        full_coverage, effective_count, effective_coverage = check_coverage(covered, S)
        if full_coverage:
            cover.append(i)
            for elem in effective_coverage:
                covered._set_bit(elem)
            num_covered += effective_count
        elif effective_count:
            S[0:effective_count] = effective_coverage
            heappush(Q, (-1 * effective_count, i))
    return cover

@njit()
def partial_set_cover(N, p, subset_array, segment_idx):
    covered = BitField(N)
    num_covered = 0
    cover = []

    Q = [(-1 * (segment_idx[i+1] - segment_idx[i]), i) for i in range(len(segment_idx))]
    heapify(Q)

    limit = N*p

    while num_covered < limit:
        priority, i = heappop(Q)
        start_idx = segment_idx[i]
        S = subset_array[start_idx : start_idx+(-1 * priority)]
        full_coverage, effective_count, effective_coverage = check_coverage(covered, S)
        if full_coverage:
            cover.append(i)
            for elem in effective_coverage:
                covered._set_bit(elem)
            num_covered += effective_count
        elif effective_count:
            S[0:effective_count] = effective_coverage
            heappush(Q, (-1 * effective_count, i))
    return cover