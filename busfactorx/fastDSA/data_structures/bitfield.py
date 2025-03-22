import numpy as np
from numba import uint32, uint8, njit, vectorize
from numba.experimental import jitclass

spec = [
    ("_N", uint32),             # Number of bits
    ("_bitarray", uint32[:])    # Actual bitarray
]

@njit()
def sparse_popcount(word):
    # Counting bits set the Kernighan way, vectorized.
    # There exists a faster algorithm. 
    v = np.uint32(word)
    c = 0
    while v:
        v &= v - 1
        c += 1
    return c

# Computes the hamming weight of a word: the number of set bits in the word.
@njit()
def popcount(n: np.uint32):
    n -= (n >>1) & 0x55555555
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333)
    return uint8(((n + (n >> 4) & 0xF0F0F0F) * 0x1010101) >> 24)

@jitclass(spec)
class BitField:
    def __init__(self, num_bits):
        dim = np.uint32(np.ceil(num_bits/32))
        self._N = num_bits
        self._bitarray = np.zeros(dim, dtype=np.uint32)
    
    # Returns the value of the bit in position idx. Allows the use of the BitField as an iterator (read-only).
    # You can write things like "for bit in B: ...", or "for idx, val in enumerate(B): ..."
    def __getitem__(self, idx):
        if idx >= self._N:
            raise IndexError("Index out of range")
        return 1 if (self._bitarray[idx >> 5] & (1 << (idx & 31))) else 0

    # Returns True if the bit in position idx is set, False otherwise
    def has(self, idx):
        return(self._bitarray[idx >> 5] & (1 << (idx & 31))) != 0

    # Sets the bit in position idx to one
    def _set_bit(self, idx):
        #assert idx < self._N, "Index out of range."
        self._bitarray[idx >> 5] |= (1 << (idx & 31))

    # Sets the bit in position idx to zero
    def _clear_bit(self, idx):
        #assert idx < self._N, "Index out of range."
        self._bitarray[idx >> 5] &= ~(1 << (idx & 31))

    # Flips the bit at position idx
    def flip(self, idx):
        self._bitarray[idx >> 5] ^= (1 << (idx & 31))

    # Allows to set bits as in an array B[0] = 1, B[0] = 0
    def __setitem__(self, idx, value):
        if value not in [0, 1]:
            raise ValueError("Bits can only take values of 0 or 1.")
        if idx >= self._N:
            raise IndexError("Index out of range")
        if value:
            self._set_bit(idx)
        else:
            self._clear_bit(idx)

    # Returns the number of bits set to one in the bitfield (the cardinality of the set)
    def count_set_bits(self):
        count = 0
        for i in range(self._bitarray.shape[0]):
            count += popcount(self._bitarray[i])
        return count

    # Overloads the function len(). When invoked on a BitField (len(B)) returns the number of set bits
    def __len__(self):
        return self.count_set_bits()

    # Returns a list with the set bit locations
    def to_list(self):
        l = []
        for i in range(self._bitarray.shape[0]):
            word = np.uint32(self._bitarray[i])
            while word:
                lsb = word & -word
                l.append((i << 5) + popcount(lsb-1))
                word ^= lsb
        return l

    # Returns an array with the set bit locations
    def to_array(self):
        arr = np.zeros(self._N, dtype = np.uint32)
        pos = 0
        for i in range(self._bitarray.shape[0]):
            word = np.uint32(self._bitarray[i])
            while word:
                lsb = word & -word
                arr[pos] = ((i << 5) + popcount(lsb-1))
                word ^= lsb
                pos += 1
        return arr[0:pos]

    # Returns a generator with the set bit locations
    def set_bits_generator(self):
        for i in range(self._bitarray.shape[0]):
            word = np.uint32(self._bitarray[i])
            while word:
                lsb = word & -word
                yield((i << 5) + popcount(lsb-1))
                word ^= lsb

    # Returns a string representation of the BitField
    def to_string(self):
        return "".join([str(self[idx]) for idx in range(self._N)])

    def _last_non_empty_word(self):
        for i in range(self._bitarray.shape[0]-1, 0-1, -1):
            if self._bitarray[i]:
                return i
        return 0

    # Checks if the BitField intersects with BitField
    def intersects(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        for i in range(N):
            if (self._bitarray[i] & other_BitField._bitarray[i]) != 0:
                return True
        return False

    # Computes the intersection between this BitField and another BitField. 
    # The current BitField is modified
    def intersection(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        self._bitarray[0:N] &= other_BitField._bitarray[0:N]

    # Computes the intersection between this BitField and another BitField. 
    # Returns a new BitField
    def new_intersection(self, other_BitField):
        num_bits = min(self._N, other_BitField._N)
        answer = BitField(num_bits)
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        answer._bitarray[0:N] = self._bitarray[0:N] & other_BitField._bitarray[0:N]
        return answer

    # Computes the size of the intersection between this BitField and another BitField.
    def intersection_size(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        answer = 0
        for i in range(N):
            answer += popcount(self._bitarray[i] & other_BitField._bitarray[i])
        return answer

    # Tests whether the current BitField is equal to another
    def equals(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        for i in range(N):
            if self._bitarray[i] != other_BitField._bitarray[i]: return False
        for i in range(N, self._bitarray.shape[0]):
            if self._bitarray[i] != 0: return False
        for i in range(N, other_BitField._bitarray.shape[0]):
            if other_BitField._bitarray[i] != 0: return False
        return True

    # Computes the difference between this BitField and another BitField.
    # Modifies this BitField
    def difference(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        self._bitarray[0:N] &= ~(other_BitField._bitarray[0:N])

    # Computes the difference between this BitField and another BitField. 
    # Returns a new BitField
    def new_difference(self, other_BitField):
        answer = BitField(self._N)
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        answer._bitarray[0:N] = self._bitarray[0:N] & ~(other_BitField._bitarray[0:N])

        # The remaining bits in this BitField are part of the difference too!
        answer._bitarray[N:] = self._bitarray[N:]
        return answer

    # Computes the size of the intersection between this BitField and another BitField.
    def difference_size(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        answer = 0
        for i in range(N):
            answer += popcount(self._bitarray[i] & ~(other_BitField._bitarray[i]))

        # The remaining bits in this BitField are part of the difference too!
        for i in range(N, self._bitarray.shape[0]):
            answer += popcount(self._bitarray[i])
        return answer

    # Computes the difference between the other BitField and this one.
    # Modifies the other BitField
    def difference_right(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        other_BitField._bitarray[0:N] &= ~(self._bitarray[0:N])

    # Computes the XOR between this BitField and another BitField.
    # Modifies this BitField
    def xor(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        self._bitarray[0:N] ^= other_BitField._bitarray[0:N]

    # Computes the XOR between this BitField and another BitField. 
    # Returns a new BitField
    def new_xor(self, other_BitField):
        answer = BitField(self._N)
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        answer._bitarray[0:N] = self._bitarray[0:N] ^ other_BitField._bitarray[0:N]

        # The remaining bits in this BitField are part of the XOR too!
        answer._bitarray[N:] = self._bitarray[N:]
        return answer

    # Computes the size of the XOR between this BitField and another BitField.
    def xor_size(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        answer = 0
        for i in range(N):
            answer += popcount(self._bitarray[i] ^ other_BitField._bitarray[i])

        # The remaining bits in this BitField are part of the XOR too!
        for i in range(N, self._bitarray.shape[0]):
            answer += popcount(self._bitarray[i])
        return answer

    # Computes the union between this BitField and another BitField.
    # Modifies this BitField
    def union(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        self._bitarray[0:N] |= other_BitField._bitarray[0:N]

    # Computes the XOR between this BitField and another BitField. 
    # Returns a new BitField
    def new_union(self, other_BitField):
        answer = BitField(self._N)
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        answer._bitarray[0:N] = self._bitarray[0:N] | other_BitField._bitarray[0:N]

        # The remaining bits in this BitField are part of the XOR too!
        answer._bitarray[N:] = self._bitarray[N:]
        return answer

    # Computes the size of the XOR between this BitField and another BitField.
    def union_size(self, other_BitField):
        N = min(self._bitarray.shape[0], other_BitField._bitarray.shape[0])
        answer = 0
        for i in range(N):
            answer += popcount(self._bitarray[i] | other_BitField._bitarray[i])

        # The remaining bits in this BitField are part of the XOR too!
        for i in range(N, self._bitarray.shape[0]):
            answer += popcount(self._bitarray[i])
        return answer