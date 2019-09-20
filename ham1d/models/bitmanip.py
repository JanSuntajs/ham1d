"""
This module contains routines for bit manipulations which
are applicable to both hard-core bosonic (spin 1/2) or
fermionic systems -> to all systems with two local degrees
of freedom.

In order to speed up calculations and hamiltonian
preparation procedures, the routines are compiled
using numba jit (just-in-time) compilation in
nopython mode with explicit type signatures
wherever possible.

Routines defined here are:

countSetbits(n) - returns the number of set (nonzero)
                  bits in a state. Even though one
                  could technically use bin(n).count('1')
                  to achieve the same result, this approach
                  cannot be used in the nopython mode of
                  numba jit.

select_states(L, nup) - a routine for selecting the states
                        with an appropriate number of particles
                        or with an appropriate projection of
                        the total spin along the z-axis.

bitflip(state, bit) - a routine for flipping a bit in a state
                      at a position 'bit' into the opposite
                      state.

getbit(state, bit) - get the value of a bit in a state at a
                     position 'bit'.


"""

import numpy as np
import numba as nb

# ---------------------------------------------------------
# STATE SELECTION HELPER ROUTINES


@nb.njit("uint32(uint64)")
def countSetbits(n):
    """
    A routine that counts the
    number of set bits in a given
    number (state) -> states in our
    code are encoded as numbers where
    we make use of the mapping to the
    binarry representation.

    INPUT:

    n: uint64
        A number(state) for which the
        number of set bits is to be
        determined.

    OUTPUT:

    count: uint32
        The number of set bits.

    """

    count = np.uint32(0)

    while(n):

        #  bitwise AND -> checks
        #  the rightmost bit
        count += n & 1
        #  right shift until
        #  there is nothing left
        #  to shift
        n >>= 1

    return count


@nb.njit('Tuple((uint64[:], uint64[:]))(uint32, uint32)')
def select_states(L, nup):
    """
    A routine for selecting states from a block
    with a given number of nup spins or, in the
    fermionic language, a given number of particles.
    This is only applicable to models which
    preserve the particle number/spin projection
    along the z-axis.

    INPUT:

    L: uint32
        The length of the 1-dimensional chain.

    nup: uint32
        The number of up-spins or particles.

    OUTPUT:

    states: np.array, dtype=uint64
        An array of appropriate states selected
        from the whole 2^L dimensional Hilbert
        space of the problem
    state_indices: np.array, dtype=uint64
        A matrix containing information on how
        the selected states from the entire Hilbert
        space are ordered in the selected nup
        subspace of the Hilbert space.

    """
    states = []
    state_indices = np.zeros(1 << L, dtype=np.uint64)
    idx = 0
    for i in range(1 << L):

        if countSetbits(i) == nup:

            states.append(i)
            state_indices[i] = idx

            idx += 1

    return np.array(states, dtype=np.uint64), state_indices


@nb.njit('Tuple((uint64[:], uint64[:]))(uint32)')
def select_states_nni(L):
    """
    A routine for constructing a basis of
    the non-interacting system.
    INPUT:

    L: uint32
        The length of the 1-dimensional chain.

    nup: uint32
        The number of up-spins or particles.

    OUTPUT:

    states: np.array, dtype=uint64
        An array of appropriate states selected
        from the whole 2^L dimensional Hilbert
        space of the problem
    state_indices: np.array, dtype=uint64
        A matrix containing information on how
        the selected states from the entire Hilbert
        space are ordered in the selected nup
        subspace of the Hilbert space.
    """

    states = []
    state_indices = []
    # state_indices = np.arange(0, L, 1, dtype=np.uint64)

    for i in range(L):
        states.append(i)
        state_indices.append(i)

    return (np.array(states, dtype=np.uint64),
            np.array(state_indices, dtype=np.uint64))


#  ----------------------------------------------------------
#  BIT MANIPULATION OPTIONS
#
#  ----------------------------------------------------------


@nb.njit("uint64(uint64, uint32)")
def bitflip(state, bit):
    """
    A routine for flipping a bit
    on the position 'bit'.

    INPUT:

    state: uint64
        The state to be manipulated
    bit: uint32
        Bit to be flipped.

    OUTPUT:

    flipped: uint64
        The state with a flipped bit.
    """
    flipped = np.uint64(state ^ (1 << bit))
    return flipped


@nb.njit("uint8(uint64,uint32)")
def getbit(state, bit):
    """
    A routine for extracting the value
    of a selected bit in a state.

    INPUT:

    state: uint64
        The state to be manipulated
    bit: uint32
        Bit whose value is to be
        determined.

    OUTPUT:

    bitval: uint8
        The state with a flipped bit.

    """

    bitval = np.uint8(((state >> (bit)) & 1))
    return bitval


@nb.njit("uint32(uint64, int32, int32)")
def countBitsInterval(n, right, left):
    """
    Determine the number of set bits in
    an interval between r (right) and
    l (left) bit for the state n:

    [r, l)

    INPUT:

    state: uint64
        The state to be manipulated.
    right: uint32
        Rightmost bit in the interval.
    left: uint32
        Leftmost bit in the interval.

    OUTPUT:

    count: the number of set bits in
           the selected interval.

    Example: 0 0 1 | 1 0 1 | 0 1
             7 6 5   4 3 2   1 0
           r = 2
           l = 5
           n = 53
           output should  be 2.

    """

    count = np.uint32(0)

    if left <= right:

        count = count

    else:

        for i in range(right, left):

            count += getbit(n, i)

    return count
