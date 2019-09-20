"""
Implementation of fermionic operators compiled using
numba's (eager) just-in-time compilation to speed
up subsequent calculations and hamiltonian creations
procedure -> the operators here are defined for the
noninteracting case and should thus allow for reaching
larger systems sizes.

"""

import numba as nb
import numpy as np

# -----------------------------------------------------------------------------


signature = "Tuple((complex128, uint64[:], uint64))(uint64[:], uint32, uint64)"


@nb.njit(signature)
def cn(state, bit, ind_in):
    """
    The number operator operator, in matrix form given as:

    cn = 0.5 * np.array([[1, 0], [0, -1]])

    Note that in our implementation, we write the cn
    operator in a particle-hole symmetric way, applying
    the transformation n -> n - 1/2
    """

    bitval = state[bit]
    ind_out = ind_in

    return 0.5 * (-1) ** bitval, state, ind_out


@nb.njit(signature)
def id2(state, bit, ind_in):
    """
    The identity operator.
    """
    ind_out = ind_in
    return 1., state, ind_out


@nb.njit(signature)
def cp(state, bit, ind_in):
    """
    creation operator, in matrix form given as:

    cp = np.array([[0, 1], [0, 0]])
    """

    bitval = state[bit]
    flip = 1 - bitval
    ind_out = ind_in

    if np.sum(state) < 1:

        state[bit] = 1
        if flip:
            ind_out = bit

        return flip, state, ind_out

    else:

        return 0, state, 0


@nb.njit(signature)
def cm(state, bit, ind_in):
    """
    annihilation operator, in matrix form given as:

    cm = np.array([[0, 0], [1, 0]])
    """

    bitval = state[bit]
    ind_out = ind_in

    if np.sum(state) > 0:

        state[bit] = 0
        if bitval:
            ind_out = bit

        return bitval, state, ind_out

    else:

        return 0, state, 0
