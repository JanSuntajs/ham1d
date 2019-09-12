"""
Implementation of fermionic operators compiled using
numba's (eager) just-in-time compilation to speed
up subsequent calculations and hamiltonian creations
procedure.

"""

import numba as nb

from ...models import bitmanip as bmp
# -----------------------------------------------------------------------------


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def cn(state, bit):
    """
    The number operator operator, in matrix form given as:

    cn = 0.5 * np.array([[1, 0], [0, -1]])

    Note that in our implementation, we write the cn
    operator in a particle-hole symmetric way, applying
    the transformation n -> n - 1/2
    """

    return 0.5 * (-1) ** bmp.getbit(state, bit), state


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def id2(state, bit):
    """
    The identity operator.
    """

    return 1., state


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def cp(state, bit):
    """
    creation operator, in matrix form given as:

    cp = np.array([[0, 1], [0, 0]])
    """

    bitval = bmp.getbit(state, bit)

    if bitval:

        return 1. - bitval, state

    else:
        return 1. - bitval, bmp.bitflip(state, bit)


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def cm(state, bit):
    """
    annihilation operator, in matrix form given as:

    cm = np.array([[0, 0], [1, 0]])
    """

    bitval = bmp.getbit(state, bit) * 1.

    if bitval:

        return bitval, bmp.bitflip(state, bit)

    else:

        return bitval, state
