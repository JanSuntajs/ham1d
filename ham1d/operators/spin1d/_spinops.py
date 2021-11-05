"""
Implementation of spin 1/2 operators compiled using
numba's (eager) just-in-time compilation to speed
up subsequent calculations and hamiltonian creations
procedure.
"""

import numba as nb  
import numpy as np

from ...models import bitmanip as bmp

# -----------------------------------------------------------------------------


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def sx(state, bit):
    """
    sx operator, in matrix form given as:

    sx = 0.5 * np.array([[0, 1], [1, 0]])
    """

    return 0.5, bmp.bitflip(state, bit)


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def sy(state, bit):
    """
    sy operator, in matrix form given as:

    sy = 0.5 * np.array([[0, 1j], [-
    1j, 0]])
    """

    factor = np.complex128(1j)
    factor = 0.5j * (-1)**(1 - bmp.getbit(state, bit))

    return factor, bmp.bitflip(state, bit)


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def sz(state, bit):
    """
    sz operator, in matrix form given as:

    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.int8)
    """

    return 0.5 * (-1)**(1 - bmp.getbit(state, bit)), state


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def id2(state, bit):
    """
    The identity operator.
    """

    return 1, state


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def sp(state, bit):
    """
    s+ operator, in matrix form given as:

    sp = np.array([[0, 1], [0, 0]])
    """

    bitval = bmp.getbit(state, bit)

    if bitval:

        return 1 - bitval, state

    else:
        return 1 - bitval, bmp.bitflip(state, bit)


@nb.njit("Tuple((complex128, uint64))(uint64, uint32)")
def sm(state, bit):
    """
    s- operator, in matrix form given as:

    sm = np.array([[0, 0], [1, 0]])
    """

    bitval = bmp.getbit(state, bit)

    if bitval:

        return bitval, bmp.bitflip(state, bit)

    else:
    
        return bitval, state

        


@nb.njit("complex128[:](uint64[:], complex128[:, :], uint32)")
def _eval_diag_Siz(basis, states, site):
    """
    Evaluate the value of the local
    Sz operator at a site i for a state
    written in the site ocupational
    basis. We typically calculate
    the quantity:

    < alpha | S^z_i | alpha>

    Where |alpha > is some state written
    in the site-ocupational basis
    (typically the energy eigenstate)
    and S^z_i is the local S^z operator
    acting on site i.

    Parameters:

    basis: 1d ndarray, dtype uint64
    An array of basis states in which the states
    are written

    states: 2D ndarray, dtype complex128
    A complex ndarray of states for which
    Siz is being evaluated

    site: uint32
    Site at which the values are evaluated.

    Returns:

    szvals: 1d ndarray, dtype complex128
    Values of sz for selected states.

    """
    szvals = np.zeros(states.shape[1], dtype=np.complex128)
    for i in range(states.shape[1]):
        for j in range(basis.shape[0]):
            szvals[i] += sz(basis[j], site)[0] * np.abs(states[j][i])**2

    return szvals
