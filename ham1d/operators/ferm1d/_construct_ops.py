"""
This module provides a numba optimized routine
for creation of the hamiltonian terms in the
spin 1D fermionic case.

An analogous interface is defined for the spin 1/2
case in the spin1d subpackage.

"""

import numpy as np
import numba as nb

from . import _fermops as fo
from ...models import bitmanip as bmp

signature = (
    'Tuple((uint64[:], uint64[:], complex128[:]))(uint64[:], uint64[:], complex128[:], uint32[:,:], uint32[:])')


@nb.njit(signature, fastmath=True, nogil=True, cache=True)
def _ham_ops(states, state_indices, couplings, sites, sel_opt):
    """
    Numba-optimized code for hamiltonian construction in the
    1D fermionic case for arbitrary couplings between (arbitrarily
    chosen) sites. This routine is internal and is intended
    for usage in the operators.buildham.buildham(...) function.
    The output of this function are the matrix elements
    corresponding to a single hamiltonian term in the decomposition
    H = \sum_i h_i
    where H is the entire hamiltonian and h_i are the aforementioned
    subterms.

    INPUT:

    states: np.array, dtype=np.uint64
        An array of states available to the system.
    state_indices: np.array, dtype=np.uint64
        A helper array used for finding the index
        of a state which was obtained after hamiltonian
        action on some selected initial state. Both
        states and state_indices arrays are usually
        provided as the output of the
        bitmanip.select_states(...) function.
    couplings: np.array, dtype=np.complex128
        An array of the coupling constants which
        are simply (complex) multiplicative
        prefactors for the hamiltonian matrix
        elements.
    sites: 2D np.array, dtype=np.uint32,
           shape=(len(couplings), len(sel_opt))
        An array indicating which sites the
        operators specified in the sel_opt (see below)
        and their corresponding coupling constants
        (see 'couplings' above) should couple.
    sel_opt: np.array, dtype=uint32
        An array specifying which operators constitute
        a particular hamiltonian term. Allowed entries
        in the sel_opt array are:

            0: c^+ operator
            1: c^- operator
            2: Identity operator
            3: n operator (c+c-)


    OUTPUT:

    rows, cols: np.array(s), dtype=np.uint64
        Rows and columns arrays needed in the
        construction of a sparse array.
    vals: np.array, dtype=np.complex128
        Values of the matrix elements whose
        positions are given by the entries in
        the rows and cols arrays.

    EXAMPLE:

        Here we show how to construct individual
        hamiltonian terms the 1D Anderson
        hamiltonian:

        H_a = t * (\sum_i (c_i^+c_{i+1}^- +
                 + c_{i+1}^+ c_{i+1}^-) +
                 + \sum_i h_i n_i

        We set L=4 with PBC in the half-filled
        case, nup=2. We set t=-1 and the
        disorder strength parameter W=1.

        First, the selection of states is
        most conveniently done as follows:

        states, state_indices = bmp.select_states(L=4, nup=2)

        We define the coupling constants:

        t = -1
        W = 1.

        Let us simulate hopping to the right first:

            couplings_r = np.array([t for i in range(L)])
            sites_r = np.array([[i, (i + 1) % L] for i in range(L) ]) #PBC
            sel_opt_r = [0, 1]

        Hopping to the left:

            couplings_l = np.array([t for i in range(L)])
            sites_l = np.array([[i % L, (i - 1) % L] for i in range(L) ]) #PBC
            sel_opt_l = [0, 1]

        Let us now define the random fields in the random part:

            couplings_rnd = np.random.uniform(-W, W, size=L)
            sites_rnd = np.array([i for i in range(L)])
            sel_opt_rnd = [3]

        To construct any of the above terms, simply call:

            _ham_ops(states, state_indices, couplings_{}, sites_{},
                     sel_opt_{})

        NOTE: as it is evident from the above cases, the sites
        array's shape should be of the shape (len(couplings), len(sel_opt)).

    """

    sel_opt = sel_opt[::-1]

    rows = []
    cols = []
    vals = []

    # for i, state in enumerate(states):
    for i in range(len(states)):
        state = states[i]
        for j, coupling in enumerate(couplings):

            sitelist = sites[j][::-1]

            newstate = state
            factor = 1.

            for k, site in enumerate(sitelist):

                if sel_opt[k] == 0:  # c+

                    # Consider the fermionic sign
                    ferm_sgn = (-1)**bmp.countBitsInterval(newstate,
                                                           0, site)

                    factor_, newstate = fo.cp(newstate, site)

                    factor_ *= ferm_sgn

                elif sel_opt[k] == 1:  # c-

                    # consider the fermionic sign
                    ferm_sgn = (-1)**bmp.countBitsInterval(newstate,
                                                           0, site)

                    factor_, newstate = fo.cm(newstate, site)

                    factor_ *= ferm_sgn

                elif sel_opt[k] == 2:  # identity

                    factor_, newstate = fo.id2(newstate, site)

                elif sel_opt[k] == 3:  # number operator

                    factor_, newstate = fo.cn(newstate, site)

                factor *= factor_

            if factor:

                rows.append(i)
                cols.append(state_indices[newstate])
                vals.append(coupling * factor)

    rows = np.array(rows, dtype=np.uint64)
    cols = np.array(cols, dtype=np.uint64)
    vals = np.array(vals, dtype=np.complex128)

    return rows, cols, vals


_trans_dict = {'+': 0,
               '-': 1,
               'I': 2,
               'n': 3}

operators = {'+': fo.cp,
             '-': fo.cm,
             'I': fo.id2,
             'n': fo.cn}
