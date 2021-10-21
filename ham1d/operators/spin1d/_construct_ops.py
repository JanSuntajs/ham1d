"""
This module provides a numba optimized routine
for creation of the hamiltonian terms in the
spin 1/2 case.

An analogous interface is defined for the fermionic
case in the ferm1d subpackage.

We wish to achieve consistency of matrix
representations for different Hamiltonian
implementations (eg. between the Numba
Hamiltonian and the Kronecker product
implementation).

Since in the spin1d implementation, the order
of construction is such that the last bit is
the most significant (i. e., it changes the
slowest) we mimic this behaviour here by adjusting
the order of multiplications in the tensor
product -> imagine we have a chain of length L:

  0  -  1  -  2  -  ...  -  i  -  i+1  -  ...  L-1

With the corresponding operators:
 A_0 -  A_1 - A_2 - ... - A_i -  A_i+1 - ... A_L-1

We would build the operator tensor product like this:

  A_L x A_{L-1} x ... x A_i x A_i-1 x A_i-2 x ... x A_1 x A_0


Also note: since the spin1d operator is constructed row-wise
(using the csr matrix format for the sparse matrix) where we
map each row into differen columns, a conjugate transpose of
the Kronecker product Hamiltonian has to taken in order
to ensure compatibility with the other variant. Since we are
typically dealing with Hamiltonian (e.g., Hermitian objects)
this shouldn't pose too much of a problem in most cases, however,
it has to be kept in mind in applications where one would want
to combine both implementations.
"""


import numpy as np
import numba as nb

from . import _spinops as so

_signature = (
    'Tuple((uint64[:], uint64[:], complex128[:]))(uint64[:], uint64[:], complex128[:], uint32[:,:], uint32[:])')


@nb.njit(_signature, fastmath=True, nogil=True, cache=True)
def _ham_ops(states, state_indices, couplings, sites, sel_opt):
    """
    Numba-optimized code for hamiltonian construction in the
    spin 1/2 case for arbitrary couplings between (arbitrarily
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

            0: S+ operator
            1: S- operator
            2: Identity operator
            3: Sx operator
            4: Sy operator
            5: Sz operator

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
        hamiltonian terms for an anisotropic XXZ
        chain with added potential disorder:

        H_xxz = J * (\sum_i (S_i^+S_{i+1}^- +
                 + S_i^- S_{i+1}^+) +
                 + delta * S_i^z S_{i+1}^z) +
                 + \sum_i h_i S_i^z

        We set L=4 with PBC and with the
        total spin projection Sz=0 (nup=2) and
        J = 1, delta = 0.55.

        First, the selection of states is
        most conveniently done as follows:

        states, state_indices = bmp.select_states(L=4, nup=2)

        We define the coupling constants:

        J = 1.
        delta = 0.55
        W = 1. # the disorder strength parameter

        Let us simulate the xy part of the anisotropic
        XXZ hamiltonian:

            couplings_xy = np.array([J * 0.5 for i in range(L)])
            sites_xy = np.array([[i, (i + 1) % L] for i in range(L) ]) #PBC
            sel_opt_xy = [0, 1]

        Note: to simulate the hermitian conjugate part, we can simply set
        sel_opt_xy = [1, 0].

        Things are very similar for the interaction along the z-axis:

            couplings_zz = np.array([J * delta for i in range(L)])
            sites_zz = np.array([[i, (i + 1) % L] for i in range(L) ]) #PBC
            sel_opt_zz = [5, 5]

        Let us now define the random fields in the random part:

            couplings_z = np.random.uniform(-W, W, size=L)
            sites_z = np.array([i for i in range(L)])
            sel_opt_z = [5]

        To construct any of the above terms, simply call:

            _ham_ops(states, state_indices, couplings_{}, sites_{},
                     sel_opt_{})

        NOTE: as it is evident from the above cases, the sites
        array's shape should be of the shape (len(couplings), len(sel_opt)).

    """

    rows = []
    cols = []
    vals = []

    # for i, state in enumerate(states):
    for i in range(len(states)):
        state = states[i]
        for j, coupling in enumerate(couplings):

            sitelist = sites[j]

            newstate = state
            factor = 1.

            for k, site in enumerate(sitelist):

                if sel_opt[k] == 0:

                    factor_, newstate = so.sp(newstate, site)

                elif sel_opt[k] == 1:

                    factor_, newstate = so.sm(newstate, site)

                elif sel_opt[k] == 2:

                    factor_, newstate = so.id2(newstate, site)

                elif sel_opt[k] == 3:

                    factor_, newstate = so.sx(newstate, site)

                elif sel_opt[k] == 4:

                    factor_, newstate = so.sy(newstate, site)

                elif sel_opt[k] == 5:

                    factor_, newstate = so.sz(newstate, site)

                factor *= factor_

            if factor:

                rows.append(i)
                cols.append(state_indices[newstate])
                vals.append(coupling * factor)

    rows = np.array(rows, dtype=np.uint64)
    cols = np.array(cols, dtype=np.uint64)
    vals = np.array(vals, dtype=np.complex128)
    return rows, cols, vals


# since numba does not support passing character
# strings as arguments to functions compiled in
# nopython mode, a _trans_dict is provided
# for conversion between string descriptions of
# operators and integers passed into _ham_ops
# functions.
_trans_dict = {'+': 0,
               '-': 1,
               'I': 2,
               'x': 3,
               'y': 4,
               'z': 5,
               'R': 6}

operators = {'+': so.sp,
             '-': so.sm,
             'I': so.id2,
             'x': so.sx,
             'y': so.sy,
             'z': so.sz,
             'R': None}
