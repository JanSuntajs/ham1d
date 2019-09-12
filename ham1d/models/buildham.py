import numpy as np
from scipy.sparse import csr_matrix

from . import bitmanip as bmp


def buildham(L, nup, static_list, build_mod):

    states, state_indices = bmp.select_states(np.uint32(L), np.uint32(nup))
    states = np.uint64(states)
    state_indices = np.uint64(state_indices)
    nstates = len(states)
    ham = csr_matrix((nstates, nstates), dtype=np.complex128)

    trans_dict = build_mod._trans_dict
    ham_ops = build_mod._ham_ops

    for term in static_list:

        ops = list(term[0])
        ops = np.array([trans_dict[op] for op in ops], dtype=np.uint32)

        coupsites = np.array(term[1])

        coups = np.float64(coupsites[:, 0])

        sites = np.uint32(coupsites[:, 1:])
        rows, cols, vals = ham_ops(states, state_indices, coups, sites, ops)
        ham += csr_matrix((vals, (rows, cols)), shape=(nstates, nstates))

    return ham

