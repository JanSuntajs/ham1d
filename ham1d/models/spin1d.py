from numpy import uint32
import numpy as np
from scipy.sparse import csr_matrix

from ..operators.spin1d import _construct_ops
from ._base_ham_cls import _hamiltonian_numba
from ..operators.spin1d._spinops import _eval_diag_Siz


class hamiltonian(_hamiltonian_numba):

    _ops = _construct_ops.operators
    _trans_dict = _construct_ops._trans_dict

    def __init__(self, L, static_list, dynamic_list=[], grain_list=[],
                 t=0, Nu=None,
                 parallel=False, mpirank=0, mpisize=0):
        super(hamiltonian, self).__init__(
            L, static_list, dynamic_list, _construct_ops, t, Nu, grain_list,
            parallel,
            mpirank, mpisize)

    def eval_diag_Siz(self, states, site):

        return _eval_diag_Siz(self.states, states, uint32(site))

    def eval_matelts(self, states, op_descriptor):

        ops = list(op_descriptor[0])

        coupsites = np.array(op_descriptor[1])
        ops = np.array([_construct_ops._trans_dict[op] for op in ops],
                       dtype=np.uint32)

        couplings = np.complex128(coupsites[:, 0])

        sites = np.uint32(coupsites[:, 1:])
        #opvals = np.zeros(states.shape[1], dtype=np.complex128)
        rows, cols, vals = _construct_ops._eval_op(self.states, self.state_indices,
                                                   states, couplings, sites, ops)
        operator = csr_matrix((vals, (rows, cols)),
                              shape=(self.nstates, self.nstates), dtype=np.complex128)

        matelts = np.matmul(np.conj(states.T), operator @ states)                
        return matelts

    # def eval_matelts_spectral(self, energies, states, site, operator, etar, eps, split_num ):

    #     operator = _construct_ops._trans_dict[operator]

    #     _construct_ops._eval_op_spectral(
    #         self.states, self.state_indices, energies, states, site, operator, etar, eps, split_num)

    #     #return engy_diffs, matelts
