from numpy import uint32
import numpy as np

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

    def eval_matelts(self, states, site, operator):

        operator = _construct_ops._trans_dict[operator]
        #opvals = np.zeros(states.shape[1], dtype=np.complex128)
        return _construct_ops._eval_op(self.states, self.state_indices,
                                       states, site, operator)