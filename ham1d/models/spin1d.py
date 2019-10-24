from ..operators.spin1d import _construct_ops
#from ..operators.spin1d import _spinops
from ._base_ham_cls import _hamiltonian_numba


class hamiltonian(_hamiltonian_numba):

    _ops = _construct_ops.operators
    _trans_dict = _construct_ops._trans_dict

    def __init__(self, L, static_list, dynamic_list=[], t=0, Nu=None,
                 parallel=False, mpirank=0, mpisize=0):
        super(hamiltonian, self).__init__(
            L, static_list, dynamic_list, _construct_ops, t, Nu, parallel,
            mpirank, mpisize)
