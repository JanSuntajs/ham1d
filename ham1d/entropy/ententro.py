"""
This module implements a simple class with tools
for calculating the entanglement entropy of a selected
quantum state. Currently, only the homogenous partitioning
into subblocks is implemented, but this can be changed
rather trivially since only an appropriate partitioning
function needs to be added in the partition_fun.py module.

"""


import numpy as np
from numpy import linalg

from .partition_fun import partition_dict


class Entangled(object):
    """
    The Entangled class implements
    simple routines and functionalities
    for the calculation of the entanglement
    entropy of a given quantum state
    for a chosen bipartition of the system.

    Attributes
    ----------

    state: ndarray
           1D ndarray representing the quantum
           state for which the entanglement
           entropy is to be calculated.
    L: int
       System size.
    L_sub: int
           Subsystem size.
    _partitioned: boolean
                  Whether the partitioning()
                  function which partitions the
                  system has already been called
                  or not.
    _svd_performed: boolean
                    Indicating whether the svd
                    (singular-value-decomposition)
                    has already been performed which
                    allows for the calculation of
                    the entanglement entropy.

    Methods
    -------
    partitioning(self, partition_type)
        Partitions the system into two subsystems
        according to the chosen 'partition_type'
        string.
    svd(self, full_matrices, *args, **kwargs)
        Performs the singular value decomposition of
        the partitioned system's 2D rectangular array.
    eentro(self)
        Returns the entanglement entropy of a given
        quantum state for a selected bipartition type.


    """

    def __init__(self, state, L, L_sub):
        super(Entangled, self).__init__()

        self.state = state
        self.L = L
        self.L_sub = L_sub
        self._partitioned = False
        self._svd_performed = False

    def partitioning(self, partition_type):

        self.eig_mat = partition_dict[partition_type](
            self.state, self.L, self.L_sub)
        self._partitioned = True
        self._partition_type = partition_type
        self._svd_performed = False

    def svd(self, full_matrices=False, *args, **kwargs):

        if self._partitioned:

            U, s, V = linalg.svd(
                self.eig_mat, full_matrices=full_matrices, *args, **kwargs)
            self._svd_performed = True

        else:

            print('Warning! Partitioning has not been performed yet!')
            s = None

        self._s_coeffs = s

    def eentro(self):

        if self._s_coeffs is not None:

            eentro = -np.sum(self._s_coeffs**2 *
                             np.log(self._s_coeffs**2))
        else:
            print('Warning! Entanglement entropy will be set to None!')
            eentro = None

        return eentro
