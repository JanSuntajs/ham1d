import numpy as np

from scipy.sparse import csr_matrix

from ham1d.models._base_ham_cls import _hamiltonian_numba
from .operators import ham_ops


def _make_correct_shape(param, dim, name):
    """
    Parse parameters into correct shape
    in case parameter values are different
    in different directions
    """

    if np.isscalar(param):

        param = np.array([param for i in range(dim)])
    else:
        param = np.array(param)

        if (param.shape != (dim,)):

            raise ValueError((f"{name} parameter should be "
                              "either a scalar or an iterable with length "
                              "equal to the system's dimensionality."))

class hamiltonian(_hamiltonian_numba):

    def __init__(self, L, dim, hopping, disorder,
                 pbc=True, t=0, parallel=False, mpirank=0,
                 mpisize=0):

        # set system parameters
        self.L = L
        self.dim = dim
        self.pbc = pbc
        self.Nu = None
        self._params_changed = False

        # set model parameters
        self.hopping = hopping
        self.disorder = disorder

        self._parallel = parallel
        self._mpirank = mpirank
        self._mpisize = mpisize

        self._make_basis()
        self._mpi_prepare_params()

        self.build_mat()

    @property
    def pbc(self):

        return self._pbc

    @pbc.setter
    def pbc(self, pbc):

        allowed_pbc = [-1, 0, -1]
        pbc = _make_correct_shape(pbc, self.dim, 'Pbc')

        # convert to int:
        # 1 -> pbc
        # 0 -> obc
        # -1 -> abc (anti-periodic)
        pbc = np.int32(pbc)

        if all(np.isin(pbc, allowed_pbc)):
            self._pbc = pbc
        else:
            return ValueError("Pbc setting error! Allowed values are: "
                              "1 for periodic bc, 0 for open bc and "
                              "-1 for anti periodic bc.")

    @property
    def hopping(self):

        return self._hopping

    @hopping.setter
    def hopping(self, hopping):

        # make sure the hopping param
        # is of correct shape
        hopping = _make_correct_shape(pbc, self.dim, 'Hopping')

        self._hopping = hopping
        self._params_changed = True

    @property
    def disorder(self):

        return self._disorder

    @disorder.setter
    def disorder(self, disorder):

        disorder = np.array(disorder)
        if (disorder.shape == tuple(self.L for i in range(self.dim))):

            self._disorder = disorder
            self._params_changed = True

        else:

            raise ValueError("Shape mismatch! The shape of the disorder array "
                             "should equal the shape of the system's lattice!")

    @property
    def num_states(self):
        return self.L ** self.dim

    @property
    def mat(self):

        if self._params_changed:
            self.build_mat()

        return self._mat

    def _make_basis(self):

        self.states = np.arange(self.num_states, dtype=np.uint64)
        self.nstates = len(self.states)

    def parity_shuffle(self):
        """
        IMPLEMENTATION PENDING
        """
        return None

    def build_mat(self):

        if self._params_changed:

            dimensions = [self.L for i in range(self.dim)]
            rows, cols, vals = ham_ops(dimensions, self.hopping,
                                       self.disorder, self.start_row,
                                       self.end_row, self.pbc)

            mat = csr_matrix((vals, (rows, cols)),
                             shape=(self.end_row -
                                    self.start_row, self.nstates),
                             dtype=np.float64)

            self._params_changed = False
            self._mat = mat

            self._mpi_calculate_nnz()
