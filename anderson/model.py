import numpy as np

from scipy.sparse import csr_matrix

from ham1d.models._base_ham_cls import _hamiltonian_numba
from .operators import ham_ops


class hamiltonian(_hamiltonian_numba):

    def __init__(self, L, dim, hopping, disorder,
                 pbc=True, t=0, parallel=False, mpirank=0,
                 mpisize=0):

        # set system parameters
        self.L = L
        self.dim = dim
        self.pbc = pbc
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
    def hopping(self):

        return self._hopping

    @hopping.setter
    def hopping(self, hopping):

        # make sure the hopping param
        # is of correct shape
        if np.isscalar(hopping):

            hopping = np.array([hopping for i in range(self.dim)])
        else:
            hopping = np.array(hopping)

            if (hopping.shape != (self.dim,)):

                raise ValueError(("Hopping parameter should be "
                                  "either a scalar or an iterable with length "
                                  "equal to the system's dimensionality."))

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
