"""
A module that contains
the implementation of the class
for user-defined hamiltonians
where the Hamiltonian matrix is defined
in advance and the user provides it
as an input to the Hamiltonian constructor
routine

"""


import numpy as np

from scipy import linalg as sla

from ._base_ham_cls import _hamiltonian_numba


class hamiltonian(_hamiltonian_numba):

    """
    A class for user-defined Hamiltonians where the
    user provides the hamiltonian matrix. 

    NOTE: for now, this is intended for calculations
    with dense matrices, so we do not use the routines
    for sparse calculations and do not intend usage of
    shift-and-invert algorithms or any similar techniques
    assuming sparsity of the Hamiltonian.

    Parameters:

    basis: ndarray, ndarray-like
    1D ndarray specifying the basis on which the
    Hamiltonian matrix acts. An empty array if the
    basis is not needed.

    mat: ndarray or csr_matrix
    Hamiltonian matrix, a ndarray if the matrix
    is dense or a csr_matrix for sparse matrices.

    params: dict
    A dictionary of Hamiltonian parameters, mostly
    for book-keeping in later analysis.

    parallel: boolean, optional
    Whether the Hamiltonian matrix is to be generated in a parallel
    distributed manner (if parallel==True) thus allowing for the
    usage of the specialized libraries for parallel computing, such
    as PETSc. Defaults to False.

    mpirank: int, optional.
        Rank of the mpi process if the Hamiltonian matrix is constructed
        in a distributed parallel manner using mpi. Defaults to 0 for
        sequential jobs.

    mpisize: int, optional.
        Size of the mpi block in case of a distributed parallel Hamiltonian
        matrix creation. Defaults to 0 for sequential jobs.  

    """

    def __init__(self, basis, mat, params,
                 parallel=False, mpirank=0, mpisize=0, dtype=np.float64):

        self._params_changed = False

        self.basis = basis
        self.params = params

        self._parallel = parallel
        self._mpirank = mpirank
        self._mpisize = mpisize
        self._dtype = dtype

        # self._mpi_prepare_params()

        self.mat = mat
        self.nstates = mat.shape[0]


    @property
    def mat(self):

        return self._mat


    @mat.setter
    def mat(self, mat):

        self._mat = mat
    

    def eigvals(self, complex=True, *args, **kwargs):
        """
        A routine for calculating only the
        eigenvalues of the Hamiltonian array.

        parameters:
        complex: boolean, optional

            Whether the hamiltonian to be diagonalised
            should be trated as complex or real-valued
            which can spare some memory.
        """
        if complex:
            return np.linalg.eigvalsh(self.mat, *args, **kwargs)
        else:
            return np.linalg.eigvalsh(np.real(self.mat),
                                      *args, **kwargs)

    def eigsystem(self, complex=True, *args, **kwargs):
        """
        A routine for calculating both the
        eigenvalues and the eigenvectors
        of the Hamiltonian array.
        """

        if complex:

            return sla.eigh(self.mat, *args, **kwargs)
        else:

            return sla.eigh(np.real(self.mat), *args, **kwargs)


    def parity_shuffle(self):
        """
        IMPLEMENTATION NOT INTENDED
        """

        return None