"""
A module that contains the implementation
of the classes for hamiltonian construction
as well as the implementation of a mixin
class with the definitions of the decorators
which check the correctness of the input
parameters and raise errors and issue
relevant error messages.

Contents of this module:

class _decorator__mixin(object)

    Class defining decorators used mostly
    for input checking.

class _hamiltonian(_decorator_mixin)

    Base class for all the hamiltonian
    classes used in this package.

class _hamiltonian_numba(_hamiltonian)

    Child class of the _hamiltonian class
    which implements functionalities
    specifically needed in cases when
    the Hamiltonains are generated using
    the numba jit-compiled code
    (see the ferm1d.py and spin1d.py modules
    in the models subpackage).

"""

import numpy as np
import functools
import numba as nb
from scipy import linalg as sla
from scipy.sparse import csr_matrix
from scipy import sparse as ssp

from . import bitmanip as bmp
from .buildham import rnd_mat


class _decorators_mixin(object):

    """
    A class for decorators which
    are mainly used for input checking.

    """

    @classmethod
    def check_ham_lists(cls, decorated):
        """
        A decorator for checking whether
        the static and dynamic lists provided
        are of the proper shape and if the
        input is ok.

        """
        @functools.wraps(decorated)
        def wrap_check_ham_lists(*args):
            _ops = args[0]._ops
            ham_list = args[1]
            for term in ham_list:

                # operator descriptions and coupling lists
                op_desc, coups = term
                # check if the operator descriptors are ok
                if not isinstance(op_desc, str):
                    raise TypeError(('Operator descriptor {} should'
                                     'be a string!').format(op_desc))

                # correct all preceeding/trailing whitespaces, if needed
                term[0] = op_desc.strip('')

                # number of interacting spins in the hamiltonian term
                n_inter = len(list(term[0]))

                # check if the entries in the operator descriptor list are
                # ok
                if not set(list(op_desc)).issubset(_ops.keys()):
                    raise ValueError(('Operator descriptor {}'
                                      ' contains invalid entries.'
                                      ' Allowed values: {}'
                                      ).format(op_desc, list(_ops.keys())))

                coups = np.array(coups)

                if coups[:, 1:].shape[1] != n_inter:
                    raise ValueError(('Number of sites in '
                                      'the site-coupling list '
                                      'should match the number of terms '
                                      'in the operator descriptor string!'
                                      ))

                term[1] = coups

            res = decorated(*args)
            return res

        return wrap_check_ham_lists


class _hamiltonian(_decorators_mixin):

    """
    Creates a class which constructs the
    spin chain hamiltonian.

    Attributes
    ----------




    Methods
    -------

    """

    # _ops = {}

    # is this a free (noninteracting) case?
    # defaults to False, we only toggle this
    # to True in the free1d case implementation
    # for noninteracting systems.
    _free = False

    def __init__(self, L, static_list, dynamic_list=[], t=0, Nu=None,
                 grain_list=[]):
        super(_hamiltonian, self).__init__()
        """
        Parameters
        ----------

        L: int
            An integer specifying the spin chain length.

        static_list: list
            A nested list of the operator description strings
            and site-coupling lists for the time-independent
            part of the Hamiltonian. An example of the
            static_ham list would be:

                static_ham = [['zz', J_zz]]

            Here, 'zz' is the operator descriptor string specifiying
            2-spin interaction along the z-axis direction. For a chain-
            of L sites with constant nearest-neighbour exchange
            J and PBC, the site coupling list would be given by:

                J_zz = [[J, i, (i+1)%L] for i in range(L)]

            In the upper expression, J is the term describing
            the interaction strength and the following entries
            in the inner list specify the positions of the coupled
            sites. The upper example should serve as a general
            template which should allow for simple extension to
            the case of n-spin interaction and varying couplings.
            See also the description of grain_list parameter in case
            the Hamiltonian contains random inclusions. In that case,
            the operator string has to be 'RR' and the site-coupling
            list should be of length 2.

        dynamic_list: list. optional
            A nested list of the operator description strings. The
            description is similar to the static_ham description,
            however, additional terms are needed to incorporate
            the time dependence: an example would be:

                dynamic_ham = [['zz', J_zz, time_fun, time_fun_args]]

            Here, 'zz' is the same operator descriptor string as the
            one in the static case. J_zz, however, now refers to
            the initial (dynamic) site coupling list at t=0. time_fun
            is a function object describing a protocol for
            time-dependent part of the hamiltonian with the following
            interface: time_fun(t, *time_fun_args) where time_fun_args
            are the possible additional arguments of the time-dependence.

        Nu: {int, None}, optional
            Number of up spins, relevant for the hamiltonians where the
            total spin z projection is a conserved quantity. Defaults to
            None.

        grain_list: list, optional
            In case the Hamiltonian contains random inclusions (or some
            other rather structureless blocks that can be conveniently
            described by means of a tensor product of single-body spaces),
            the grain_list contains the random matrices defined on the
            "ergodic subsets" of the whole Hamiltonian chain. The order
            and the dimensions of the matrices should be compatible with
            the entries in the static list having the 'RR' operator string.
            Note: this will only work with Nu=None, e.g. for Hamiltonians
            with no number conservation.
        """
        # make sure this class cannot be instantiated
        # on its own:
        if self.__class__.__name__ == '_hamiltonian':
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")

        self.L = L
        self._static_changed = False
        self.static_list = static_list
        self.dynamic_list = dynamic_list
        self.Nu = Nu
        self.grain_list = grain_list
        # self.build_mat()

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, L):
        if type(L) is not int:
            raise TypeError("L must be an integer!")
        if L <= 0:
            raise ValueError("L must be greater than zero!")

        self._L = L

    @property
    def num_states(self):

        return 1 << self.L

    @property
    def Nu(self):

        return self._Nu

    @Nu.setter
    def Nu(self, Nu):
        if Nu is None:
            Nu_list = None
        elif type(Nu) is int:
            Nu_list = [Nu]
        else:
            try:
                Nu_list = list(Nu)
            except TypeError:
                raise TypeError(("Nf must be an"
                                 "iterable returning integers!"))

            if any((type(Nu) is not int) for Nu in Nu_list):
                raise TypeError("Nf must be iterable returning integers")
            if any(((Nu < 0) or (Nu >= self.L)) for Nu in Nu_list):
                raise ValueError("Nf cannot be greater than L or smaller than "
                                 "zero!")
            if len(Nu_list) != len(set(Nu_list)):
                raise ValueError(("There must be no duplicates"
                                  " in a list of Nf values!"))
        self._Nu = Nu_list

    @property
    def static_list(self):

        return self._static_list

    @static_list.setter
    @_decorators_mixin.check_ham_lists
    def static_list(self, static_list):
        """
        Perform checking on the shapes and values
        of the static_list input nested list.

        INPUT:

        static_ham: list
            A nested list of the operator description
            strings and site-coupling lists for the
            time-independent part of the Hamiltonian.
            See class' docstring for more details.

        """

        self._static_list = static_list
        self._static_changed = True

    @property
    def grain_list(self):

        return self._grain_list

    @grain_list.setter
    def grain_list(self, grain_list):
        """
        Perform checking on the shapes
        and values of the grain_list input
        nested list.
        """
        # if Nu is not None, grain list must
        # be empty

        grain_present = grain_list != []

        if ((self.Nu is not None) and grain_present):
            raise ValueError(("If Nu is not None "
                              "grain_list must be empty!"))

        # check if random grain terms are in the static list
        grain_terms = [term[1] for
                       term in self.static_list if term[0] == 'RR']

        grain_specified = grain_terms != []

        conditions = []
        if (grain_present and grain_specified):
            grain_term = grain_terms[0]

            for i, grain in enumerate(grain_term):

                grain_size = np.diff(grain[1:])[0] + 1

                conditions.append(grain_list[i].shape == (2**grain_size,
                                                          2**grain_size))
        if all(conditions):
            print('Setting the grain list.')
            self._grain_list = grain_list
        else:
            err = ('There was a shape mismatch when '
                   'setting the grain list. See how the '
                   f'shapes matched: {conditions}')
            raise ValueError(err)

    def build_mat(self):

        pass

    @property
    def dynamic_list(self):

        return self._dynamic_list

    @dynamic_list.setter
    @_decorators_mixin.check_ham_lists
    def dynamic_list(self, dynamic_list):
        """
        Perform checking on the shapes and values
        of the dynamic_ham input nested list.

        INPUT:

        dynamic_ham: list
            A nested list of the operator description
            strings and site-coupling lists for the
            time-independent part of the Hamiltonian.
            See class' docstring for more details.

        """
        self._dynamic_list = dynamic_list

    def _matsum(self):
        """
        Construct the hamiltonian
        matrix -> sum the entries
        in the _ham_stat dict.

        """
        print('Please wait, building the Hamiltonian ...')
        mat = 0

        for value in self._mat_static.values():

            mat += value

        print('Building the Hamiltonian finished!')

        self._mat = mat
        # return mat

    @property
    def mat(self):

        if self._static_changed:
            self.build_mat()

        return self._mat

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
            return np.linalg.eigvalsh(self.mat.todense(), *args, **kwargs)
        else:
            return np.linalg.eigvalsh(np.real(self.mat).todense(),
                                      *args, **kwargs)

    def eigsystem(self, complex=True, *args, **kwargs):
        """
        A routine for calculating both the
        eigenvalues and the eigenvectors
        of the Hamiltonian array.
        """


        if complex:

            return sla.eigh(self.mat.todense(), *args, **kwargs)
        else:

            return sla.eigh(np.real(self.mat.todense()), *args, **kwargs)

    # @property
    # def dynamic(self):

    #     return self._dynamic

    # @dynamic.setter
    # def dynamic(self, dynamic_ham

    def _build_rnd_grain(self, op_string, coupling, grain_index):
        """
        Constructs the Hamiltonian in the
        special case in which Hamiltonian (or,
        more likely, some part of it) is modeled
        as a random matrix. For now, we only allow
        for homogenous blocks useful in simulations
        of "ergodic grains".

        Parameters:

        op_string: string
                A string describing which single-body
                operators comprise the many-body
                hamiltonian. For the random grain
                construction, we only allow for 'RR'
                operator string, standing for the
                random matrix. Two 'R's need to be
                given because the operator string
                length should match the length of
                the site coupling list. Hence,
                only 
                    op_string = 'RR'
                is allowed in this routine by design.

        coupling: list
                A list describing the strength of the
                coupling constant (actually, the magnitude
                of the typical random matrix elements)
                as well as the range of sites on which
                the random grain is defined. So, for
                a random grain defined between sites
                1 and 3 with the multiplicative
                prefactor of \beta, one would specify:

                coupling = [\beta, 1, 3]

        grain_index: int
                Integer specifying which term from the
                self.grain_list object corresponds to the
                grain term being constructed.

        """

        #op_string = list(op_string)

        # sanity checks
        if op_string != 'RR':
            print(('If you wish to specify '
                   'the random grain hamiltonian '
                   'specify the operator string as RR '
                   'Now you '
                   f'have op_string={op_string}. Exiting.'))
            sys.exit()

        # first entry of the coupling array is the
        # exchange constant, the second one is the
        # site coupling list. In this case, length
        # of the coupling list should be equal to 2,
        # their values specifying the grain boundaries.
        exchange, sites = coupling[0], coupling[1:]

        # this will define the size of the
        # grain matrix

        dims = np.diff(sites) + 1

        # rnd_mat(np.int(2**dims[0]))
        rnd_matrix = self.grain_list[grain_index]
        # this takes care of the identity
        # part preceeding the grain
        dims = np.insert(dims, 0, sites[0])
        # this takes care of the identity part
        # following the grain
        dims = np.append(dims, self.L - 1 - sites[-1])
        # construct the matrices
        id_before = sites[0]
        id_after = self.L - 1 - sites[-1]
        matrices = [ssp.eye(2**id_before), rnd_matrix, ssp.eye(2**id_after)]

        temp = ssp.eye(1)

        for mat in matrices:

            temp = ssp.kron(mat, temp)

        return temp * exchange


class _hamiltonian_numba(_hamiltonian):
    """
    A child class of the _hamiltonian class which is
    intented for calculations in both the fermionic
    and spin 1/2 cases using the numba jit-compiled
    code. Note that the examples below are given
    for the spin 1/2 case.

    Attributes
    ----------


    Methods
    -------


    """

    def __init__(self, L, static_list, dynamic_list, build_mod, t=0, Nu=None,
                 grain_list=[], parallel=False, mpirank=0, mpisize=0, dtype=np.complex128):
        """
        Parameters:
        -----------
        L: int
            An integer specifying the spin chain length.

        static_list: list
            A nested list of the operator description strings
            and site-coupling lists for the time-independent
            part of the Hamiltonian. An example of the
            static_ham list would be:

                static_ham = [['zz', J_zz]]

            Here, 'zz' is the operator descriptor string specifiying
            2-spin interaction along the z-axis direction. For a chain-
            of L sites with constant nearest-neighbour exchange
            J and PBC, the site coupling list would be given by:

                J_zz = [[J, i, (i+1)%L] for i in range(L)]

            In the upper expression, J is the term describing
            the interaction strength and the following entries
            in the inner list specify the positions of the coupled
            sites. The upper example should serve as a general
            template which should allow for simple extension to
            the case of n-spin interaction and varying couplings.

        dynamic_list: list
            A nested list of the operator description strings. The
            description is similar to the static_ham description,
            however, additional terms are needed to incorporate
            the time dependence: an example would be:

                dynamic_ham = [['zz', J_zz, time_fun, time_fun_args]]

            Here, 'zz' is the same operator descriptor string as the
            one in the static case. J_zz, however, now refers to
            the initial (dynamic) site coupling list at t=0. time_fun
            is a function object describing a protocol for
            time-dependent part of the hamiltonian with the following
            interface: time_fun(t, *time_fun_args) where time_fun_args
            are the possible additional arguments of the time-dependence.

        Nu: {int, None}
            Number of up spins, relevant for the hamiltonians where the
            total spin z projection is a conserved quantity. Defaults to
            None.

        build_mod: python module
            Python module which should have the following attributes:

                build_mod._ham_ops:

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

        self._ham_ops = build_mod._ham_ops
        self._trans_dict = build_mod._trans_dict
        self._ops = build_mod.operators

        super(_hamiltonian_numba, self).__init__(
            L, static_list, dynamic_list, t, Nu, grain_list)

        # mpi params, if needed
        self._mpisize = mpisize
        self._mpirank = mpirank
        self._parallel = parallel

        self._dtype = dtype

        self._make_basis()
        self._mpi_prepare_params()

        self.build_mat()

    def _make_basis(self):
        """
        A routine for preparing the basis states with which
        we work.

        """

        # case for noninteracting systems is different
        if not self._free:
            # if Nu is None, all states are considered, not
            # just a particular block.
            if self.Nu is not None:

                states = []
                state_indices = []
                for nu in self.Nu:
                    states_, state_indices_ = bmp.select_states(
                        np.uint32(self.L), np.uint32(nu))
                    states = np.append(states, states_)
                    state_indices = np.append(
                        state_indices, state_indices_)

            else:
                states = state_indices = [i for i in range(1 << self.L)]

            self.states = np.array(states, dtype=np.uint64)
            self.state_indices = np.array(state_indices, dtype=np.uint64)

        else:

            # in the noninteracting case, select states using
            # a different routine
            self.states, self.state_indices = bmp.select_states_nni(
                np.uint32(self.L))

        self.nstates = len(self.states)

    def parity_shuffle(self):
        """
        A routine for obtaining the basis state indices
        after the parity operator has acted on the basis.
        The parity operator is a mapping:
        x -> -x
        or, schematically for some representative state:
        1 0 1 0 -> 0 1 0 1

        Returns
        -------

        parity_indices: ndarray, dtype=np.uint64
                        An array of indices which corresponds
                        to the indices of the states which we
                        obtained after the basis has been
                        acted on by the parity operator. We can
                        then use the parity_indices array in order
                        to reshuffle an arbitrary vector thereby
                        simulating the action of the parity operator
                        on the state. The shuffling of the vector
                        is done like this:
                        shuffled_state = state[parity_indices],
                        given that state is stored as a numpy array.
        """
        if not self._free:
            parity_indices = bmp.get_parity_indices(
                self.states, self.state_indices, np.uint32(self.L))

        else:
            print(('Parity symmetry not yet '
                   'implemented for the free model!'))
            return

        return parity_indices

    def particle_hole_shuffle(self):
        """
        A routine for obtaining the basis state indices
        after the particle-hole reversal operator has acted on the basis.
        By particle-hole reversal we refer to switching holes
        (down spins) with the particles (up spins) and vice versa.
        An example:

        0 1 1 0 1 - (particle-hole reversal) -> 1 0 0 1 0

        Returns
        -------

        parity_indices: ndarray, dtype=np.uint64
                        An array of indices which corresponds
                        to the indices of the states which we
                        obtained after the basis has been
                        acted on by the parity operator. We can
                        then use the parity_indices array in order
                        to reshuffle an arbitrary vector thereby
                        simulating the action of the parity operator
                        on the state. The shuffling of the vector
                        is done like this:
                        shuffled_state = state[parity_indices],
                        given that state is stored as a numpy array.
        """
        if not self._free:

            # if a special sector of Nu is considered
            # we need to make sure the switched states
            # are also inside our Hilbert space
            if self.Nu is not None:
                # check that both nu and L-nu are present
                condition = all([(self.L - nu in self.Nu) for nu in self.Nu])

                if not condition:
                    print(('particle_hole_shuffle info: '
                           'Both Nu and L-Nu need to be present. '
                           'Particle-hole symmetry will not be tested.'))
                    return

            parity_indices = bmp.get_particle_hole_indices(
                self.states, self.state_indices, np.uint32(self.L))

        else:
            print(('particle_hole_shuffle info: '
                   'Particle-hole symmetry not yet '
                   'implemented for the free model!'))
            return

        return parity_indices

    def build_mat(self):
        """
        A routine for building the whole hamiltonian
        matrix from individual hamiltonian terms.


        """

        # check if the flag indicating the change of the
        # self.static_list has changed and hence the
        # hamiltonian matrix has to be rebuilt

        if self._static_changed:

            ham_static = {
                static_key[0]:
                csr_matrix((self.end_row - self.start_row,
                            self.nstates),
                           dtype=self._dtype)
                for static_key in self.static_list}

            for term in self.static_list:

                static_key = term[0]
                ops = list(static_key)
                ops = np.array([self._trans_dict[op] for op in ops],
                               dtype=np.uint32)

                coupsites = np.array(term[1])
                coups = np.complex128(coupsites[:, 0])

                sites = np.uint32(coupsites[:, 1:])

                if static_key != 'RR':
                    rows, cols, vals = self._ham_ops(
                        self.states[self.start_row:self.end_row],
                        self.state_indices, coups, sites, ops)

                    mat = csr_matrix((vals, (rows, cols)),
                                     shape=(self.end_row - self.start_row,
                                            self.nstates),
                                     dtype=self._dtype)
                    # NOTE: this step assigns all the terms corresponding
                    # to the same operator descriptor string to the same
                    # key in the ham_static dictionary. In case one for
                    # instance has different terms corresponding to the
                    # '+-' operator descriptor, those would all be writen
                    # to the same csr matrix. Multiple occurences of the
                    # same operator descriptor can happen, for example when
                    # one wishes to describe nearest and next-nearest
                    # neighbour couplings in the same hamiltonian.
                    ham_static[static_key] += mat
                else:

                    for i, coupling in enumerate(term[1]):

                        mat = self._build_rnd_grain(
                            static_key, coupling, i)
                        # NOTE: mpi structure will not work here
                        # mat = csr_matrix((vals, (rows, cols)),
                        #                  shape=(self.end_row - self.start_row,
                        #                         self.nstates),
                        #                  dtype=np.complex128)
                        mat = np.conj(mat.T)
                        # print(mat.dtype)
                        ham_static[static_key] += mat

            self._static_changed = False
            self._mat_static = ham_static
            self._matsum()

            # if self._parallel:

            self._mpi_calculate_nnz()

    # mpi preparation routines

    def _mpi_prepare_params(self):
        """
        A routine for preparing mpi parameters for creation of a parallel
        distributed array - over which states to iterate during the matrix
        creation.

        Creates attributes:

        self.start_row, self.end_row -> the starting and ending row/state
        during the iteration over states in the Hamiltonian array creation.

        In case self._parallel attribute of the class instance is set to False,
        this code does nothing, except that it sets the ending and starting
        state equal to the first and the last basis state.

        """

        if self._parallel:
            print('Preparing mpi parameters!')
            # distribute evenly
            local_block_sizes = np.ones(
                self._mpisize, dtype=np.int64) * np.int64(
                self.nstates / self._mpisize)
            # correct if the size of the basis is not divisible by mpisize
            for i in range(0, self.nstates % self._mpisize):
                local_block_sizes[i] += 1

            start_row = 0
            end_row = 0

            for i in range(0, self._mpirank):
                start_row += local_block_sizes[i]

            end_row = start_row + local_block_sizes[self._mpirank]

            self.start_row = start_row
            self.end_row = end_row

            print('Preparing mpi parameters finished!')
        else:
            self.start_row = 0
            self.end_row = self.nstates

    def _mpi_calculate_nnz(self):
        """
        Calculates the total number of nonzero
        matrix elements in a given matrix block
        as well as numbers of nonzero matrix
        elements in the corresponding diagonal
        and offdiagonal matrix blocks. Those
        data are needed for parallel creation
        of Hamiltonian matrices.

        """
        # nonzero elements
        print('Calculating nnz, o_nnz, d_nnz!')
        nnz = self.mat.nnz

        cols = self.mat.indices
        # nonzero matrix elements for each row
        indptr = self.mat.indptr
        d_nnz = np.diff(indptr)
        o_nnz = np.zeros_like(d_nnz)

        o_nnz, d_nnz = self._sparse_calc(indptr, cols, self.start_row,
                                         self.end_row, d_nnz)

        self._nnz = nnz
        self._o_nnz = o_nnz
        self._d_nnz = d_nnz
        print('Calculating nnz, o_nnz, d_nnz finished!')

    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True)
    def _sparse_calc(indptr, cols, start_row, end_row, d_nnz):

        o_nnz = np.zeros_like(d_nnz)
        low = 0
        for i, ind in enumerate(indptr[1:]):

            for j in range(low, ind):

                if (cols[j] < start_row or cols[j] > end_row):

                    d_nnz[i] -= 1
                    o_nnz[i] += 1

            low = ind

        return o_nnz, d_nnz
