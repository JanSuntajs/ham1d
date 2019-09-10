"""
A module that implements the
hamiltonian class which constructs
the model hamiltonian from a given
set of input parameters.g

"""

import numpy as np
from scipy import linalg as sla
from scipy import sparse as ssp
import functools


from . import ham_ops
from .construct_ops import operators_mixin

_ops = ham_ops.operators


class decorators_mixin(object):
    """
    A mixin class for decorators which
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

                # check if the entries in the operator descriptor list are ok
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
                                      'in the operator descriptor string!'))

                # sort sites in the site coupling list
                # coups[:, 1:] = np.sort(coups[:, 1:], axis=1)
                term[1] = coups

            res = decorated(*args)
            return res

        return wrap_check_ham_lists


class _hamiltonian(decorators_mixin, operators_mixin):
    """
    Creates a class which constructs the
    spin chain hamiltonian.

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


    """

    def __init__(self, L, static_list, dynamic_list, t=0, Nu=None):
        super(_hamiltonian, self).__init__()

        # make sure this class cannot be instantiated
        # on its own:
        if self.__class__.__name__ == '_hamiltonian':
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")

        self.L = L

        self._static_changed = False

        self.static_list = static_list
        self.dynamic_list = dynamic_list
        self.build_mat()

        self.Nu = Nu

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
    @decorators_mixin.check_ham_lists
    def static_list(self, static_list):
        """
        Perform checking on the shapes and values
        of the static_ham input nested list.

        INPUT:

        static_ham: list
            A nested list of the operator description
            strings and site-coupling lists for the
            time-independent part of the Hamiltonian.
            See class' docstring for more details.

        """

        self._static_list = static_list
        self._static_changed = True

    def build_mat(self):

        pass

    @property
    def dynamic_list(self):

        return self._dynamic_list

    @dynamic_list.setter
    @decorators_mixin.check_ham_lists
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

    def eigvals(self, *args, **kwargs):
        """
        A routine for calculating only the
        eigenvalues of the Hamiltonian array.
        """

        return sla.eigvalsh(self.mat.todense(), *args, **kwargs)

    def eigsystem(self, *args, **kwargs):
        """
        A routine for calculating both the
        eigenvalues and the eigenvectors
        of the Hamiltonian array.
        """

        return sla.eigh(self.mat.todense(), *args, **kwargs)

    # @property
    # def dynamic(self):

    #     return self._dynamic

    # @dynamic.setter
    # def dynamic(self, dynamic_ham):
