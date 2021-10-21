from scipy import sparse as ssp

from ..operators.spin1d_kron._construct_ops import operators_mixin
from ..operators.spin1d_kron import _spinops
from ._base_ham_cls import _hamiltonian


class hamiltonian(operators_mixin, _hamiltonian):

    def __init__(self, L, static_list, dynamic_list, t=0, Nu=None, grain_list=[]):
        self._ops = _spinops.operators
        super(hamiltonian, self).__init__(
            L, static_list, dynamic_list, t, Nu, grain_list)
        self.build_mat()

    # build the hamiltonian matrix
    def build_mat(self):
        # def _ham_stat(self):
        """
        Build the entire (static) hamiltonian from the static
        list.

        The idea of this code is to build the entire
        hamiltonian as a tensor product of single spin
        operators which automatically also ensures the
        validity of periodic boundary conditions if those
        are specified. No special PBC flag is needed in
        this case, one only needs to properly format
        the couplings list.

        In case we had a Hamiltonian defined on a chain
        of length L = 5 with two spins on sites 1 and 3
        interacting via exchange interaction along the z-axis,
        we would do the following:

        Id_2 x Sz x Id_2 x Sz x Id_2

        Here x denotes the tensor product of the Hilbert
        spaces, Id_2 is the identity over a single spin
        Hilbert space and we have enumerated the states
        according to python's indexing (0, 1, ... , L - 1)

        Returns
        -------

        ham_static: dict
            A dict of key-value pairs where keys are
            the operator descriptor strings and values
            are the hamiltonian terms

        """
        # initialize an empty dict
        ham_static = {}

        if self._static_changed:
            # if the static_list has changed,
            # rebuild the static hamiltonian
            # dict.

            # iterate over different hamiltonian
            # terms in the static list
            for ham_term in self.static_list:

                static_key = ham_term[0]
                # the dimensionality of the default placeholder
                # Hamiltonian must match the Hilbert space dimension
                # which scales exponentially with system size as 2 ** L
                ham = 0 * ssp.eye(2 ** self.L)

                # coupling constants and sites
                couplings = ham_term[1]

                for coupling in couplings:

                    if static_key != 'RR':
                        ham += self.make_op(static_key, coupling)
                    else:

                        ham += self._build_rnd_grain(static_key, coupling)

                if static_key in ham_static.keys():
                    static_key = static_key + '_'
                ham_static[static_key] = ham

            self._static_changed = False
            self._mat_static = ham_static

            self._matsum()
