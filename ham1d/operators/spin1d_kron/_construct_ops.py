import numpy as np
from scipy import sparse as ssp

from . import _spinops

_ops = _spinops.operators


class operators_mixin(object):

    """
    A class with methods for operator construction.
    """

    def make_op(self, op_string, coupling):
        """
        A function for building an operator
        acting over a many-body Hilbert space
        where the user provides an operator string
        and the value of exchange constant and
        sites on which the single-particle operators
        act.

        Parameters
        ----------

        op_string: string
                A string describing which single-body
                operators comprise the many-body hamiltonian.
                An example:
                                op_string = 'zz'
                This would describe a two-body operator of interaction
                between two spins in the z-direction.

        coupling: list
                A list describing the strength of the coupling constant
                as well as where the coupled spins reside. The structure
                is as follows:

                            [exchange, site_1, site_2, ..., site_n]

                in the n-body coupling case.
                In an exemplary case of a 5-site chain with interaction
                J between two spins at sites 1 and 3 (following Python's
                indexing notation), one would have:

                            coupling = [J, 1, 3]

                Combining the op_string and coupling parameters together,
                the following many-body operator would be constructed:

                            J * (id2 x Sz x id2 x Sz x id2)

                Here, x denotes the tensor product of the Hilbert spaces.


        Returns
        -------

        temp * exchange: csr matrix

                A sparse matrix in the csr format -> the operator
                multiplied by the exchange constant.

        """

        # converts the operator string to a list of
        # operator string values
        op_string = list(op_string)

        # first entry of the coupling array is the
        # exchange constant, the second one is the
        # site coupling list
        exchange, sites = coupling[0], coupling[1:]

        sites = np.sort(sites)

        # in the PBC case, one needs to take care of the
        # operator ordering -> if the operators "wrap around",
        # one needs to consider this and properly reorder the
        # operator descriptor string list
        sites_sorted = np.argsort(coupling[1:])

        # determine the dimensionalities of the
        # intermediate identity operators which
        # 'act' between the spin operators at
        # specified sites
        dims = np.diff(sites) - 1
        # make sure that boundary cases are also
        # properly considered
        dims = np.insert(dims, 0, sites[0])
        dims = np.append(dims, self.L - 1 - sites[-1])
        # create the intermediate identity matrices
        eyes = [ssp.eye(2 ** dim) for dim in dims]
        # NOTE: the above construction ensures that the
        # cases where nontrivial operators are not present
        # at the edge sites are also properly considered

        temp = ssp.eye(1)  # defaults to an identity

        for i, eye in enumerate(eyes[:-1]):
            # an iterative step term -> consisting
            # of an identity matrix and an operator
            temp_ = ssp.kron(eye, _ops[op_string[sites_sorted[i]]])

            temp = ssp.kron(temp, temp_)

        # take care of the final boundary case
        temp = ssp.kron(temp, eyes[-1])

        return temp * exchange
