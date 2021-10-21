"""
This module contains tools for the
creation of spin operators acting
on the chosen Hilbert space. The
operators are written in the site-occupational
basis where the last (i.e. leftmost)
bit is the most significant
(changes the slowest) and hence determines
the block structure of the operator matrices.

NOTE: THE LOCAL BASIS HERE IS:
I 1 >, | 0 >
In this basis, the Sz operator
has the following structure:
| 1  0 |
| 0 -1 |

We wish to achieve consistency of matrix
representations for different Hamiltonian
implementations (eg. between the Numba
Hamiltonian and the Kronecker product
implementation).

Since in the spin1d implementation, the order
of construction is such that the last bit is
the most significant (i. e., it changes the
slowest) we mimic this behaviour here by adjusting
the order of multiplications in the tensor
product -> imagine we have a chain of length L:

  0  -  1  -  2  -  ...  -  i  -  i+1  -  ...  L-1

With the corresponding operators:
 A_0 -  A_1 - A_2 - ... - A_i -  A_i+1 - ... A_L-1

We would build the operator tensor product like this:

  A_L x A_{L-1} x ... x A_i x A_i-1 x A_i-2 x ... x A_1 x A_0


Also note: since the spin1d operator is constructed row-wise
(using the csr matrix format for the sparse matrix) where we
map each row into differen columns, a conjugate transpose of
the Kronecker product Hamiltonian has to taken in order
to ensure compatibility with the other variant. Since we are
typically dealing with Hamiltonian (e.g., Hermitian objects)
this shouldn't pose too much of a problem in most cases, however,
it has to be kept in mind in applications where one would want
to combine both implementations.

"""


import numpy as np
import sys
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

                NOTE: 


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
            temp_ = ssp.kron(_ops[op_string[sites_sorted[i]]], eye)

            temp = ssp.kron(temp_, temp)

        # take care of the final boundary case
        temp = ssp.kron(eyes[-1], temp)

        return temp * exchange

