"""
A module with definitions of
the single spin hamiltonian
operators used in 1D spin chain
calculations for the general
case.

NOTE: THE LOCAL BASIS HERE IS:
I 0 >, | 1 >
In this basis, the Sz operator
in the S = 1 / 2 case
has the following structure:
| -1  0 | 
| 0 1 |

For S = 1, we have:
| -1 >, | 0 > , | 1 >
and the following structure
for the Sz operator:
| -1 0 0 |
| 0  0 0 |
| 0  0 1 |
"""

import numpy as np

from numpy import random

# OPERATOR DEFINITIONS
def _spin_projections(spin):
    """
    Get all the possible m_z projections
    of the spin along the z-axis.
    
    """
    return np.arange(-spin, spin + 1, 1)


# create the raising and lowering operators
def sp(spin):
    """
    Define a general function for the spin raising
    operators according to the general normalization
    rule/action of the raising operator on a state
    of total spin I and projection m along the z-axis:

    sp| I m > = \sqrt{(I)*(I + 1) - m * (m + 1)} |I (m + 1) >
    
    """

    proj = _spin_projections(spin)

    diag = np.sqrt(spin * (spin + 1) - proj * (proj + 1) )[:-1]

    return np.diag(diag, -1)


def sm(spin):
    """
    Define a general function for the spin lowering
    operators according to the general normalization
    rule/action of the lowering operator on a state
    of total spin I and projection m along the z-axis:

    sm| I m > = \sqrt{(I)*(I + 1) - m * (m - 1)} |I (m - 1) >
    """
    proj = _spin_projections(spin)

    diag = np.sqrt(spin * (spin + 1) - proj * (proj - 1) )[1:]

    return np.diag(diag, 1)


def sz(spin):
    """
    Define the general matrix for the sz operator which is
    simply a diagonal matix with the projections of the spin
    along the z-axis on the diagonal.
    
    """

    return np.diag(_spin_projections(spin), 0)


def sx(spin):
    """
    Define the general matrix for the sx operator.
    
    """

    sx = 0.5 * (sp(spin) + sm(spin))
    return sx


def sy(spin):
    """
    Define the general matrix for the sy operator.
    
    """
    sy = (0.5/(1j)) * (sp(spin) - sm(spin))

    return sy




def id(spin):
    """
    Identity operator of the dimension

    dim x dim where dim = 2 * spin + 1
    
    """
    return np.eye(int(2*spin + 1))




# ---------------------------------------------
#
# Build a dictionary that associates operators
# with their corresponding description strings
#
# NOTE: R is for random matrix
#
# ---------------------------------------------

operators = {'x': sx, 'y': sy, 'z': sz, '+': sp, '-': sm, 'I': id,
             'R': None}

# also allow for random matrices
