"""
A module with definitions of
the single spin 1/2 hamiltonian
operators used in 1D spin chain
calculations.

NOTE: THE LOCAL BASIS HERE IS:
I 0 >, | 1 >
In this basis, the Sz operator
has the following structure:
| -1  0 | 
| 0 1 |
"""

import numpy as np

from numpy import random

# OPERATOR DEFINITIONS

sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.int8)
sy = 0.5 * np.array([[0, -1j], [1j, 0]])
sz = 0.5 * np.array([[-1, 0], [0, 1]], dtype=np.int8)

# spin up operator
sp = np.array([[0, 0], [1, 0]], dtype=np.int8)

# spin down operator
sm = np.array([[0, 1], [0, 0]], dtype=np.int8)

# identity
id2 = np.eye(2, dtype=np.int8)


# ---------------------------------------------
#
# Build a dictionary that associates operators
# with their corresponding description strings
#
# NOTE: R is for random matrix
#
# ---------------------------------------------

operators = {'x': sx, 'y': sy, 'z': sz, '+': sp, '-': sm, 'I': id2,
             'R': None}

# also allow for random matrices
