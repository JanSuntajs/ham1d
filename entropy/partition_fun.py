import numpy as np


def _partition_homogenous(eigenstate, L, L_sub):
    """
    A partitioning scheme that partitions a system
    into two homogenous subblocks next to each other,
    which can be schematically shown as:

    L -> L_A | L_B

    Here, A and B denote subsystems A and B. The
    corresponding Hilbert space dimensions are:

    The whole system: N = 2**L
    Subsystem A: N_A = 2**L_A
    Subsystem B: N_B = 2**L_B

    Parameters
    ----------

    eigenstate: ndarray
                1D ndarray containing an eigenstate
                to be partitioned.
    L: int
       System size
    L_sub: int
       Subsystem size for one of the subsystem. The
       size of the remaining one is of course
       L - L_sub.

    Returns
    -------

    eig_mat: ndarray
                 2D ndarray -> eigenstate partitioned
                 as a tensor product of the two subsystems
                 described above.

    """
    # convert eigenstate to an array
    eigenstate = np.array(eigenstate)
    # the remaining subsystem's size
    L_sub_ = L - L_sub

    # reshape the eigenstate -> write it as a tensor
    # product of the two subsystems
    eig_mat = eigenstate.reshape(eigenstate, (2**L_sub, 2**L_sub_))

    return eig_mat


partition_dict = {'homogenous': _partition_homogenous}
