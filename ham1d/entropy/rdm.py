import numpy as np
import numba as nb
import scipy.sparse as sparse
from scipy.special import comb
from ._combinatorics import binomial


@nb.njit()
def _ind0(iconf, npart):
    """
    A routine that finds a configuration index for
    a lattice with only two entities (spins and holes
    or up and down spins, for instance). NOTE:
    This is an internal (helper) routine, that is
    meant to be called from the ind routine.

    Parameters:
    -----------
    iconf - a configuration array, entries
            correspond to the sites at which
            the entities reside. For the
            following spin configuration:
            u d d u
            the iconf array would be:
            0 3 0 0
    npart - number of particles on a lattice

    Returns:
    --------

    indx: int
          The index of the state with a given
          configuration of up spins/particles.

    """
    indx = 1

    for i in range(0, npart):

        # we obtain the site at which
        # the particle resides
        site = iconf[i]

        # we add the number of possible
        # preceeding combinations
        # i + 1 ensures proper increase
        indx += binomial(site, i + 1, )

    return indx


@nb.njit()
def ind(iconf, L, nu):
    """
    A routine that finds a configuration index for a
    lattice with a
    given configuration of holes and up, down spins.

    INPUT:
    iconf - configuration array
    L - lattice size
    nu - number of up spins
    """

    icu = np.zeros(L, dtype=np.int64)
    # jh = 0
    ju = 0
    for i in range(L):

        if iconf[i] == 1:
            icu[ju] = i
            ju += 1

    # find indexes of the hole and up spin configurations:
    iu = _ind0(icu, nu)

    ind = iu

    return ind


@nb.njit()
def _conf0(ind, L, npart):
    """
    A routine to find a given configuration index for
    the case when there are only two entities on a lattice
    (for instance electrons/holes, up/down spins)

    Parameters:
    -----------

    ind: int
         configuration index of a requested state.
    L: int
       system size.
    npart: int
           Number of entities, such as particles or
           up pointing spins.

    Returns:
    --------

    iconf: ndarray, int
           A configuration array specifying at which
           sites the entities reside. Example for
           the following spin configuration:
           u d d u
           iconf => 0 3 0 0

    """

    if ind < 1:

        # print("Indexing should start with 1.")

        return

    ind -= 1
    iconf = np.zeros(L, dtype=np.int64)

    isite = L - 1
    while npart > 0:
        if ind >= binomial(isite, npart, ):
            iconf[npart - 1] = isite
            ind -= binomial(isite, npart, )

            npart -= 1

        isite -= 1

    return iconf


@nb.njit()
def conf(ind, L, nu, nh=0):
    if ind < 1:
        # print("Indexing starts with 1.")
        return

    iconf = np.zeros(L, dtype=np.int64)
    # iconfh=np.zeros(L, dtype=int)
    # iconfu=np.zeros(L, dtype=int)
    # the number of spin up configurations:

    iconfu = _conf0(ind, L, nu)

    for i in range(nu):
        iconf[iconfu[i]] = 1

    return iconf


# @nb.njit('Tuple((i8[:],i8[:,:]))(i8,i8,i8)')
# @nb.njit()
def rdm_blocks(L, subsize, npart):
    """
    A helper routine for simpler
    state indexation.

    rdm_blocks() returns the number fo states
    in each rdm block, where a block corresponds
    to a given number of up-spins and down-spins
    in a chosen subystem.

    Parameters:
    -----------

    L - int
        system size
    subsize: int
        subsystem size
    npart: int
        number of up spins/particles

    Returns:
    --------

    states: ndarray, int
                An array containing the number of spins
                in each block
    conf: ndarray, int
          An array containing configurations for each
          block -> this would be the number of up and
          down spins in each block.

    """

    # minimum and maximum numbers of holes for a given
    # subsystem and given nu

    umax = min(subsize, npart)
    umin = max(0, npart - (L - subsize))

    states = np.zeros(2 + umax - umin, dtype=np.int64)
    config = np.zeros((1 + umax - umin, 2), dtype=np.int64)

    i = 0
    for nup in range(umax, umin - 1, -1):

        # find the number of possible combinations
        # for
        nstates = binomial(subsize, nup, )

        states[i + 1] = nstates

        config[i, :] = np.array([nup, subsize - nup])
        i += 1

    # return np.cumsum(np.insert(states, 0, 0)), np.array(conf)
    return np.cumsum(states), np.array(config)
    
# @nb.njit()


def rdm_conf(rdm_ind, L, subsize, nu, blocks):
    """
    A function that returns a configuration corresponding to
    a given index in the rdm_matrix state ordering.

    Parameters:
    -----------

    rdm_ind: int
             rdm state index
    L: int
       system size
    subsize: int
             subsystem size
    nu: int
        number of up spins
    blocks
        the return of the rdm_blocks function -> the whole tuple

    """

    rdm_nstates = blocks[0][-1]

    if rdm_ind <= rdm_nstates:

        block_ind = np.argmin(blocks[0] < rdm_ind)
        istate = rdm_ind - blocks[0][block_ind - 1]
        block = blocks[1][block_ind - 1]

        return conf(istate, subsize, *block[:-1])

    else:
        print(('Index {} largert than the allowed number of '
               'rdm_states, which is {}.').format(rdm_ind,
                                                  rdm_nstates))


# @nb.jit()
def rdm_ind(rdm_conf, L, subsize, nu, blocks):
    """
    A function that finds the index corresponding to a
    given configuration in the rdm_matrix state ordering.

    """

    params = list(map(lambda x: np.count_nonzero(rdm_conf == x), [0, 1]))

    for i, block in enumerate(blocks[1]):

        if np.array_equal(params, block):

            block_ind = i

    ind_ = ind(rdm_conf, subsize, *params[:-1])

    return ind_ + blocks[0][block_ind]


# @nb.njit()
def _build_rdm(state, subsize, L, nu):
    """

    A function that builds the reduced density matrix
    for the smaller of the two subsystems.

    Parameters:
    ----------

    state - the state for which the entanglement entropy
    is to be computed.


    subsize: int
             subsystem size
    L: int
       system size
    nu: int
        number of up spins in the system

    OUTPUT:

    block diagonal matrix in sparse format


    WORKFLOW:

    The assert statements in the beginning check if the input
    arguments are valid for our considered model.

    subsize=min(subsize, L-subsize) ensures that the smaller
    of the two subsystems is always considered.

    Calls to rdm_blocks routines are made in order to obtain
    blocksA and blocksB lists. The latter contain the numbers of
    states for given configurations (nu, nd) and block
    structure (see the documentation of the rdm_blocks() function).
    Entries in blocksB list are reversed so that there is a simple
    correspondence between the configurations in both subsystems.

    The loops following the RDM CONSTRUCTION comment take care
    of the reduced density matrix construction. The outermost
    loop is over the configurations of the smaller subsystem.


    The matrix structure is obtained by using the np.outer() function
    which calculates the outer product. Finally, the block diagonal
    matrix in sparse format is obtained and returned using the
    scipy.sparse.block_diag() function.


    FORMULA FOR THE MATRIX ELEMENTS OF THE RDM:

    (RDM)_alpha,alpha'= SUM_beta'' (C_alpha,beta'')*(C^*_alpha',beta'')

    One has to multiply the state coefficients, corresponding to given
    alpha, alpha' first and then sum over all possible realizations of
    beta (states in the remaining subsystem)
    """

    assert len(state) == binomial(L, nu, )
    assert L >= subsize

    # make sure that the smaller matrix from the two is built
    # since the entanglement entropy of the two subsystems is
    # the same for A and B, it makes sense to calculate it for
    # the smaller subsystem
    subsize = min(subsize, L - subsize)

    # ---------------------------------------------------------
    #
    #           BLOCKS A and B
    #
    # ---------------------------------------------------------
    # find blocks - a block is a structure with a given number
    # of holes and up spins
    blocksA = rdm_blocks(L, subsize, nu)
    blocksB = rdm_blocks(L, L - subsize, nu)

    # reverse so we get  correspondence easily
    blocksB = [el[::-1] for el in blocksB]

    # SUBSYSTEM A
    num_statesA = np.diff(blocksA[0])

    # SUBSYSTEM B
    num_statesB = -np.diff(blocksB[0])

    # initialize block matrices for the whole system
    blocks = [np.zeros((nstates, nstates), dtype=np.complex128)
              for nstates in num_statesA]

    # ----------------------------------------------------------
    #
    #           RDM CONSTRUCTION
    #
    # ----------------------------------------------------------

    for i, (blockA, blockB) in enumerate(zip(blocksA[1], blocksB[1])):

        # iterate over appropriate states in the subsystem B
        for k in range(1, num_statesB[i] + 1):

            # spin and hole configuration in the subsystem B
            iconfB = conf(k, L - subsize, *blockB[:-1])
            # initialize states in the subsystem A to zero
            states_blockA = np.zeros(num_statesA[i], dtype=np.complex128)

            for j in range(1, num_statesA[i] + 1):
                # spin configuration in the A subsystem
                iconfA = conf(j, subsize, *blockA[:-1])
                # state index in the whole system
                ind_ = ind(np.append(iconfA, iconfB), L, nu)
                # fill the matrix with coefficients
                states_blockA[j - 1] = state[ind_ - 1]

            # add a matrix block corresponding to a state in the subsystem
            # BLOCKS
            blocks[i] += np.outer(states_blockA, np.conj(states_blockA))

    return blocks


def build_rdm(state, subsize, L, nu):

    blocks = _build_rdm(state, subsize, L, nu)

    return sparse.block_diag(blocks)
