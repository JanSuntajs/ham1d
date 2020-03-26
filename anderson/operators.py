import numpy as np
import numba as nb


_signature1 = 'uint64[:](uint64, uint64[:])'
_signature2 = 'uint64(uint64[:], uint64[:])'
_signature3 = ('Tuple((uint64[:], uint64[:], float64[:]))(uint64[:]'
               ', float64[:], float64[:], uint64, uint64, b1)')


@nb.njit('uint64[:](uint64[:])', nogil=True, fastmath=True, cache=True)
def _get_products(dimensions):

    products = np.zeros_like(dimensions, dtype=np.uint64)
    products[0] = 1
    products[1:] = np.cumprod(dimensions[:-1])

    return products


@nb.njit(_signature1, fastmath=True, nogil=True, cache=True)
def get_coordinates(state, dimensions):

    coordinates = np.zeros(len(dimensions), dtype=np.uint64)
    products = _get_products(dimensions)
    for i in range(len(dimensions) - 1, -1, -1):

        idx = np.int(state / products[i])
        coordinates[i] = idx

        state -= idx * products[i]

    return coordinates


@nb.njit(_signature2, fastmath=True, nogil=True, cache=True)
def get_idx(coordinates, dimensions):

    products = _get_products(dimensions)

    return np.uint64((1.0 * coordinates).dot(products * 1.0))


@nb.njit(_signature3, fastmath=True, nogil=True, cache=True)
def _ham_ops(dimensions, hopping, disorder, start_row, end_row, pbc):

    rows = []
    cols = []
    vals = []

    for state in range(start_row, end_row):

        coords = get_coordinates(state, dimensions)

        for j, coord in enumerate(coords):

            condition1 = (coord == 0)
            condition2 = (coord == dimensions[j] - 1)
            condition3 = not (condition1 or condition2)

            coords_new = np.copy(coords)
            if (pbc or condition3):
                for k in range(2):

                    coords_new[j] = (coords[j] + (-1) ** k) % dimensions[j]
                    state_new = get_idx(coords_new, dimensions)
                    rows.append(state)
                    cols.append(state_new)
                    vals.append(hopping[j])
            elif (condition1 or condition2):
                if condition1:
                    coords_new[j] = (coords[j] + 1)
                    state_new = get_idx(coords_new, dimensions)

                elif condition2:
                    coords_new[j] = (coords[j] - 1)
                    state_new = get_idx(coords_new, dimensions)

                rows.append(state)
                cols.append(state_new)
                vals.append(hopping[j])

        rows.append(state)
        cols.append(state)
        vals.append(disorder[state])

    rows = np.array(rows, dtype=np.uint64)
    cols = np.array(cols, dtype=np.uint64)
    vals = np.array(vals, dtype=np.float64)
    return rows, cols, vals


def ham_ops(dimensions, hopping, disorder, start_row, end_row, pbc=True):

    dimensions = np.array(dimensions, dtype=np.uint64)
    hopping = np.array(hopping, dtype=np.float64)
    disorder = np.array(disorder, dtype=np.float64).flatten()

    rows, cols, vals = _ham_ops(
        dimensions, hopping, disorder, start_row, end_row, pbc)

    return rows, cols, vals
