#!/usr/bin/env python3
import numpy as np
from scipy import linalg


def lu(M_in, pv):
    """
    Compute the LU decomposition of a matrix M=(P)LU.
    Input:
    nxn matrix M_in; pv=T/F (pivoting).
    Output:
    matrix M containing U in the upper triangle and
    L below the main diagonal (NB: the diagonal elements
    of L are equal to 1, not stored);
    vector z describing the permutations: row i was
    interchanged with row z[i].
    """
    M = np.copy(M_in)
    [m, n] = M.shape
    if m != n:
        raise SystemExit("M is not a square matrix: exit")
    if abs(linalg.det(M)) < 1e-8:
        raise SystemExit("M is singular")

    # initialize line swap vector
    z = np.arange(n)
    for j in range(n - 1):
        # pivot search
        pivot = abs(M[j, j])
        p = j
        if pv:
            for i in range(j + 1, n):
                tmp_pivot = abs(M[i, j])
                if tmp_pivot > pivot:
                    pivot = tmp_pivot
                    p = i  # update index

        if pivot != 0:
            if pv:
                # line swapping (elegant)
                M[[j, p]] = M[[p, j]]
                z[j], z[p] = z[p], z[j]
            for i in range(j + 1, n):
                M[i, j] = (
                    M[i, j] / M[j, j]
                )  # division by pivot, M[i,j] now stores L[i,j]
                M[i, j + 1 : n] = M[i, j + 1 : n] - M[i, j] * M[j, j + 1 : n]
        else:
            print(f"pivot = 0! j = {j}")

    return M, z
