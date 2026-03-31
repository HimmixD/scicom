#!/usr/bin/env python3
import numpy as np
from scipy import linalg

def qr_hh(A):
    """ 
    Compute the QR decomposition of an mxn matrix A with Householder reflections.
    In output the mxn matrix A and an n-dimensional vector r. 
    The normalized Householder vectors v (from which the Householder matrices 
    are built) are stored in the lower triangular part of A and its diagonal. 
    The upper triangular matrix R resulting from the successive application of 
    Householder matrices is stored above the diagonal of A.
    The diagonal elements of R are stored in the vector r.
    The Householder matrix Q_k can be built from the Householder vector v_k.
    """
    m,n = A.shape
    A=A.astype(float)
    r = np.zeros(n)
    for k in range(n): # loop over all columns of A
        alpha = -np.sign(A[k,k]) * linalg.norm(A[k:m,k])
        norm_v = np.sqrt( -2*alpha * (A[k,k] - alpha) ) # norm of v
        r[k] = alpha # diagonal elements of R
        # store normalized Householder vector v_k in the lower triangle of A
        A[k,k] = (A[k,k] - alpha) / norm_v # first element
        A[k+1:m,k] = A[k+1:m,k] / norm_v # remaining components
        
        # compute Q_1 ... Q_k * A
        for j in range(k+1,n): # loop over remaining columns of A
        # scalar product of H. vector with j-th column of A, starting from line k
            pj = 2 * np.dot( A[k:m,k], A[k:m,j] )
            A[k:m,j] = A[k:m,j] - pj * A[k:m,k] # update the jth column of A
    
    return A, r