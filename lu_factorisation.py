def lu_factorisation(A):
    """
    Compute the LU factorisation of a square matrix A.

    The function decomposes a square matrix ``A`` into the product of a lower
    triangular matrix ``L`` and an upper triangular matrix ``U`` such that:

    .. math::
        A = L U

    where ``L`` has unit diagonal elements and ``U`` is upper triangular.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the square matrix to
        factorise.

    Returns
    -------
    L : numpy.ndarray
        A lower triangular matrix with shape ``(n, n)`` and unit diagonal.
    U : numpy.ndarray
        An upper triangular matrix with shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    A = np.array(A, dtype=float, copy=False)
    # construct arrays of zeros
    L, U = np.zeros_like(A), np.zeros_like(A)

    # notes say need a column by column construction of LU 
    L[0, 0] = 1
    U[0, 0] = 1

    for j in range(n):
        L[j, j] = 1.0

        for i in range(j + 1):
            s = 0.0
            for k in range(i):
                s += L[i,k] * U[k, j]
            U[i, j] = A[i, j] - s

        if np.isclose(U[j, j], 0.0):
            raise ZeroDivisionError()


        for i in range(j + 1, n):
            s = 0.0
            for k in range(j):
                s += L[i, k] * U[k, j]
            L[i, j] = (A[i, j] - s) / U[j, j]

    return L, U
    
    