def matrix_matrix(A, B):
    assert 0 < len(A)
    assert 0 < len(B)
    assert 0 < len(A[0])
    assert 0 < len(B[0])

    n = len(A)
    p = len(A[0])
    q = len(B)
    m = len(B[0])

    assert p == q

    C = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            elem = 0
            for k in range(p):
                elem += A[i][k] * B[k][j]
            C[i][j] = elem

    return C


def matrix_vector(A, x):
    assert 0 < len(A)
    assert len(A[0]) == len(x)

    b = [0.0] * len(A)
    for i, row in enumerate(A):
        for j, elem in enumerate(row):
            b[i] += x[j] * elem

    return b
