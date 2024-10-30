def trace(A):
    assert 0 < len(A) == len(A[0])

    d_sum = 0
    for i in range(len(A)):
        d_sum += A[i][i]

    return d_sum
