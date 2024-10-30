
import os
import random

import numpy as np

from basic_linear_algebra.creation import *
from basic_linear_algebra.dot_product import *
from basic_linear_algebra.hist import *
from basic_linear_algebra.io import *
from basic_linear_algebra.matrix_trace import *
from basic_linear_algebra.multiplication import *
from basic_linear_algebra.timecheck import *
from basic_linear_algebra.vector import *

n = random.randint(3, 10)
p = random.randint(3, 10)
m = random.randint(3, 10)

A = create_matrix(n, p)
B = create_matrix(p, m)
R = create_matrix(n, n)
x = create_vector(p)
y = create_vector(p)
v = create_vector_gauss(1000, 17.42, 5.8)
kernel = [-1, 0, 1]

C = matrix_matrix(A, B)
b = matrix_vector(A, x)
dot = dot_product(x, y)
bb = bins(v, 50)
convolved = convolve(x, kernel)

assert np.all(np.isclose(np.array(A) @ B, C))
assert np.all(np.isclose(np.array(A) @ x, b))
assert np.isclose(np.dot(x, y), dot)
assert np.isclose(np.trace(R), trace(R))

# visualize_bins(bb, 50)

assert np.all(np.isclose(np.convolve(x, kernel[::-1], 'valid'), convolved))

save_matrix(A, 'test_matrix_save.txt', '15.6')
save_vector(x, 'test_vector_save.txt', '15.6')

A_new = load_matrix('test_matrix_save.txt')
x_new = load_vector('test_vector_save.txt')

try:
    assert np.all(np.isclose(A, A_new))
    assert np.all(np.isclose(x, x_new))
finally:
    os.remove('test_matrix_save.txt')
    os.remove('test_vector_save.txt')

fname = 'measurements.txt'
try:
    os.remove(fname)
except BaseException:
    pass

# matrix multiplication
for size in [10, 100, 1000]:
    it_count = 1_000_000 // (size * size)

    m1 = create_matrix(size, size)
    m2 = create_matrix(size, size)
    check_time(matrix_matrix, m1, m2, fname=fname, iter_count=it_count,
               message=f'{size}x{size} x {size}x{size}')

# matrix-vector multiplication
for size in [10, 100, 1000]:
    it_count = 1_000_000 // (size * size)

    m1 = create_matrix(size, size)
    v1 = create_vector(size)
    check_time(matrix_vector, m1, v1, fname=fname, iter_count=it_count,
               message=f'{size}x{size} x {size}')

# trace of matrix
for size in [10, 100, 1000, 10000]:
    it_count = 1_000_000 // size

    m1 = create_matrix(size, size)
    check_time(trace, m1, fname=fname, iter_count=it_count,
               message=f'{size}x{size}')

# dot product
for size in [10, 100, 1000, 10000]:
    it_count = 1_000_000 // size

    v1 = create_vector(size)
    v2 = create_vector(size)
    check_time(dot_product, v1, v2, fname=fname, iter_count=it_count,
               message=f'{size} x {size}')

# histogram
for size in [10, 100, 1000, 10000]:
    it_count = 1_000_000 // size

    v1 = create_vector_gauss(size, 17.42, 5.8)
    check_time(bins, v1, 50, fname=fname, iter_count=it_count,
               message=f'{size} x {size}')

# convolution
for data_size in [10, 100, 1000]:
    for kernel_size in [5, 50, 500]:
        if kernel_size > data_size:
            continue
        it_count = 1_000_000 // data_size

        v1 = create_vector(data_size)
        v2 = create_vector(kernel_size)
        check_time(convolve, v1, v2, fname=fname, iter_count=it_count,
                   message=f'{data_size} x {kernel_size}')

# save matrix
for size in [10, 100, 1000, 10000]:
    it_count = 10_000 // size

    m1 = create_matrix(size, size)
    check_time(save_matrix, m1, 'test_matrix_save.txt', fname=fname,
               iter_count=it_count, message=f'{size}x{size}')

# save vector
for size in [10, 100, 1000, 10000]:
    it_count = 100_000 // size

    v1 = create_vector(size)
    check_time(save_vector, v1, 'test_vector_save.txt', fname=fname,
               iter_count=it_count, message=f'{size}')

# load matrix
for size in [10, 100, 1000, 10000]:
    it_count = 10_000 // size

    m1 = create_matrix(size, size)
    save_matrix(m1, 'test_matrix_save.txt')
    check_time(load_matrix, 'test_matrix_save.txt', fname=fname,
               iter_count=it_count, message=f'{size}x{size}')

# load vector
for size in [10, 100, 1000, 10000]:
    it_count = 100_000 // size

    v1 = create_vector(size)
    save_vector(v1, 'test_vector_save.txt')
    check_time(load_vector, 'test_vector_save.txt', fname=fname,
               iter_count=it_count, message=f'{size}')

try:
    os.remove('test_matrix_save.txt')
    os.remove('test_vector_save.txt')
except BaseException:
    pass
