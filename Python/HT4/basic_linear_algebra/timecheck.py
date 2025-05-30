from time import time


def check_time(func, *args, iter_count=1000,
               fname='measurements.txt', message=''):
    t1 = time()
    for _ in range(iter_count):
        res = func(*args)
    t = (time() - t1) / iter_count

    with open(fname, 'a') as f:
        print(f'{func.__name__:>15} {message:^25}: {t:.5e} sec, ', file=f)

    return res
