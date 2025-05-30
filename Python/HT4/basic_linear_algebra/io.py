def load_matrix(fname):
    with open(fname, 'r') as file:
        return [list(map(float, line.split()))
                for line in file if line.strip()]


def load_vector(fname):
    with open(fname, 'r') as file:
        return list(map(float, file.readline().split()))


def save_matrix(A, fname, format='10.3'):
    with open(fname, 'w') as file:
        for row in A:
            for elem in row:
                print(f'{elem:{format}}', end='', file=file)
            print(file=file)
        print(file=file)


def save_vector(v, fname, format='10.3'):
    with open(fname, 'w') as file:
        for elem in v:
            print(f'{elem:{format}}', end='', file=file)
        print(file=file)
