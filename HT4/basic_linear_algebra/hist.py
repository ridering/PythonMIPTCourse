def bins(v, n):
    assert 0 < n

    count = [0] * n

    min_val = min(v)
    max_val = max(v)
    variation = max_val - min_val

    for elem in v:
        normalized = (elem - min_val) / variation
        if elem == max_val:
            count[-1] += 1
        else:
            count[int(normalized * n)] += 1

    return {(i / n * variation + min_val, (i + 1) / n * variation +
             min_val): elem for i, elem in enumerate(count)}


def visualize_bins(bins, scale):
    max_len = max(bins.values())
    for (a, b), count in bins.items():
        print(f'[{a:10.2}:{b:10.2}]: ', end='')
        print('o' * int(count / max_len * scale))
    print()
