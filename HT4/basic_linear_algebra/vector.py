def convolve(data, kernel):
    assert len(kernel) <= len(data)

    res = [0] * (len(data) - len(kernel) + 1)
    for i in range(len(res)):
        for j in range(len(kernel)):
            res[i] += data[i + j] * kernel[j]

    return res
