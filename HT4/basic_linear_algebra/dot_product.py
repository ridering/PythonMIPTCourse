def dot_product(v1, v2):
    assert len(v1) == len(v2)

    product = 0
    for elem1, elem2 in zip(v1, v2):
        product += elem1 * elem2

    return product
