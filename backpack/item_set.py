import numpy as np

def _sort_items(value_matrix, weights):
    assert 2 == value_matrix.ndim
    assert 1 == weights.ndim

    m, n = value_matrix.shape
    assert n == len(weights)

    value_matrix = -value_matrix / weights
    if m == 1:
        order = np.argsort(value_matrix[0, :])
    else:
        order = np.lexsort(value_matrix)
    return order

class ItemSet(object):
    def __init__(self, value_matrix, weights):
        self.n_criteria, self.n_items = value_matrix.shape
        assert (self.n_items,) == weights.shape, \
               f"Invalid shape of weight vector ({weight.shape} for ({self.n_items},))"
        self.item_order = _sort_items(value_matrix, weights)
        self.value_matrix = value_matrix[:, self.item_order]
        self.weights = weights[self.item_order]
