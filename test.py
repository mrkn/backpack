import os
import sys

sys.path.append(os.path.dirname(os.path.join(os.getcwd(), __file__)))

import backpack
import numpy as np

class Solver(object):
    def __init__(self, item_set):
        self.item_set = item_set
        self.capacity = 128
        self.n_solutions = 1
        self.solutions = np.zeros((self.capacity, self.n_items), dtype=np.bool)
        self.states = np.zeros((self.capacity, self.n_criteria + 1), dtype=np.float32)
        self.validities = np.zeros((self.capacity,), dtype=np.bool)
        self.validities[0] = True

    @property
    def n_items(self):
        return self.item_set.n_items

    @property
    def n_criteria(self):
        return self.item_set.n_criteria

    # This function is a Python implementation of Algorithm 3 described in the following paper:
    #
    # - C. Bazgan, et al. Solving efficiently the 0-1 multi-objective knapsack problem.
    #   Computers & Operations Research 36(1), 260-279 (2019)
    def solve(self, C):
        total_remaining_weights = np.sum(self.item_set.weights)

        for k in range(self.n_items):
            n0 = self.n_solutions
            M = []
            self.reserve(n0 * 2)

            # C^{k-1} is in [0:n0]

            for j in range(n0):
                if self.states[j, -1] + total_remaining_weights > C:
                    break

            # j is the index of the first state whose full completion cannot fit the knapsack

            s = np.zeros((self.n_criteria + 1,))
            for i in range(n0):
                if self.states[i, -1] + self.item_set.weights[k] > C:
                    break
                s[:-1] = self.states[i, :-1] + self.item_set.value_matrix[k, :]
                s[-1] = self.states[i, -1] + self.item_set.weights[k]
                while j < n0 and lexical_ge(self.states[j, :], s):
                    self._maintain_non_dominated(self.states[j, :], M, n0)
                    j += 1
                self._maintain_non_dominated(s, M, n0)

            # i is the index of the first state that cannot contain the k-th item

            while j < n0:
                self._maintain_non_dominated(self.states[j, :], M, n0)
                j += 1

            # Apply D_b here after
            if k == self.n_items - 1:
                return M

            # TODO;

    def _maintain_non_dominated(self, s, M, n0):
        dominated = False
        for i in range(len(M)):
            if not lexical_dominate_eq(self.states[M[i], :-1], s):
                break
            if dominate_eq(self.states[M[i]], s):
                dominated = True
                break
        if not dominated:
            self.states[self.n_solutions, :] = s
            M.append(self.n_solutions)
            self.n_solutions += 1
            while i < len(M):
                if dominate_eq(s, self.states[M[i]]):
                    del M[i]
                else:
                    i += 1

    def reserve(self, size):
        if self.capacity >= size:
            return
        old_capacity = self.capacity
        while self.capacity < size:
            self.capacity = round(self.capacity * 1.5)
        self.solutions.resize((self.capacity, self.solutions.shape[1]), refcheck=False)
        self.states.resize((self.capacity, self.states.shape[1]), refcheck=False)
        self.validities.resize((self.capacity,), refcheck=False)

n_items = 15
n_criteria = 2

rng = np.random.default_rng()
w = rng.integers(1001, size=n_items)
v1 = rng.integers(1001, size=n_items)
v2 = rng.integers(np.maximum(900 - v1, 1), np.minimum(1100 - v1, 1000))
v = np.vstack((v1, v2))
item_set = backpack.ItemSet(v, w)
solver = Solver(item_set)

print((100*v/w).astype(int))
print(item_set.item_order)
print((100*item_set.value_matrix/item_set.weights).astype(int))
