import os
import sys

sys.path.append(os.path.dirname(os.path.join(os.getcwd(), __file__)))

import backpack

import collections.abc
import numpy as np

# TODO: Use RedBlackTree for maintain a non-dominated set

# $\Underscore{\Delta}$ in the Bazgan's paper
def dominate_eq(a, b):
    va, vb = a[1], b[1]
    return np.all(va >= vb)

# $\Underscore{\Delta}_\mathit{lex}$ in the Bazgan's paper
def lexical_dominate_eq(a, b):
    va, vb = a[1], b[1]
    if np.all(va == vb):
        return True
    j = min(np.where(va != vb))
    return va[j] > vb[j]

# $\geq_\mathit{lex}$ in the Bazgan's paper
def lexical_ge(a, b):
    wa, wb = a[2], b[2]
    if wa < wb:
        return True
    if wa == wb:
        return lexical_dominate_eq(a, b)
    return False

class StateSet(collections.abc.MutableSequence):
    def __init__(self):
        self.solutions = []
        self.value_vectors = []
        self.weights = []

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, i):
        return (self.solutions[i], self.value_vectors[i], self.weights[i])

    def __setitem__(self, i, s):
        # s should be (solution, value_vector, weight)
        self.solutions[i] = s[0]
        self.value_vectors[i] = s[1]
        self.weights[i] = s[2]

    def __delitem__(self, i):
        del self.solutions[i]
        del self.value_vectors[i]
        del self.weights[i]

    def insert(self, i, s):
        # s should be (solution, value_vector, weight)
        self.solutions.insert(i, s[0])
        self.value_vector.insert(i, s[1])
        self.weights.insert(i, s[2])

class Solver(object):
    def __init__(self, item_set):
        self.item_set = item_set

    @property
    def n_items(self):
        return self.item_set.n_items

    @property
    def n_criteria(self):
        return self.item_set.n_criteria

    @property
    def value_matrix(self):
        return self.item_set.value_matrix

    @property
    def weights(self):
        return self.item_set.weights

    def solve(self, knapsack_capacity):
        total_remaining_weights = np.sum(self.item_set.weights)

        self.states = StateSet()
        self.states.append((np.zeros(n_items, dtype=np.bool),
                            np.zeros(n_criteria, dtype=np.float32),
                            0))

        for k in range(self.n_items):
            self.states = self._compute_kth_candidates(knapsack_capacity, k, total_remaining_weights)
            total_remaining_weights -= self.item_set.weights[k]

    def _compute_kth_candidates(self, knapsack_capacity, k, total_remaining_weights):
        """Generate k-th candidates from the previous candidates

           This function is a Python implementation of Algorithm 3 described in the following paper:

           - C. Bazgan, et al. Solving efficiently the 0-1 multi-objective knapsack problem.
             Computers & Operations Research 36(1), 260-279 (2019)
        """
        assert(knapsack_capacity > 0)
        assert(total_remaining_weights > 0)
        assert(0 <= k && k < self.n_items)

        # Assume there are at leaset 1 candidate in the current state
        assert(len(self.states) > 0)

        v_k = self.item_set.value_matrix[k, :]
        w_k = self.item_set.weights[k]

        # Candidates generated in the previous phase is in [0:r]
        r = len(self.states)

        # Find the first index j at which the first D^k_r-dominant state appears
        for j in range(r):
            if self.states.weights[j] + total_remaining_weights > knapsack_capacity:
                break

        # The new state set
        new_states = StateSet()

        # The set of non-dominated candidates
        non_dominated = StateSet()

        # x and v is the placeholder of the current solution and value_vector
        x = np.zeros(self.n_items, dtype=np.bool)
        v = np.zeros(self.n_criteria, dtype=np.float32)
        w = 0

        for i in range(r):
            if self.states.weights[i] + w_k > knapsack_capacity:
                break
            x[:] = self.states.solutions[i]
            x[k] = True
            v[:] = self.states.value_vectors[i] + v_k
            w = self.states.weights[i] + w_k
            s = (x, v, w)
            while j < r and lexical_ge(self.states[j], s):
                self._maintain_non_dominated(self.states[j], non_dominated, new_states)
                j += 1
            self._maintain_non_dominated(s, non_dominated, new_states)

        # i is the index of the first state that cannot contain the k-th item

        while j < r:
            self._maintain_non_dominated(self.states[j], non_dominated, new_states)
            j += 1

        # Apply D_b here after
        if k == self.n_items - 1:
            return non_dominated
        else:
            F = []
            for order in ("sum", "max"):
                # TODO: Relabeling remaining items
                for s in non_dominated:
                    x[:] = s[0]
                    v[:] = s[1]
                    w = s[2]
                    while j in range(k + 1, self.n_items):
                        if w + self.item_set.weights[j] <= knapsack_capacity:
                            x[j] = True
                            v += self.item_set.value_vectors[j, :]
                            w += self.item_set.weights[j]
                    self._keep_non_dominates((x, v, w), F)
            remove = True
            for i in range(len(new_states)):
                # TODO: Compute upper-bounds
                remove = False
                for j in range(len(F)):
                    if not lexical_dominate_eq(F[j], u):
                        break
                    if remove:
                        break
                    if dominate_eq(F[j], u)
                        remove = True
                        break
                if remove:
                    del new_state[i]
            return new_state

    def _maintain_non_dominated(self, s, non_dominated, new_state):
        l = len(non_dominated)
        dominated = False
        for i in range(l):
            if not lexical_dominate_eq(non_dominated[i], s):
                break
            if dominate_eq(non_dominated[i], s):
                dominated = True
                break
        if not dominated:
            new_state.append(s)
            non_dominated.append(s)
            while i < l:
                if dominate_eq(s, non_dominated[i]):
                    del non_dominated[i]
                else:
                    i += 1

    def reserve(self, size):
        if self.capacity >= size:
            return
        old_capacity = self.capacity
        while self.capacity < size:
            self.capacity = round(self.capacity * 1.618)
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
