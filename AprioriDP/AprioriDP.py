import numpy as np
from collections import defaultdict


def construct_frequent_sets(T, item2num, num2item, min_supp):
    '''Construct sets of items that satisfy min_supp constraint

    Args:
        T : database (contains sets) of size p
        item2num : map from item to its unique order number
        num2item : list of items in their order
        min_supp : minimum support of a subset

    Returns:
        L : set of frequent subsets
        freq_size : map from frequent subset to number of transactions
    '''
    p = len(T)  # total transactions
    n_items = len(item2num)  # total items
    count_table = np.zeros((n_items, n_items))
    freq_size = defaultdict(int)

    L_1 = set()  # 1-item frequent subsets
    L_2 = set()  # 2-item frequent subsets
    L = set()  # all frequent subsets

    # count 1- and 2-item subsets
    for transaction in T:
        for elem1 in transaction:
            for elem2 in transaction:
                count_table[item2num[elem1], item2num[elem2]] += 1

    # add 1- and 2- frequent item subsets to L_1 and L_2
    for i in range(n_items):
        if count_table[i][i] / p >= min_supp:
            L_1.add(frozenset([num2item[i]]))
            freq_size[frozenset([num2item[i]])] += count_table[i][i]
            for j in range(i + 1, n_items):
                if count_table[i][j] / p >= min_supp:
                    tmp_subset = frozenset([num2item[i], num2item[j]])
                    L_2.add(tmp_subset)
                    freq_size[tmp_subset] += count_table[i][j]

    # update L
    L.update(L_1)
    L.update(L_2)

    L_k = L_2.copy()  # set of frequent subsets of size k
    C_k = set()  # candidate set
    k = 3

    def generate_candidate_set(freq_set, sz):
        '''generate set of sz-elem subsets based on (sz-1)-elem freq subsets'''
        answ = set()
        for a in freq_set:
            for b in freq_set:
                if len(a.union(b)) == sz:
                    answ.add(frozenset(a.union(b)))
        return answ

    def generate_frequent_set(candidate_set, DB, freq_counter, min_supp):
        '''generate set of frequent subsets using info from candidate_set'''
        answ = set()
        local_counter = defaultdict(int)

        for transaction in DB:
            for subset in candidate_set:
                if subset.issubset(transaction):
                    local_counter[frozenset(subset)] += 1

        for subset in candidate_set:
            fr_subset = frozenset(subset)
            if local_counter[fr_subset] / p >= min_supp:
                answ.add(fr_subset)
                freq_counter[fr_subset] += local_counter[fr_subset]

        return answ

    # try to get freq sets of bigger size
    while len(L_k) != 0:
        C_k.clear()
        L_k.clear()

        C_k = generate_candidate_set(L_k, k)
        L_k = generate_frequent_set(C_k, T, freq_size, min_supp)

        L.update(L_k)
        k += 1

    return L, freq_size
