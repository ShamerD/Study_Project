import numpy as np
from collections import defaultdict

T_1 = []
T_1.append(frozenset(("butter", "bread", "milk", "meat")))
T_1.append(frozenset(("butter", "bread", "meat")))
T_1.append(frozenset(("butter", "bread")))
T_1.append(frozenset(("bread", "milk", "meat")))
T_1.append(frozenset(("bread", "milk")))
T_1.append(frozenset(("milk", "meat")))
T_1.append(frozenset(("bread", "meat")))
default_sets = T_1


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
        C_k = generate_candidate_set(L_k, k)

        L_k.clear()
        L_k = generate_frequent_set(C_k, T, freq_size, min_supp)

        L.update(L_k)
        k += 1

    return L, freq_size


def check_subset(common_subset, current_subset, rules, min_conf, freq_size):
    '''Recursively check whether (cur -> (comm - cur)) is a confident rule

    Args:
        common_subset : subset to check
        current_subset : subset to check
        rules : list of tuples ((from), (to), confidence) - changable object
        min_conf : confidence (supp(from | to) / supp(from)) constraint
        freq_size : number of times each subset contains in database

    Returns:
        nothing
    '''
    if len(current_subset) == 0:
        return

    cur_confidence = freq_size[common_subset] / freq_size[current_subset]

    # check confidence constraint
    if (cur_confidence >= min_conf):
        # add only if (to) is not empty
        if (common_subset != current_subset):
            rules.append((tuple(current_subset),
                          tuple(common_subset.difference(current_subset)),
                          cur_confidence))
        # check for subset of size - 1
        for element in current_subset:
            new_subset = frozenset(current_subset.difference(set([element])))
            check_subset(common_subset, new_subset,
                         rules, min_conf, freq_size)


def construct_rules(frequent_sets, freq_size, min_conf):
    '''Construct confident rules based on frequent subsets

    Args:
        frequent_sets : set of frequent subsets
        freq_size : map from freq subset to number of times it contains in DB
        min_conf : confidence constraint
    '''
    rules = []
    for common_subset in frequent_sets:
        check_subset(common_subset, common_subset, rules, min_conf, freq_size)
    return rules


def apriori(min_supp, min_conf, T=default_sets):
    '''Run AprioriDP on database T

    Args:
        T: database - iterable that contains sets of items
        min_supp : minimum support constraint
        min_conf : minimum confidence constraint

    Returns:
        freq_subsets : frequent subsets that satisfy support constraint
                       list of (set, support)
        conf_rules : rules that satisfy confidence constraint
                     list of (from, to, confidence)
    '''
    # maps used in construct freq sets
    item2num = {}
    num2item = {}
    items = set()
    p = len(T)

    # construct all-items set
    for transaction in T:
        for item in transaction:
            items.add(item)

    # fill maps
    for num, item in enumerate(items):
        num2item[num] = item
        item2num[item] = num

    L, freq_size = construct_frequent_sets(T, item2num, num2item, min_supp)

    # return format
    freq_subsets = []
    for subset in L:
        freq_subsets.append((tuple(subset), freq_size[subset] / p))

    # construct rules
    conf_rules = construct_rules(L, freq_size, min_conf)

    return freq_subsets, conf_rules
