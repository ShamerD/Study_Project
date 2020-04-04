from AprioriDP import apriori


def run_test(T, min_supp, min_conf):
    print("Database-----------------")
    for num, transaction in enumerate(T_1):
        print(num, ":", *transaction)

    freq_sets_1, rules_1 = apriori(T_1, min_supp, min_conf)

    print("Frequent sets-------------")

    for subset in freq_sets_1:
        print("set:", subset[0], "support:", subset[1])

    print("Rules---------------------")

    for rule in rules_1:
        print("rule:", rule[0], "=>", rule[1], "confidence", rule[2])


print("Example 1")

T_1 = []
T_1.append(frozenset(("butter", "bread", "milk", "meat")))
T_1.append(frozenset(("butter", "bread", "meat")))
T_1.append(frozenset(("butter", "bread")))
T_1.append(frozenset(("bread", "milk", "meat")))
T_1.append(frozenset(("bread", "milk")))
T_1.append(frozenset(("milk", "meat")))
T_1.append(frozenset(("bread", "meat")))

run_test(T_1, 0.4, 0.7)
