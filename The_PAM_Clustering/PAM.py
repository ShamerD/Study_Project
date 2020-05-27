import numpy as np


def manhattan(x, y):
    '''Classic Manhattan distance

    Args:
        x : 1d array with size n
        y : 1d array with size n

    Returns:
        Manhattan distance between x and y
    '''
    return np.sum(np.absolute(x - y))


def PAM_Build(d, k):
    ''' BUILD phase for PAM Clustering algorithm

    Args:
        d : array of shape(n_objects, n_objects) - matrix of pairwise distances
        k : desired num of clusters

    Returns:
        S : set of numbers of medoids
        U : set of numbers of non-medoids
        C : array of size n_objects - cluster labels for each point
        d_nearest : array of size n_objects - distances to closest medoids
        totalDistance : sum of distances from points to their medoids
    '''
    n_objects = d.shape[0]
    S = set()
    U = set(range(n_objects))

    s = np.argmin(np.sum(d, axis=1))  # first medoid
    S.add(s)
    U.remove(s)
    C = np.array([s for i in range(n_objects)])
    d_nearest = np.array([d[i, s] for i in range(n_objects)])
    totalDistance = np.sum(d_nearest)

    while len(S) < k:
        diff_TD_best, m_best = None, None
        for unselected in U:  # unselected candidate to become a medoid
            diff_TD = 0  # evaluate difference in total distance
            for element in U:
                if element == unselected:
                    continue
                delta = d[unselected, element] - d_nearest[element]
                if delta < 0:
                    diff_TD += delta
            if diff_TD_best is None or diff_TD < diff_TD_best:
                diff_TD_best = diff_TD  # take best difference
                m_best = unselected
        totalDistance += diff_TD_best
        S.add(m_best)
        U.remove(m_best)
        for i in range(n_objects):  # update nearest medoids for each object
            if d[i, m_best] < d_nearest[i]:
                C[i] = m_best
                d_nearest[i] = d[i, m_best]

    return S, U, C, d_nearest, totalDistance


def PAM_Search(d, C, d_nearest, d_second, S, U, totalDistance, maxIter):
    '''SWAP Phase for PAM Clustering

    Args:
        d : array of shape(n_objects, n_objects) - matrix of pairwise distances
        C : array of size n_objects - cluster labels for each point
        d_nearest : array of size n_objects - distances to closest medoids
        d_second : array of size n_objects - distances to second closest
        S : set of numbers of medoids
        U : set of numbers of non-medoids
        totalDistance : sum of distances from points to their medoids
        maxIter : maximum iterations in SWAP phase

    Returns:
        S : set of numbers of medoids
        C : list of size n_objects - cluster labels for each point
        totalDistance : sum of distances from points to their medoids
    '''
    n_objects = d.shape[0]
    gets_better = True  # flag to continue
    iter_count = 0

    while gets_better:
        iter_count += 1
        gets_better = False
        diff_TD_best, m_best, x_best = 0, None, None

        med2ord = {}  # map from global medoid id to medoid id (<k)
        ord2med = {}  # vice versa
        for i, elem in enumerate(S):
            med2ord[elem] = i
            ord2med[i] = elem

        for non_medoid in U:
            delta = np.zeros(len(S))  # delta for each medoid (size k)
            delta -= d_nearest[non_medoid]  # loss for making non-med a medoid
            for other in range(n_objects):  # each point
                if other == non_medoid:
                    continue
                nearest = med2ord[C[other]]

                # change for nearest
                delta[nearest] += min(d[other, non_medoid],
                                      d_second[other]) - d_nearest[other]

                # change for non-nearest
                change = d[other, non_medoid] - d_nearest[other]
                if change < 0:
                    delta += change
                    delta[nearest] -= change  # already counted before

            best_medoid = int(np.argmin(delta))
            if delta[best_medoid] < diff_TD_best:
                diff_TD_best = delta[best_medoid]
                m_best = ord2med[best_medoid]
                x_best = non_medoid

        if diff_TD_best < 0:  # perform best swap
            gets_better = True
            S.remove(m_best)
            S.add(x_best)
            U.remove(x_best)
            U.add(m_best)
            totalDistance += diff_TD_best

            for i in range(n_objects):  # upgrade nearest, second nearest
                if C[i] != m_best:
                    if d[i, x_best] < d_nearest[i]:
                        C[i] = x_best
                        d_second[i] = d_nearest[i]
                        d_nearest[i] = d[i, x_best]
                    else:
                        tmp = d[i, np.array(list(S))]
                        d_second[i] = np.partition(tmp, 1)[1]
                else:
                    if d[i, x_best] < d_second[i]:
                        C[i] = x_best
                        d_nearest[i] = d[i, x_best]
                    else:
                        d_nearest[i] = d_second[i]
                        for j in S:
                            if d[i, j] == d_nearest[i]:
                                C[i] = j
                        tmp = d[i, np.array(list(S))]
                        d_second[i] = np.partition(tmp, 1)[1]
            if iter_count >= maxIter:
                break
    return S, C, totalDistance


def PAM(X, k, dist=manhattan, maxIter=10000):
    '''The PAM Clustering algorithm

    Args:
        X : iterable of size (n_objects)
        k : desired number of clusters
        dist : distance function, default - manhattan distance
        maxIter : maximum iterations in SWAP phase

    Returns:
        c : list of medoids of size k
        C : list of size n_objects - cluster labels for each point
        totalDistance : sum of distances from points to their medoids
    '''
    n_objects = len(X)

    tmp_d = [[0 for _ in range(n_objects)] for j in range(n_objects)]
    for i, item1 in enumerate(X):
        for j, item2 in enumerate(X):
            tmp_d[i][j] = dist(item1, item2)

    d = np.array(tmp_d)

    S, U, C, d_nearest, totalDistance = PAM_Build(d, k)  # see PAM_Build

    if k > 1:
        d_second = np.zeros(n_objects)  # distance to second nearest medoid
        for i in range(n_objects):
            tmp = d[i, np.array(list(S))]
            d_second[i] = np.partition(tmp, 1)[1]

        S, C, totalDistance = PAM_Search(d, C, d_nearest,
                                         d_second, S, U,
                                         totalDistance, maxIter)

    c = []
    for med in S:
        c.append(X[med])

    return c, C, totalDistance


if __name__ == "__main__":  # example
    X = np.array([[1, 0, 3], [0, 3, 2], [6, 1, 3], [2, 4, 3], [3, 8, 1]])
    k = 2
    medoids, cluster, totalDistance = PAM(X, k)
    print(totalDistance)
    print(medoids)
    print(cluster)
