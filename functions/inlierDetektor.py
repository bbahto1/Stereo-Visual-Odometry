import numpy as np

def findClique(d3dPointsT1, d3dPointsT2, distDifference):
    # in-lier detection algorithm
    brojTacki = d3dPointsT1.shape[0]
    W = np.zeros((brojTacki, brojTacki))

    cnt = 0
    maxn = 0
    maxc = 0
    # diff of pairwise euclidean distance between same points in T1 and T2
    for i in range(brojTacki):
        T1Diff = d3dPointsT1[i,:] - d3dPointsT1
        T2Diff = d3dPointsT2[i,:] - d3dPointsT2
        T1Dist = np.linalg.norm(T1Diff, axis=1)
        T2Dist = np.linalg.norm(T2Diff, axis=1)
        absDiff = abs(T2Dist - T1Dist)
        wIdx = np.where(absDiff < distDifference)
        W[i,wIdx] = 1
        cnt = np.sum(W[i,:])
        if cnt > maxc:
            maxc = cnt
            maxn = i
        cnt=0

    clique = [maxn]
    isin = True

    while True:
        potentialnodes = list()
        # Find potential nodes which are connected to all nodes in the clique
        for i in range(brojTacki):
            Wsub = W[i, clique]
            sumForIn = np.sum(Wsub)
            if sumForIn == len(clique):
                isin = True
            else:
                isin = False
                
            if isin == True and i not in clique:
                potentialnodes.append(i)
            isin=True

        cnt = 0
        maxn = 0
        maxc = 0
        # Find the node which is connected to the maximum number of potential nodes and store in maxn
        for i in range(len(potentialnodes)):
            Wsub = W[potentialnodes[i], potentialnodes]
            cnt = np.sum(Wsub)
            if cnt > maxc:
                maxc = cnt
                maxn = potentialnodes[i]
            cnt = 0
        if maxc == 0:
            break
        clique.append(maxn)

        if (len(clique) > 100):
            break

    return clique
