#!/usr/bin/env python3
import gillespie
import numpy as np
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pickle
from dklib.param_manager import param_manager


def cluster_counter(gs):
    # take in a 128,128 grid, calculate clusters and output the cluster sizes
    # find the indexes of all the 2s
    ind = np.where(gs == 2)
    # convert to array
    ind = np.array(ind).T
    # do my dbscan
    # print(ind)
    # plt.imshow(gs)
    # plt.show()
    clustering = DBSCAN(eps=1, min_samples=1).fit(ind)
    labels = clustering.labels_
    # make some plots of clusters
    c_size = []
    for l in set(labels):
        # mask = (labels==l)
        # points = ind[mask]
        # X,Y = points.T
        # plt.scatter(Y,X,marker='.',s=1)
        c_size.append(np.sum(labels == l))
    n = len(set(labels))
    return np.max(c_size), n, c_size


def create_gstate(X):
    # assign lam2 and lam3
    lam2, lam3, L, seed = X
    print(X)
    gstate = gillespie.grid_state(np.array([1, lam2, lam3]), L, seed)
    i = 0
    while True:
        dt = gillespie.step_state_gstate(1, gstate)
        if dt == 0:
            break
        i += 1
        if i > L * L:
            print("TOO MANY ITERS, DEBUG THIS GARBAGE!")
            break
    print(
        "frozen: ",
        np.sum(gstate.state == gillespie.FROZEN_STATE),
        "evaporated: ",
        np.sum(gstate.state == gillespie.EVAPORATED_STATE),
        "liquid: ",
        np.sum(gstate.state == gillespie.LIQUID_STATE),
        X,
    )
    # calculate clusters
    # print(gstate.state)
    max_c, n_c, c_size = cluster_counter(gstate.state)

    return c_size

if __name__ == "__main__":
    # space to search over
    N = 10
    lam2s = np.geomspace(0.01, 0.2, N)
    lam3s = np.geomspace(0.01, 0.02, N)
    Ls = np.geomspace(32, 1024, 6, True, dtype=int)
    Nrep = 10
    seeds = np.arange(Nrep)*420+1

    pm = param_manager([lam2s, lam3s, Ls, seeds])

    P = Pool(48)
    param_array = [pm.get_params(i) for i in range(pm.REP_TOTAL)]
    cluster_sizes = P.map(create_gstate,param_array)
    output_dic = {
        "lam2s": lam2s,
        "lam3s": lam3s,
        "Ls": Ls,
        "param_manager": pm,
        "cluster_sizes": cluster_sizes,
    }

    with open("data/simulation_01.dat", "wb") as fh:
        pickle.dump(fh, output_dic)
