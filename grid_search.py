#!/usr/bin/env python3
import gillespie
import numpy as np
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from changing_lambda import evolving_lambda_sim

# space to search over
N = 100
lam2 = np.linspace(0.01, 0.2, N)
lam3 = np.linspace(0.1, 100, N)


def cluster_counter(gs):
    # take in a 128,128 grid, calculate clusters and output the cluster sizes
    # find the indexes of all the 2s
    ind = np.where(gs == 2)
    # convert to array
    ind = np.array(ind).T
    # do my dbscan
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
    max_c, n_c, c_size = cluster_counter(gstate)

    return c_size


def create_gstate_2(X):
    lam2, lam3 = X
    return evolving_lambda_sim(lam2, lam3)


my_gstates = np.empty((N, N, 128, 128), dtype=float)
P = Pool(48)
for i in range(N):
    # generate X_array to feed into create_gstate
    x_arr = []
    for j in range(N):
        x_arr.append((lam2[i], lam3[j]))
    my_gstates[i, :, :, :] = P.map(create_gstate_2, x_arr)

np.save("grid_search_2", {"lam2": lam2, "lam3": lam3, "g": my_gstates})
