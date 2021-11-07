#!/usr/bin/env python3
#import gillespie
import numpy as np
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def cluster_counter(gs):
    #take in a 128,128 grid, calculate clusters and output the cluster sizes
    # find the indexes of all the 2s
    ind = np.where(gs==2)
    #convert to array
    ind = np.array(ind).T
    #do my dbscan
    clustering = DBSCAN(eps=1, min_samples=1).fit(ind)
    labels = clustering.labels_
    #make some plots of clusters
    c_size = []
    for l in set(labels):
        # mask = (labels==l)
        # points = ind[mask]
        # X,Y = points.T
        # plt.scatter(Y,X,marker='.',s=1)
        c_size.append(np.sum(labels==l))
    return np.max(c_size)

#load my gstates
states = np.load("grid_search.npy",allow_pickle=1).tolist()
#need to do cluster size counting
lam2 = states['lam2']
lam3 = states['lam3']
g = states['g']
max_cluster = np.empty((len(lam2),len(lam3)))
p = Pool(50)
for i in range(len(lam2)):
    gs_arr = []
    for j in range(len(lam3)):
        gs_arr.append(g[i,j,:,:])
    # for i in gs_arr:
        # print(cluster_counter(i))
    max_cluster[i,:] = p.map(cluster_counter,gs_arr)
    print(i)
p.close()
X,Y = np.meshgrid(lam2,lam3)
plt.pcolormesh(X,Y,max_cluster)
plt.xlabel("lambda2")
plt.ylabel("lambda3")
plt.show()
import pdb; pdb.set_trace()
