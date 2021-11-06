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

#lets test the clustering
gstate = create_gstate([0.001,100])
plt.figure()
plt.imshow(gstate)
plt.colorbar()
print(cluster_counter(gstate))
import pdb; pdb.set_trace()
