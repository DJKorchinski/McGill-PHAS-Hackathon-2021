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
    n = len(set(labels))
    return np.max(c_size),n,c_size

#load my gstates
states = np.load("grid_search.npy",allow_pickle=1).tolist()
#need to do cluster size counting
lam2 = states['lam2']
lam3 = states['lam3']
g = states['g']
max_cluster = np.empty((len(lam2),len(lam3)))
num_cluster = np.empty((len(lam2),len(lam3)))
cluster_sizes = np.empty((len(lam2),len(lam3)),dtype = object)

p = Pool(50)
for i in range(len(lam2)):
    gs_arr = []
    for j in range(len(lam3)):
        gs_arr.append(g[i,j,:,:])
    # for i in gs_arr:
        # print(cluster_counter(i))
    store = p.map(cluster_counter,gs_arr)
    for k,item in enumerate(store):
        max_cluster[i,k],num_cluster[i,k],cluster_sizes[i,k] = item
    print(i)
p.close()
X,Y = np.meshgrid(lam2,lam3)
plt.pcolormesh(X,Y,max_cluster/128**2,shading='auto')
plt.xlabel(r"$\lambda_2$")
plt.ylabel(r"$\lambda_3$")
cbar = plt.colorbar()
cbar.set_label('Max size of frozen patches', rotation=270,labelpad = 10)
plt.show()
X,Y = np.meshgrid(lam2,lam3)
plt.pcolormesh(X,Y,num_cluster,shading='auto')
plt.xlabel(r"$\lambda_2$")
plt.ylabel(r"$\lambda_3$")
cbar2 = plt.colorbar()
cbar2.set_label('Number of frozen patches', rotation=270,labelpad = 10)
plt.show()
# np.save('clusters',{"max":max_cluster,"num":num_cluster,"cluster_size":cluster_sizes,'lam2':X,'lam3':Y})
import pdb; pdb.set_trace()
