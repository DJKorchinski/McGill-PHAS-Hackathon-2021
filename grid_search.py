#!/usr/bin/env python3
import gillespie
import numpy as np
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#space to search over
N = 10
lam2s = np.geomspace(0.01,0.2,N)
lam3s = np.geomspace(0.01,.02,N)
Ls = np.geomspace(32,1024,6,True)
Nrep = 10
repnos = np.arange(Nrep)

from dklib.param_manager import param_manager
pm = param_manager(lam2s, lam3s, Ls, repnos)

def create_gstate(X):
    #assign lam2 and lam3
    print(X)
    lam2,lam3 = X
    L = 128
    gstate = gillespie.grid_state(  np.array([1, lam2, lam3]), L , 420)
    iter = 0
    while(True):
        dt = gillespie.step_state_gstate(1, gstate)
        if(dt ==0):
            break
        iter +=1
        if(iter > L*L):
            print('TOO MANY ITERS, DEBUG THIS GARBAGE!')
            break
    print('frozen: ',np.sum(gstate.state == gillespie.FROZEN_STATE),'evaporated: ' ,np.sum(gstate.state == gillespie.EVAPORATED_STATE), 'liquid: ',np.sum(gstate.state == gillespie.LIQUID_STATE),X)
    return gstate.state


P = Pool(48)
param_array = [pm.get_params(i) for i in range(pm.REP_TOTAL)]
cluster_sizes = P.map(create_gstate(param_array)
output_dic = {'lam2s':lam2s,'lam3s':lam3s,'Ls':Ls,'param_manager':pm, 'cluster_sizes':cluster_sizes}
import pickle
with open('data/simulation_01.dat','wb') as fh:  
    pickle.dump(fh, output_dic)
