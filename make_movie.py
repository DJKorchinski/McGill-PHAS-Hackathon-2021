#!/usr/bin/env python3
#Assume that Daniel has passed me a matrix of NxM , each with timesteps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
def make_movie(freeze_array):
    freeze_array = np.array(freeze_array)
    n=1

    x,y,z = freeze_array.shape
    X = np.linspace(1,x,x)
    Y = np.linspace(1,y,y)
    for i in range(z):
        fig, ax = plt.subplots()
        current_freeze = freeze_array[:,:,i]
        psm = ax.pcolormesh(Y,X,current_freeze)
        plt.savefig(str(i)+'.png')
        plt.show()
        plt.close()

sample_freeze = np.array([[1,0,1],[0,1,0]])
sample_freeze = np.stack((sample_freeze,sample_freeze), axis=2)
make_movie(sample_freeze)
