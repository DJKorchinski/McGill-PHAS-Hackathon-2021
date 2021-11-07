import numpy as np
import matplotlib.pyplot as plt

states=np.load('grid_search.npy',allow_pickle=1).tolist()
g=states['g']
lam2=states['lam2']
lam3=states['lam3']





n=11
l2v=np.asarray([0,10,20,30,40,50,60,70,80,90,99])
l3v=np.asarray([0,10,20,30,40,50,60,70,80,90,99])

#n=6
#l2v=np.asarray([8,9,10,12,16,21])
#l3v=np.asarray([99,20,15,10,5,1])

fig,axs=plt.subplots(n,n,sharex='all',sharey='all')


for i in range(0,n):
    for j in range(0,n):
        axs[i,j].imshow(g[l2v[i],l3v[j],:,:])
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

plt.tight_layout(pad=0,w_pad=0,h_pad=0)
plt.show()