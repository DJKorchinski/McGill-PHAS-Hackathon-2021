import gillespie
import numpy as np 
import matplotlib.pyplot  as plt 

L = 128
gstate = gillespie.grid_state(  np.array([1, 0.1,100]), L , 420)
#gstate.freeze_site(L//2, L//2)#freeze the central site.

iter = 0
while(True):
    dt = gillespie.step_state_gstate(1, gstate)
    if(dt ==0):
        break 
    iter +=1 
    if(iter > L*L):
        print('TOO MANY ITERS, DEBUG THIS GARBAGE!')
        break 

print('frozen: ',np.sum(gstate.state == gillespie.FROZEN_STATE),'evaporated: ' ,np.sum(gstate.state == gillespie.EVAPORATED_STATE), 'liquid: ',np.sum(gstate.state == gillespie.LIQUID_STATE))

# plt.imshow(gstate.state)
# plt.show()
