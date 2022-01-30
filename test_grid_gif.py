import gillespie
import numpy as np 
import matplotlib.pyplot  as plt 
import PIL as pillow
from matplotlib import cm
import time

def run_im(gs):
    gs[gs==2]=5
    gs[gs==1]=2
    gs[gs==0]=1
    gs[gs==5]=0
    Im=pillow.Image.fromarray(np.uint8(cm.cividis(gs/2)*255))
    gs[gs==0]=5
    gs[gs==1]=0
    gs[gs==2]=1
    gs[gs==5]=2
    return Im

images=[]

L = 128
l1=1
l2=0.025
l3=40
gstate = gillespie.grid_state(  np.array([l1, l2,l3]), L , 420)
#gstate.freeze_site(L//2, L//2)#freeze the central site.

t1 = time.time()

iter = 0
tcumulative = 0

#im=pillow.Image.fromarray(np.uint8(cm.hot(gstate.state/2)*255))
im=run_im(gstate.state)
images.append(im)

threshold=0.02

states=[]
timing=[]
while(True):
    dt = gillespie.step_state_gstate(1, gstate)
    tcumulative+= dt 
    #print(dt)
    print(tcumulative)
    if(tcumulative> threshold):
        print('in')
        # states.append(gstate.state)
        # timing.append(gstate.t)
        tcumulative = 0.0 
        #im=pillow.Image.fromarray(np.uint8(cm.hot(gstate.state/2)*255))
        im=run_im(gstate.state)
        images.append(im)
        # plt.imshow(gstate.state)
        # plt.savefig(str(int(100*gstate.t))+'.png')
        # plt.show()
        # plt.close()
    if(dt ==0):
        break 
    iter +=1 
    if(iter > L*L):
        print('TOO MANY ITERS, DEBUG THIS GARBAGE!')
        break 

t2 = time.time()
print('took: ',t2-t1)
print('frozen: ',np.sum(gstate.state == gillespie.FROZEN_STATE),'evaporated: ' ,np.sum(gstate.state == gillespie.EVAPORATED_STATE), 'liquid: ',np.sum(gstate.state == gillespie.LIQUID_STATE))

plt.imshow(gstate.state)
plt.colorbar()
plt.show()

#im=pillow.Image.fromarray(np.uint8(cm.hot(gstate.state/2)*255))
im=run_im(gstate.state)
images.append(im)
images[0].save('states_'+str(int(1000*l2))+'_'+str(int(l3))+'t_'+str(threshold)+'.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=80, loop=0)


#np.save('states_'+str(int(1000*l2))+'_'+str(int(l3))+'t_'+str(threshold)+'.npy',states)
#np.save('time_'+str(int(1000*l2))+'_'+str(int(l3))+'t_'+str(threshold)+'.npy',timing)