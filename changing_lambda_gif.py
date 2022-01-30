#!/usr/bin/env python3

import gillespie
import numpy as np
import matplotlib.pyplot  as plt
import time
import PIL as pillow
from matplotlib import cm


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

#gstate.freeze_site(L//2, L//2)#freeze the central site.
def evolving_lambda_sim(lam3_init,lam2_init,t_interval=1e-3,L=128,A=20,m=1,seed=1337,plot_final = True,name=None):
    t1 = time.time()
    # m = 1
    iter = 0
    # A =20
    lam1 = m/lam3_init
    gstate = gillespie.grid_state(  np.array([lam1, lam2_init,lam3_init]), L , seed)
    #gstate.freeze_site(L//2, L//2)#freeze the central site.
    lambdas = gstate.lambdas
    images=[]
    im=run_im(gstate.state)
    images.append(im)

    tcumulative = 0.0 
    times = []
    areas = []
    states = []
    while(True):
        #need to update the parameters of gstate
        dt = gillespie.step_state_gstate(1, gstate)
        nf = np.sum(gstate.state == gillespie.FROZEN_STATE)
        #update lambda to penalise for lower humidty in air as things freeze
        gstate.lambdas[2] = gstate.lambdas[2]*np.exp(-(nf/L**2)*dt*A)
        gstate.lambdas[0] = m/gstate.lambdas[2]
        # print(gstate.lambdas[2])
        # print(dt)
        tcumulative += dt
        #print(dt)
        if(tcumulative > t_interval):
            #print('in')
            tcumulative = 0.
            times.append(gstate.t)
            areas.append([np.sum(gstate.state == i ) for i in range(2)])
            states.append(np.array(gstate.state))
            im=run_im(gstate.state)
            images.append(im)
        if(dt ==0):
            break
        iter +=1
        if(iter > L*L):
            print('TOO MANY ITERS, DEBUG THIS GARBAGE!')
            break

    t2 = time.time()

    if(plot_final):
        print('took: ',t2-t1)
        print('frozen: ',np.sum(gstate.state == gillespie.FROZEN_STATE),'evaporated: ' ,np.sum(gstate.state == gillespie.EVAPORATED_STATE), 'liquid: ',np.sum(gstate.state == gillespie.LIQUID_STATE))
        plt.imshow(gstate.state)
        plt.savefig(name)
        #plt.show()

    im=run_im(gstate.state)
    images.append(im)

    output_dic = { 'final_gstate':gstate, 'times':times, 'states':states, 'areas':areas }
    return output_dic, images

# if(__name__ == '__main__'):
#     evolving_lambda_sim(20,0.01)


#l2 vals:0.005, 0.005,  0.005,  0.01,   0.01,   0.02,   0.025,  0.025,  0.075,  0.125,  0.150,  0.150
#l3 vals:5,     10,     20,     5,      20,     10,     20,     40,     20,     80,     10,     80

l2=0.025
l3=20
threshold=0.01


output,images=evolving_lambda_sim(l3,l2,t_interval=threshold,L=128,A=20,m=1,seed=1337,plot_final = True,name='Fstates_'+str(int(1000*l2))+'_'+str(int(l3))+'t_'+str(threshold)+'.pdf')


images[0].save('Nstates_'+str(int(1000*l2))+'_'+str(int(l3))+'t_'+str(threshold)+'.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=80, loop=0)