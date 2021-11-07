#!/usr/bin/env python3

import gillespie
import numpy as np
import matplotlib.pyplot  as plt
import time


#gstate.freeze_site(L//2, L//2)#freeze the central site.
def evolving_lambda_sim(lam3_init,lam2_init,t_interval=1e-3,L=128,A=20,m=1,seed=1337,plot_final = True):
    t1 = time.time()
    # m = 1
    iter = 0
    # A =20
    lam1 = m/lam3_init
    gstate = gillespie.grid_state(  np.array([lam1, lam2_init,lam3_init]), L , seed)
    lambdas = gstate.lambdas

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
        if(tcumulative > t_interval):
            tcumulative = 0.
            times.append(gstate.t)
            areas.append([np.sum(gstate.state == i ) for i in range(3)])
            states.append(np.array(gstate.state))

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
        plt.show()

    output_dic = { 'final_gstate':gstate, 'times':times, 'states':states, 'areas':areas }
    return output_dic

if(__name__ == '__main__'):
    evolving_lambda_sim(20,0.01)
