import numpy as np
import numpy.random 
import numba 

LIQUID_STATE = 0
EVAPORATED_STATE = 1
FROZEN_STATE = 2

class grid_state:
    def __init__(self,lambdas, L,seed = 1337):
        self.lambdas = lambdas 
        self.state = np.zeros(shape = (L,L),dtype = int) #0 = liquid, 1 = evaporated, 2 = frozen.
        self.num_frozen_neighbours = np.zeros(shape = (L*L),dtype = int )
        self.ind_to_state_ij = np.zeros( (L*L ,2)  , dtype = int )
        self.state_ij_to_ind = np.zeros( (L,L) , dtype = int) 
        tot = 0
        for i in range(L):
            for j in range(L):
                self.ind_to_state_ij[tot,:] = np.array([i,j])
                self.state_ij_to_ind[i,j] = tot 
                tot +=1 
        
        self.liquid_sites_indexes = np.arange(L*L) #only sensible between 0, num_liquid_site

        self.liquid_sites_mask = np.ones((L,L),dtype = np.bool)
        self.num_liquid_sites = L*L 
        self.num_freeze_attempts = 0

        self.L = L 
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def freeze_site(self,i,j):
        global_index = self.state_ij_to_ind[i,j]
        liquid_index = np.where(self.liquid_sites_indexes == global_index)
        # import pdb; pdb.set_trace()
        if(liquid_index[0].size > 0):
            liquid_index = liquid_index[0][0]
            self.num_liquid_sites , self.num_freeze_attempts =  freeze(liquid_index,self.state,self.num_frozen_neighbours,self.liquid_sites_mask,self.ind_to_state_ij,self.state_ij_to_ind,self.liquid_sites_indexes, self.L,self.num_liquid_sites,self.num_freeze_attempts)

# @profile
@numba.njit(cache=True)
def pop_liquid_site_ON(liquid_site_index,liquid_inds,i_num_liquid):
    liquid_inds[liquid_site_index:i_num_liquid-1] = liquid_inds[liquid_site_index+1:i_num_liquid]
    
@numba.njit(cache=True)
def pop_liquid_site(liquid_site_index,liquid_inds,i_num_liquid):
    #O(1) version: 
    liquid_inds[liquid_site_index] = liquid_inds[i_num_liquid-1]
    
    
@numba.njit(cache=True)
def freeze(liquid_site_index,state,num_frozen_neighbours,liquid_sites_mask,state_ind_to_ij,ij_to_site_ind,liquid_inds, L,i_num_liquid,i_num_freeze_attempts):
    global_index =liquid_inds[liquid_site_index]
    ij = state_ind_to_ij[global_index]
    i,j = ij[0],ij[1]
    state[i,j] = FROZEN_STATE
    liquid_sites_mask[i,j] = False

    #now, change the count of frozen neighbours.
    dfreeze_attempts = 0 
    if( i < L-1 ):
        num_frozen_neighbours[ij_to_site_ind[i+1,j]] +=1
        dfreeze_attempts += liquid_sites_mask[i+1,j]
    if( i > 0 ):
        num_frozen_neighbours[ij_to_site_ind[i-1,j]] +=1
        dfreeze_attempts += liquid_sites_mask[i-1,j]
    if( j < L-1 ):
        num_frozen_neighbours[ij_to_site_ind[i,j+1]] +=1
        dfreeze_attempts += liquid_sites_mask[i,j+1]
    if( j > 0 ):
        num_frozen_neighbours[ij_to_site_ind[i,j-1]] +=1
        dfreeze_attempts += liquid_sites_mask[i,j-1]

    pop_liquid_site(liquid_site_index,liquid_inds,i_num_liquid)

    num_liquid = i_num_liquid - 1
    num_freeze_attempts = i_num_freeze_attempts - num_frozen_neighbours[global_index] + dfreeze_attempts
    return num_liquid, num_freeze_attempts

@numba.njit(cache=True)
def evaporate(liquid_site_index,state,num_frozen_neighbours,liquid_sites_mask,state_ind_to_ij,liquid_inds,i_num_liquid,i_num_freeze_attempts):
    global_index = liquid_inds[liquid_site_index]
    ij = state_ind_to_ij[global_index]
    i,j = ij[0],ij[1]
    state[i,j] = EVAPORATED_STATE
    liquid_sites_mask[i,j] = False

    #removing the liquid site.
    pop_liquid_site(liquid_site_index,liquid_inds,i_num_liquid)

    return i_num_liquid - 1, i_num_freeze_attempts - num_frozen_neighbours[global_index]

    
def step_state_gstate(numsteps,gstate):
    dt,num_liquid,num_freeze_attempts = step_state_numba(gstate.rng,numsteps,gstate.lambdas,gstate.state,gstate.num_frozen_neighbours,\
        gstate.ind_to_state_ij,gstate.state_ij_to_ind, gstate.liquid_sites_mask, gstate.liquid_sites_indexes, \
            gstate.num_liquid_sites, gstate.num_freeze_attempts)
    gstate.t+=dt 
    gstate.num_liquid_sites = num_liquid  
    gstate.num_freeze_attempts = num_freeze_attempts
    #check for safety: 
    # if(not np.sum(gstate.state == 0) == num_liquid):
    #     print('broken!')
    # if(not (num_freeze_attempts == np.sum(gstate.num_frozen_neighbours[gstate.state_ij_to_ind[gstate.liquid_sites_mask]]))):
    #     print('borked 2')
    return dt

# @numba.njit(cache=True)
# @numba.njit(cache=True)
def step_state_numba(rng,numsteps, lambdas,state, num_frozen_neighbours, ind_to_state_ij, state_ij_to_ind,liquid_sites_mask,liquid_inds,i_num_liquid,i_num_freeze_attempts):
    #lambda 1 and 2 are the spontaneous rates for evaporation and freezing respectivly.
    #lambda 3 are the 
    t_incr = 0
    L = np.shape(state)[0]
    N = np.size(state)
    num_liquid = i_num_liquid
    num_freeze_attempts = i_num_freeze_attempts
    for i in range(numsteps):
        #finding hte frozen sites, and number of edges along which frost can freeze other sites. Could be cached to speed this up.
        # liquid_sites_mask = (state == LIQUID_STATE)
        
        # Nrange = np.arange(N)
        # liquid_sites_indexes = Nrange[ state_ij_to_ind[liquid_sites_mask] ]
        # num_liquid = np.sum(liquid_sites_mask)
        # num_freeze_attempts = np.sum(num_frozen_neighbours[liquid_sites_mask])

        #calculating the rates 
        rates = np.zeros(3)
        rates[0] =  lambdas[0] * num_liquid
        rates[1] =  lambdas[1] * num_liquid
        rates[2] =  lambdas[2] * num_freeze_attempts
        rate = rates[0] + rates[1] + rates[2]
        
        if(rate == 0):
            #everything is frozen or evaporated.
            break
        
        t_next = np.log(1-rng.random()) / rate
        activation_u = rng.random() 
        activation_type = -1
        while(activation_u > 0):
            activation_type += 1
            activation_u -= rates[activation_type] / rate 
        
        #then, do the activation. For types 0,1 it's a spontaneous freezing / evaporation event. 
        if(activation_type == 0 or activation_type == 1):
            liquid_index = rng.integers(num_liquid)
            # index = liquid_inds[]
            # ij = ind_to_state_ij[index]
            
            #TODO: update the edge state.
            if(activation_type == 0):
                num_liquid,num_freeze_attempts = evaporate(liquid_index,state,num_frozen_neighbours,liquid_sites_mask,ind_to_state_ij,liquid_inds,num_liquid,num_freeze_attempts)
            else:
                num_liquid,num_freeze_attempts = freeze(liquid_index,state,num_frozen_neighbours,liquid_sites_mask,ind_to_state_ij,state_ij_to_ind,liquid_inds,L,num_liquid,num_freeze_attempts)
        else: 
            #let's find the frozen site:
            attempt_number = rng.integers(num_freeze_attempts)
            # import pdb; pdb.set_trace()

            # liquid_ijs = ind_to_state_ij[liquid_inds[:i_num_liquid]]
            num_frozen_neighbours_of_liquid_sites = num_frozen_neighbours[liquid_inds[:i_num_liquid]]
            cumsums = np.cumsum(num_frozen_neighbours_of_liquid_sites)
            liquid_index = np.searchsorted(cumsums,attempt_number,'right')
            num_liquid,num_freeze_attempts =freeze(liquid_index,state,num_frozen_neighbours,liquid_sites_mask,ind_to_state_ij,state_ij_to_ind,liquid_inds,L,num_liquid,num_freeze_attempts)

        t_incr += t_next 
    return t_incr, num_liquid,num_freeze_attempts