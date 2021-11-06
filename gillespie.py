import numpy as np
import numpy.random 
# import numba 

LIQUID_STATE = 0
EVAPORATED_STATE = 1
FROZEN_STATE = 2

class grid_state:
    def __init__(self,lambdas, L,seed = 1337):
        self.lambdas = lambdas 
        self.state = np.zeros(shape = (L,L),dtype = int) #0 = liquid, 1 = evaporated, 2 = frozen.
        self.num_frozen_neighbours = np.zeros(shape = (L,L),dtype = int )
        self.ind_to_state_ij = np.zeros( (L*L ,2)  , dtype = int )
        self.state_ij_to_ind = np.zeros( (L,L) , dtype = int) 
        tot = 0
        for i in range(L):
            for j in range(L):
                self.ind_to_state_ij[tot,:] = np.array([i,j])
                self.state_ij_to_ind[i,j] = tot 
                tot +=1 
        
        # self.liquid_sites_indexes = np.arange(L*L) #only sensible between 0, num_liquid_site
        # self.site_ind_to_liquid_site_ind_ind = np.arange(L*L) # maps from N -> (0,n) where n is num_liquid_sites, only if i is one of the liquid sites.

        self.liquid_sites_mask = np.ones((L,L),dtype = np.bool)
        self.num_liquid_sites = L*L 
        self.num_freeze_attempts = 0

        self.L = L 
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def freeze_site(self,i,j):
        self.num_liquid_sites , self.num_freeze_attempts =  freeze((i,j),self.state,self.num_frozen_neighbours, self.L)


# def pop_liquid_site(site_index,liquid_inds,site_ind_to_liquid_site_ind_ind,i_num_liquid):
#     #call this, before reducing i_num_liquid by one!
#     liquid_inds_ind = site_ind_to_liquid_site_ind_ind[site_index]
#     last_liquid_inds_ind = i_num_liquid-1
#     last_liquid_site = liquid_inds[last_liquid_inds_ind]
#     #moving last liquid site to the new position.
#     liquid_inds[liquid_inds_ind] = last_liquid_site
#     site_ind_to_liquid_site_ind_ind[last_liquid_site] = liquid_inds_ind
#     #moving our old site to the end of the list.
#     liquid_inds[last_liquid_inds_ind] = site_index
#     site_ind_to_liquid_site_ind_ind[site_index]  = last_liquid_inds_ind

# @numba.njit(cache=True)
def freeze(ij,state,num_frozen_neighbours,liquid_sites_mask, L,i_num_liquid,i_num_freeze_attempts):
    i,j = ij[0],ij[1]
    state[i,j] = FROZEN_STATE
    liquid_sites_mask[i,j] = False

    #now, change the count of frozen neighbours.
    dfreeze_attempts = 0 
    if( i < L-1 ):
        num_frozen_neighbours[i+1,j] +=1
        dfreeze_attempts += liquid_sites_mask[i+1,j]
    if( i > 0 ):
        num_frozen_neighbours[i-1,j] +=1
        dfreeze_attempts += liquid_sites_mask[i-1,j]
    if( j < L-1 ):
        num_frozen_neighbours[i,j+1] +=1
        dfreeze_attempts += liquid_sites_mask[i,j+1]
    if( j > 0 ):
        num_frozen_neighbours[i,j-1] +=1
        dfreeze_attempts += liquid_sites_mask[i,j-1]

    # site_index = state_ij_to_ind
    # pop_liquid_site()

    num_liquid = i_num_liquid - 1
    num_freeze_attempts = i_num_freeze_attempts - num_frozen_neighbours[i,j] + dfreeze_attempts
    return num_liquid, num_freeze_attempts


def evaporate(ij,state,num_frozen_neighbours,liquid_sites_mask,i_num_liquid,i_num_freeze_attempts):
    i,j = ij[0],ij[1]
    state[i,j] = EVAPORATED_STATE
    liquid_sites_mask[i,j] = False
    return i_num_liquid - 1, i_num_freeze_attempts - num_frozen_neighbours[i,j]

    
def step_state_gstate(numsteps,gstate):
    dt,num_liquid,num_freeze_attempts = step_state_numba(gstate.rng,numsteps,gstate.lambdas,gstate.state,gstate.num_frozen_neighbours,\
        gstate.ind_to_state_ij,gstate.state_ij_to_ind,gstate.liquid_sites_mask,gstate.num_liquid_sites, gstate.num_freeze_attempts)
    gstate.t+=dt 
    gstate.num_liquid_sites = num_liquid  
    gstate.num_freeze_attempts = num_freeze_attempts
    #check for safety: 
    if(not np.sum(gstate.state == 0) == num_liquid):
        print('broken!')
    if(not (num_freeze_attempts == np.sum(gstate.num_frozen_neighbours[gstate.liquid_sites_mask]))):
        print('borked 2')
    return dt

# @numba.njit(cache=True)
# @profile 
def step_state_numba(rng,numsteps, lambdas,state, num_frozen_neighbours, ind_to_state_ij, state_ij_to_ind,liquid_sites_mask,i_num_liquid,i_num_freeze_attempts):
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
        
        Nrange = np.arange(N)
        liquid_sites_indexes = Nrange[ state_ij_to_ind[liquid_sites_mask] ]
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

            index = liquid_sites_indexes[rng.integers(num_liquid)]
            ij = ind_to_state_ij[index]
            
            #TODO: update the edge state.
            if(activation_type == 0):
                num_liquid,num_freeze_attempts = evaporate(ij,state,num_frozen_neighbours,liquid_sites_mask,num_liquid,num_freeze_attempts)
            else:
                num_liquid,num_freeze_attempts = freeze(ij,state,num_frozen_neighbours,liquid_sites_mask,L,num_liquid,num_freeze_attempts)
        else: 
            #let's find the frozen site:
            attempt_number = rng.integers(num_freeze_attempts)
            cumsums = np.cumsum(num_frozen_neighbours[liquid_sites_mask])
            liquid_index = np.searchsorted(cumsums,attempt_number,'right')
            index = liquid_sites_indexes[ liquid_index ]
            num_liquid,num_freeze_attempts = freeze(ind_to_state_ij[index],state,num_frozen_neighbours,liquid_sites_mask,L,num_liquid,num_freeze_attempts)

        t_incr += t_next 
    return t_incr, num_liquid,num_freeze_attempts