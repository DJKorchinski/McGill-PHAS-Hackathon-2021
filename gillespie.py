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
        
        self.L = L 
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def freeze_site(self,i,j):
        freeze((i,j),self.state,self.num_frozen_neighbours, self.L)


# @numba.njit(cache=True)
def freeze(ij,state,num_frozen_neighbours,L):
    i,j = ij[0],ij[1]
    state[i,j] = FROZEN_STATE
    # num_frozen_neighbours[i,j] = 0
    
    #now, change the count of frozen neighbours.
    if( i < L-1 ):
        num_frozen_neighbours[i+1,j] +=1
    if( i > 0 ):
        num_frozen_neighbours[i-1,j] +=1
    if( j < L-1 ):
        num_frozen_neighbours[i,j+1] +=1
    if( j > 0 ):
        num_frozen_neighbours[i,j-1] +=1


def evaporate(ij,state):
    i,j = ij[0],ij[1]
    state[i,j] = EVAPORATED_STATE

    
def step_state_gstate(numsteps,gstate):
    dt = step_state_numba(gstate.rng,numsteps,gstate.lambdas,gstate.state,gstate.num_frozen_neighbours,gstate.ind_to_state_ij,gstate.state_ij_to_ind)
    gstate.t+=dt 
    return dt

# @numba.njit(cache=True)
def step_state_numba(rng,numsteps, lambdas,state, num_frozen_neighbours, ind_to_state_ij, state_ij_to_ind):
    #lambda 1 and 2 are the spontaneous rates for evaporation and freezing respectivly.
    #lambda 3 are the 
    t_incr = 0
    L = np.shape(state)[0]
    N = np.size(state)
    for i in range(numsteps):
        #finding hte frozen sites, and number of edges along which frost can freeze other sites. Could be cached to speed this up.
        liquid_sites_mask = (state == LIQUID_STATE)
        
        # import pdb; pdb.set_trace()
        liquid_sites_indexes = np.arange(N)[ state_ij_to_ind[liquid_sites_mask] ]
        num_liquid = np.sum(liquid_sites_mask)
        num_freeze_attempts = np.sum(num_frozen_neighbours[liquid_sites_mask])
        #calculating the rates 
        rates = np.zeros(3)
        rates[0] =  lambdas[0] * num_liquid
        rates[1] =  lambdas[1] * num_liquid
        rates[2] =  lambdas[2] * num_freeze_attempts
        rate = np.sum(rates) 
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
                evaporate(ij,state)
            else:
                freeze(ij,state,num_frozen_neighbours,L)
        else: 
            #let's find the frozen site:
            attempt_number = rng.integers(num_freeze_attempts)
            liquid_index = np.searchsorted(np.cumsum(num_frozen_neighbours[liquid_sites_mask]),attempt_number,'right')
            index = liquid_sites_indexes[ liquid_index ]
            freeze(ind_to_state_ij[index],state,num_frozen_neighbours,L)

        t_incr += t_next 
    return t_incr