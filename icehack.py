import gillespy2 as gp  
import numpy as np 
import matplotlib.pyplot as plt 


#given L by L box of numbers 1 or 2 (evaporated or frozen)


#box counting dimension
#D_0=lim(eps to 0) (log(N(eps))/log(1/eps))
#eps is the box size, N the number of boxes that cover the pattern

#I think one way to do this would be to lay down boxes that cover the whole pattern and then if any of the boxes have a number in them they cound and if they are empty then they don't count

def read_data(filename):
    dat=np.load(filename)
    print(dat.shape)
    return data

def countN(dat,bsize):
    result=np.add.reduceat(np.add.reduceat(dat,np.arange(0,dat.shape[0],bsize),axis=0),np.arange(0,dat.shape[1],bsize),axis=1)
    N=np.count_nonzero(result)
    return N#,result

def getD0(N,bsize):
    return np.log(N)/np.log(1/bsize)

def getD0_tot(dat):
    pow2=np.log2(dat.shape[0])
    scl=np.arange(0,pow2)
    bsize=np.array(dat.shape[0]/2**scl,dtype=int)
    Ns=np.empty_like(bsize)
    for i in range(len(bsize)):
        Ns[i]=countN(dat,bsize[i])
    D0s=getD0(Ns,bsize)
    return D0s, Ns, bsize


