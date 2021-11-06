import gillespy2 as gp  
import numpy as np 
import matplotlib.pyplot as plt 
import sys

##to run type icehack.py filename.npy bool cutvalue
#where filename.npy is the data file, bool is True or False (or anything not True really) as to whether the data needs to be cut up and cutval is the size of the final data product (should be a power of 2)
#eg. icehack.py identity512.npy True 256
#the third value does nothing if bool!='True'


#given L by L box of numbers 1 or 2 (evaporated or frozen)

#box counting dimension
#D_0=lim(eps to 0) (log(N(eps))/log(1/eps))
#eps is the box size, N the number of boxes that cover the pattern

def read_data(filename):
    dat=np.load(filename)
    #print(dat.shape)
    return dat

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

def cutDat(cutv,dat):
    centerPix=int(dat.shape[0]/2)
    centerPiy=int(dat.shape[1]/2)
    print(centerPix,centerPiy,cutv)
    out=dat[int(centerPix-cutv/2):int(centerPix+cutv/2),int(centerPiy-cutv/2):int(centerPiy+cutv/2)]
    print(out.shape)
    return out

#filename='identity512.npy'

dat=read_data(sys.argv[1])

print(sys.argv)

if sys.argv[2]=='True':
    print('in loop')
    dat=cutDat(int(sys.argv[3]),dat)

d,n,L=getD0_tot(dat)

plt.plot(np.log(n),np.log(1/L))
plt.show()

slopevals=(np.log(n[0:-2])-np.log(n[-1]))/(np.log(1/L[0:-2])-np.log(1/L[-1]))

print('slope=',slopevals)