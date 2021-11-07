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
    tot_zero=np.count_nonzero(result==0)
    tot_full=np.count_nonzero(result==bsize*bsize)
    N=int(len(result)**2)-tot_zero-tot_full
    N2=np.count_nonzero(result)
    return N,N2#,result

def getD0(N,bsize):
    return np.log(N)/np.log(1/bsize)

def getD0_tot(dat):
    #pow2=np.log2(dat.shape[0])
    #scl=np.arange(0,pow2)
    #bsize=np.array(dat.shape[0]/2**scl,dtype=int)
    bsize=np.asarray([1,2,3,5,8,13,21,34,55,89,144]) #233
    Ns=np.empty_like(bsize)
    Ns2=np.empty_like(bsize)
    for i in range(len(bsize)):
        Ns[i],Ns2[i]=countN(dat,bsize[i])
    D0s=getD0(Ns,bsize)
    return D0s, Ns, Ns2, bsize

def cutDat(cutv,dat):
    centerPix=int(dat.shape[0]/2)
    centerPiy=int(dat.shape[1]/2)
    print(centerPix,centerPiy,cutv)
    out=dat[int(centerPix-cutv/2):int(centerPix+cutv/2),int(centerPiy-cutv/2):int(centerPiy+cutv/2)]
    print(out.shape)
    return out

#filename='identity512.npy'

dat=read_data(sys.argv[1])

#print(sys.argv)

if sys.argv[2]=='True':
    print('in loop')
    dat=cutDat(int(sys.argv[3]),dat)

d,n,n2,L=getD0_tot(dat)

slopevals=(np.log(n[0:-1])-np.log(n[1:]))/(np.log(1/L[0:-1])-np.log(1/L[1:]))
print('slope (edges)=',slopevals)

slopevals2=(np.log(n2[0:-1])-np.log(n2[1:]))/(np.log(1/L[0:-1])-np.log(1/L[1:]))
print('slope (cover)=',slopevals2)

p=np.polyfit(np.log(1/L),np.log(n),1)
print('Edges deg',p)

p2=np.polyfit(np.log(1/L),np.log(n2),1)
print('Cover deg',p2)

plt.imshow(dat)
plt.show()

plt.plot(np.log(1/L),np.log(n),label='Edges')
plt.plot(np.log(1/L),np.log(n2),label='Cover')
#plt.plot(np.log(1/L),np.log(1/L)*p[0]+p[1],label='Edge BF')
#plt.plot(np.log(1/L),np.log(1/L)*p2[0]+p2[1],label='Cover BF')
plt.legend()
plt.show()

