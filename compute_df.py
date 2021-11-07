import numpy as np

def df(data, Ls = None):
    #check data for non-zero values.
    if(np.sum(data == 0) + np.sum(data ==1) < np.size(data)):
        print("Clean your data you filthy animal! Data should be 0 or 1 ")
        return 
    Ns = []
    if(Ls is None):
        Ls = np.arange(8,2,-1)
    for L in Ls: 
        Ns.append(countN(data,L)[0])
    coeffs = np.polyfit(np.log(Ls), np.log(Ns), 1)
    slope = coeffs[0]
    return slope, (Ls,Ns,coeffs)


def countN(dat,bsize):
    result=np.add.reduceat(np.add.reduceat(dat,np.arange(0,dat.shape[0],bsize),axis=0),np.arange(0,dat.shape[1],bsize),axis=1)
    tot_zero=np.count_nonzero(result==0)
    tot_full=np.count_nonzero(result==bsize*bsize)
    N=int(len(result)**2)-tot_zero-tot_full
    N2=np.count_nonzero(result)
    return N,N2#,result
