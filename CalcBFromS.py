import numpy as np 


def B(Svals,L):
    g=0
    sumSvals=0
    sumSvalssq=0

    for i in range(len(Svals)):
        #g=g+np.max(Svals[i])
        ind=np.argmax(Svals[i])
        g=g+Svals[i][ind]
        sumSvals=sumSvals+(np.sum(Svals[i][:ind])+np.sum(Svals[i][ind+1:]))/len(Svals[i]-1)
        sumSvalssq=sumSvalssq+(np.sum(Svals[i][:ind]**2)+np.sum(Svals[i][ind+1:]**2))/len(Svals[i]-1)

    g=g/L**2
    #g=g/len(Svals)/L**2
    #sumSvals=sumSvals/len(Svals)
    #sumSvalssq=sumSvalssq/len(Svals)

    B=g*sumSvalssq/sumSvals**2






