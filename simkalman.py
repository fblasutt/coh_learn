# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:57:00 2020

@author: Fabio
"""

#Simulate kalman filter and get error mean and variances





if __name__ == '__main__':   
    
    
    import numpy as np
    from mc_tools import mc_simulate, int_prob,int_proba
    from gridvec import VecOnGrid
    from statutils import kalman
    import matplotlib.pyplot as plt
    
    N=1000000
    T=10
    np.random.seed(19)
   
    sigma0=1.0
    sigmae=0.030224609375
    sigmamu=1.0#0.32880859375000004
    
    shocke0=np.random.normal(0.0, sigma0, N)
    shockmu=np.reshape(np.random.normal(0.0, sigmamu, N*T),(N,T))
    shocke=np.reshape(np.random.normal(0.0, sigmae, N*T),(N,T))
    shocke[:,0]=shocke0
    np.random.seed(8)
    initial=np.random.normal(0.0, sigmamu, N)
    
    true=np.ones(shocke.shape)*-1
    noise=np.ones(shocke.shape)*-1
    pred=np.ones(shocke.shape)*-1
    upred=np.ones(shocke.shape)*-1
    error=np.ones(shocke.shape)*-1
    
    K,sigmav=kalman(1.0,sigmae**2,sigmamu**2,sigma0**2,T)
        
    #Initialize first period
    for i in range(T):
        true[:,i]=shocke[:,i] if i==0 else true[:,i-1]+shocke[:,i]
        noise[:,i]=true[:,i]+shockmu[:,i]
        pred[:,i]=0 if i==0 else upred[:,i-1]
        upred[:,i]=pred[:,i]+K[i]*(noise[:,i]-pred[:,i]) 
        error[:,i]=np.absolute(upred[:,i]-true[:,i])
    
    trueu=true[:,1:]-true[:,0:-1]
    nex=np.zeros(true.shape)
    nex[:,1:]=upred[:,:T-1]
    upredu=nex-upred
    
    stdint=np.sqrt(K**2*(sigmav**2+sigmamu**2))
    
    print(stdint-np.std(upredu,axis=0))
    print(stdint)
    print(np.std(upredu,axis=0))
    print(np.mean(error,axis=0))
    pos0=(shockmu[:,0]>=0.0)
    pose0=(shocke[:,0]>=0.0)
    pos1=(shockmu[:,1]>=0.0)
    pose1=(shocke[:,1]>=0.0)
    pos2=(shockmu[:,2]>=0.0)
    pose2=(shocke[:,2]>=0.0)
    innov=(noise-pred>=0.0)
    

