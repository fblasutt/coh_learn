# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:57:00 2020

@author: Fabio
"""

#Simulate kalman filter and get error mean and variances





if __name__ == '__main__':   
    
    
    import numpy as np
    from statutils import kalman
    import matplotlib.pyplot as plt
    
    
    #Set parameters
    N=100000 #realizations
    T=10 #length of time series
    
    #set the seed
    np.random.seed(19)
   
    #parameters of the kalman filter
    sigma0=0.057529296875000004
    sigmae=0.0#0.030224609375
    sigmamu=0.057529296875000004*2.170234375#1.1#0.32880859375000004
    
    
    #initialize arrays
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
    
    #Get the kalman gain from external routine
    K,sigmav=kalman(1.0,sigmae**2,sigmamu**2,sigma0**2,T)
        
    #Actual simulation!!
    for i in range(T):
        true[:,i]=shocke[:,i] if i==0 else true[:,i-1]+shocke[:,i]
        noise[:,i]=true[:,i]+shockmu[:,i]
        pred[:,i]=0 if i==0 else upred[:,i-1]
        upred[:,i]=pred[:,i]+K[i]*(noise[:,i]-pred[:,i]) 
        error[:,i]=np.absolute(upred[:,i]-true[:,i])
    
    
    #Get some interesting moments from the simulation
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
    

