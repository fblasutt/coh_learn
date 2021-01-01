#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aCreated on Tue Sep 17 19:14:08 2019
 
@author: Egor Kozlov
"""
 
 
 
 
 
if __name__ == '__main__':
     
    try:
        from IPython import get_ipython
        get_ipython().magic('reset -f')
    except:
        pass
 
 
from platform import system
     
import os
if system() != 'Darwin' and system() != 'Windows':      
    os.environ['QT_QPA_PLATFORM']='offscreen'
    
if system() == 'Darwin':
    os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin/'
   

import numpy as np
from residuals import mdl_resid
from data_moments3 import dat_moments
import pickle
 
print('Hi!')
 
 
 
 
if __name__ == '__main__':
     
     
    #import warnings
    #warnings.filterwarnings("error")
    #For graphs later
    graphs=True
    
    #Get the covariance matrix from data
    dat_moments(period=1,sampling_number=100,weighting=True,covariances=True,transform=1)
    
    #Get Data Moments
    with open('moments.pkl', 'rb') as file:
        packed_data=pickle.load(file)
        
    WW=np.linalg.inv(packed_data['W'])
    
    #Build  data moments and pickle them
    dat_moments(period=1,sampling_number=100,weighting=True,transform=1)
    
         
  
    x0 = np.array([ 0.378574,0.0636783,14.1403,2.59918,0.165142,0.608711,-0.297201])
   
    #Name and location of files
    if system() == 'Windows':   
        path='D:/blasutto/store_model'
    else:
        path='D:/blasutto/store_model'
    
    
    #First, solve the model for the main parameter
    resn,W = mdl_resid(x0,return_format=['all residuals','W'],
                                      #load_from=['mdl_save_bil_le.pkl'],#'mdl_save_uni.pkl'],
                                      solve_transition=False,                                    
                                      #save_to=['mdl_save_bil_le.pkl'],#'mdl_save_uni.pkl'],
                                      store_path=path,
                                      verbose=False,calibration_report=False,
                                      welf=False,se=True,draw=False,graphs=False) #Switch to true for decomposition of welfare analysis
                         
    
    #Matrix of derivative
    G=np.zeros((len(W[:,0]),len(x0)))
    xn=x0.copy()
    #Loop over parameters to get derivatives
    for j in range(len(x0)):
        
        #Get the new parameter
        xn=x0.copy()
        xn[j]=x0[j]*1.025
        if j==0:
            xn[j]=1-(1-x0[j])*1.025
        
    
        
        #Solve
        res,W = mdl_resid(xn,return_format=['all residuals','W'],                                    
                                      solve_transition=False,                                                           
                                      store_path=path,
                                      verbose=False,calibration_report=False,
                                      welf=False,se=True,draw=False,graphs=False )#Switch to true for decomposition of welfare analysis
                         
        #Store 
        G[:,j]=(res-resn)/abs(xn[j]-x0[j])
        if j==3:
            G[:,j]=(res-resn)/abs(x0[1]*xn[j]-x0[1]*x0[j])

    
        
    #Standard  Errors
    
    #se=np.sqrt(np.diag(np.linalg.inv(np.dot(np.dot(np.transpose(G),W),G)))*(1+1/45000))
    
    aaa=np.linalg.inv(G.T @ W @ G)
    se=np.sqrt(np.diag(aaa.T @ (G.T @ (W.T @ WW @ W) @ G) @ aaa))*(1+1/45000)
    
    print(se)
    print('the stanrard errors are {}'.format(se))