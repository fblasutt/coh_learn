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
 
print('Hi!')
 
 
 
 
if __name__ == '__main__':
     
     
    #import warnings
    #warnings.filterwarnings("error")
    #For graphs later
    graphs=False
    #Build  data moments and pickle them
    dat_moments(period=1,sampling_number=1,transform=1)
    
         
    #Initialize the file with parameters
    
    

    #Second
    #x0 = np.array([0.0,   0.06565744,  1.5,  0.2904853,   0.7371481,  0.018159483 - 0.6, -0.091977, 0.805955,0.1])
    #x0 = np.array([1.10511688 , 0.10725931,  3.6224206,   0.44856022 , 0.0472732 ,  0.02879032, -0.09039855,  1.23986084 , 0.10953983])
    x0 = np.array([0.0 , 0.04725931,  7.06224206,   0.2, 1.1 ,  0.01-0.0, -0.09039855,  1.13986084 , 0.20953983])


    #Name and location of files
    if system() == 'Windows':   
        path='D:/blasutto/store_model'
    else:
        path='D:/blasutto/store_model'
    
    out, mdl, agents, res = mdl_resid(x0,return_format=['distance','models','agents','scaled residuals'],
                                      #load_from=['mdl_save_bil_le.pkl'],#'mdl_save_uni.pkl'],
                                      solve_transition=False,                                    
                                      #save_to=['mdl_save_bil_le.pkl'],#'mdl_save_uni.pkl'],
                                      #store_path=path,
                                      verbose=True,calibration_report=False,draw=graphs,graphs=graphs,
                                      welf=False) #Switch to true for decomposition of welfare analysis
                         
    print('Done. Residual in point x0 is {}'.format(out))
     
    #assert False
    
    #Indexes for the graphs
    if graphs:
        ai=0
        zfi=3
        zmi=3
        psii=0
        ti=4
        thi=5
        dd=0
        edu=['e','e']
         
        #Actual Graphs
        mdl[0].graph(ai,zfi,zmi,psii,ti,thi,dd,edu)
        get_ipython().magic('reset -f')
        #If you plan to use graphs only once, deselect below to save space on disk
        #os.remove('name_model.pkl')
     
     
  
    
    
        
