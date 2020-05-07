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
    
    
    

    x0 = np.array([0.2,0.1110307,1.11501,0.543047,0.050264,0.005,-0.09])
    x0 = np.array([1.4,0.3110307,2.11501,0.343047,0.7550264,0.015,-0.09])
    x0 = np.array([0.5535,0.599,1.84,0.246,0.7639,0.0168,-0.100])
    x0 = np.array([0.919368,0.479426,2.05565,0.299191,0.730532,0.0232399,-0.0794582,1.07])
    x0 = np.array([0.8155269,   0.06565744,  3.5,  0.2904853,   0.9371481,  0.018159483, -0.091977,  1.0505955,0.15])
   
    #Best
    x0 = np.array([2.33670898,  0.37723633,  3.03123047,  0.31875977,  0.5732832,   0.07676172, -0.07463965,  1.0028418,   0.38623047]) 
   
   #thirs 
    x0 = np.array([1.25855,0.0989453,2.40195,0.618086,0.77493,0.0279531,-0.118074,1.05691,0.123047])

    #Second
    x0 = np.array([0.7415771,   0.06438326,  3.09944071,  0.99,  .028429337,  0.0485463-1.0, -0.12966307,  0.62101063,  0.05349712])

  
    #x0 = np.array([2.24964844,  0.63925781,  2.93851562,  0.29136719,  0.59011719,  0.0613125, -0.05425391,  1.02433594,  0.05273438])
    #x0 = np.array([ 0.866640625, 0.857421875,3.1042187500000002,0.297265625,0.6507343750000001,0.01860078125,-0.0893671875,1.07])
    # 0.50371334  0.27677831  2.9702419   0.48783594  0.67964845  0.01472046-0.11059054  1.10997993

             

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
                                      welf=True) #Switch to true for decomposition of welfare analysis
                         
    print('Done. Residual in point x0 is {}'.format(out))
     
    #assert False
    
    #Indexes for the graphs
    if graphs:
        ai=0
        zfi=3
        zmi=3
        psii=4
        ti=4
        thi=5
        dd=0
        edu=['e','e']
         
        #Actual Graphs
        mdl[0].graph(ai,zfi,zmi,psii,ti,thi,dd,edu)
        get_ipython().magic('reset -f')
        #If you plan to use graphs only once, deselect below to save space on disk
        #os.remove('name_model.pkl')
     
     
  
    
    
        
