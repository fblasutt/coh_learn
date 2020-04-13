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
from data_moments import dat_moments
 
print('Hi!')
 
 
 
 
if __name__ == '__main__':
     
     
    #import warnings
    #warnings.filterwarnings("error")
    #For graphs later
    graphs=True
    #Build  data moments and pickle them
    #dat_moments(period=6,sampling_number=4,transform=2)
    
         
    #Initialize the file with parameters
    
    
    
    x0 = np.array([0.701399,0.1810307,1.11501,0.643047,0.180264,0.0,0.71854,(1-0.21)/0.21])
    x0 = np.array([0.0,0.1810307,1.11501,0.543047,0.050264,0.08,0.9999,(1-0.21)/0.21])
    x0 = np.array([0.2,0.1110307,1.11501,0.543047,0.050264,0.005,-0.09])
    x0 = np.array([1.4,0.3110307,2.11501,0.343047,0.7550264,0.015,-0.09])
    x0 = np.array([0.5535,0.599,1.84,0.246,0.7639,0.0168,-0.100])
    x0 = np.array([0.919368,0.479426,2.05565,0.299191,0.730532,0.0232399,-0.0794582,1.07])
    x0 = np.array([0.5155269,   0.43565744,  2.84074454,  0.4604853,   0.67371481,  0.02159483, -0.11831977,  1.10505955])
    #x0 = np.array([ 0.866640625, 0.857421875,3.1042187500000002,0.297265625,0.6507343750000001,0.01860078125,-0.0893671875,1.07])
    # 0.50371334  0.27677831  2.9702419   0.48783594  0.67964845  0.01472046-0.11059054  1.10997993

             

    #Name and location of files
    if system() == 'Windows':   
        path='D:/blasutto/store_model'
    else:
        path = None
    
    out, mdl, agents, res = mdl_resid(x0,return_format=['distance','models','agents','scaled residuals'],
                                      #load_from=['mdl_save_bil.pkl','mdl_save_uni.pkl'],
                                      solve_transition=False,                                    
                                      #save_to=['mdl_save_bil.pkl','mdl_save_uni.pkl'],
                                      store_path=path,
                                      verbose=True,calibration_report=False,draw=graphs,graphs=graphs,
                                      welf=True) #Switch to true for decomposition of welfare analysis
                         
    print('Done. Residual in point x0 is {}'.format(out))
     
    #assert False
    
    #Indexes for the graphs
    if graphs:
        ai=0
        zfi=3
        zmi=4
        psii=1
        ti=4
        thi=5
         
        #Actual Graphs
        mdl[0].graph(ai,zfi,zmi,psii,ti,thi)
        #get_ipython().magic('reset -f')
        #If you plan to use graphs only once, deselect below to save space on disk
        #os.remove('name_model.pkl')
     
     
    
        
