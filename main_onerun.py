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
    graphs=True
    #Build  data moments and pickle them
    dat_moments(period=1,sampling_number=1,transform=1)
    
         
    #Initialize the file with parameters
    
    

    #Second
    #x0 = np.array([0.0,   0.06565744,  1.5,  0.2904853,   0.7371481,  0.018159483 - 0.6, -0.091977, 0.805955,0.1])
    #x0 = np.array([1.10511688 , 0.10725931,  3.6224206,   0.44856022 , 0.0472732 ,  0.02879032, -0.09039855,  1.23986084 , 0.10953983])
    x0 = np.array([0.3 , 0.04725931*2.996194651745017, 10/2.996194651745017,   0.25, 1.1 ,  0.0075-0.0, -0.09039855,  1.13986084 , 0.30953983*2.996194651745017])

    #1 1
    x0 = np.array([0.321094,0.167578,3.69922,0.214648,1.14922,0.00989453,-0.0951172,1.11156,0.959375])
    
    #1 2
    x0 = np.array([0.251807,0.228589,4.37695,0.183325,1.10332,0.00648584,-0.0843213,1.12707,0.395508])
    
    #Wisdom
    x0 = np.array([0.287148,0.43265,4.43904,0.197515,1.1088,0.0067326,-0.084137,1.1219,1.532568])
    x0 = np.array([0.44474121,  0.32749023,  7.0759082,   0.2302002,   1.09989648,  0.00966553,-0.06185889,  1.1963916,   1.37988281])
    x0 = np.array([0.44474121,  0.42749023*1.2,  27.0759082,   0.2302002,   0.003989648,  0.0,-0.06185889,  1.1963916,   0.77988281*1.2])
    x0 = np.array([0.44474121,  0.42749023*1.2,  0.0759082,   0.2302002,   0.000001,  0.0,-0.06185889,  1.1963916,   0.77988281*1.2])   
    x0 = np.array([0.8296875,   0.238125,    0.79570312,  0.59226563,  0.36492187,  0.01349219, -0.08658594,  1.03046875,  0.2703125])   
    x0 = np.array([0.23673339843750002, 0.19666015625,0.21979980468750004,0.4838623046875,0.1846240234375, -0.12382861328125,1.0734130859375,0.40615234375000003])
   
    #New way
    x0 = np.array([0.75822509765625,0.030224609375, 0.40762451171875,0.48373779296875, 0.45178955078125,-0.077394775390625,1.30259521484375, 0.32880859375000004])
    x0 = np.array([0.75822509765625,0.030224609375, 0.40762451171875,0.48373779296875, 0.15178955078125,-0.1177394775390625,1.0259521484375, 0.32880859375000004])
   



    #Name and location of files
    if system() == 'Windows':   
        path='D:/blasutto/store_model'
    else:
        path='D:/blasutto/store_model'
    
    out, mdl, agents, res = mdl_resid(x0,return_format=['distance','models','agents','scaled residuals'],
                                      #load_from=['mdl_save_bil_le.pkl'],#'mdl_save_uni.pkl'],
                                      solve_transition=False,                                    
                                      save_to=['mdl_save_bil_le.pkl'],#'mdl_save_uni.pkl'],
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
     
     
  
    
    
        
