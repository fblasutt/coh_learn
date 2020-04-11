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
    
if system() != 'Darwin' and system() != 'Windows':   
    import os
    os.environ['QT_QPA_PLATFORM']='offscreen'


import numpy as np
from numpy.random import random_sample as rs
from data_moments import dat_moments
from tiktak import tiktak
print('Hi!')

from residuals import mdl_resid
from calibration_params import calibration_params

if __name__ == '__main__':
    
    
    #Build  data moments and pickle them
    dat_moments(period=1,sampling_number=4,transform=2) # refresh
    
    
    
    #Initialize the file with parameters

    lb, ub, x0, keys, translator = calibration_params() # bounds are set in a separate file
    
    
    ##### FIRST LET'S TRY TO RUN THE FUNCTION IN FEW POINTS
    
    print('Testing the workers...')
    from p_client import compute_for_values
    pts = [lb + rs(lb.shape)*(ub-lb) for _ in range(1)]
    pts = [('compute',translator(x)) for x in pts]    
    outs = compute_for_values(pts,timeout=72000.0)
    print('Everything worked, output is {}'.format(outs))
    
    
    print('')
    print('')
    print('running tic tac...')
    print('')
    print('')
    
    

    #Tik Tak Optimization
    param=tiktak(N=400,N_st=15,skip_local=False,skip_global=False)
    
    print('f is {} and x is {}'.format(param[0],param[1]))
    
    #Now Re do the computation with graphs!
    out, mdl = mdl_resid(param[1],return_format=['distance','model'],calibration_report=False,
                         verbose=True,draw=True)
    
    
   
        

