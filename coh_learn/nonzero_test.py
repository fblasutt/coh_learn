#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:33:51 2019

@author: egorkozlov
"""


import numpy as np

nrows = 1000
ncols = 200
rho = -0.3
sig = 0.5
x = np.empty((nrows,ncols),dtype=np.float32)


x[:,0] = np.random.normal(np.zeros(nrows))
for j in range(1,ncols):
    x[:,j] = rho*x[:,j-1] + sig*np.random.normal(np.zeros(nrows))
    
x = x[np.any(x>0,axis=1),:]


x_pos = (x>0)

from ren_mar import ind_no_sc, ind_no_sc_robust

i0 = ind_no_sc(x_pos)
i1 = ind_no_sc_robust(x_pos)

assert np.allclose(i0,i1)


