#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:13:02 2020

@author: egorkozlov
"""

from model import Model
import numpy as np


import os
os.environ['MKL_CBWR']='AUTO'

from numba import config
config.NUMBA_NUM_THREADS = 2
 


x0 = np.array([0.2,0.0710307,1.11501,0.543047,0.050264,0.005,-0.09])

from calibration_params import calibration_params

pars = calibration_params()[-1](x0)
print(pars)

pars.pop('alost')

mdl = Model(**pars,verbose=False,solve_till=0,display_v=False)

mdl.time_statistics()

decisions = mdl.decisions[-2]['Female, single']['Decision']

assert np.allclose(mdl.V[0]['Couple, M']['V'].mean(),-342.69626387623344)
assert np.allclose(mdl.V[0]['Female, single']['V'].mean(),-242.52742451161123)