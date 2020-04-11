#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:11:58 2019

@author: egorkozlov
"""

from scipy.stats import norm
import sobol_seq
v = norm.ppf(sobol_seq.i4_sobol_generate(3,10))
print(v)