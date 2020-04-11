#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:29:50 2019

@author: egorkozlov
"""

import numpy as np
from scipy.stats import norm

def int_prob_standard(vec,trim=True,trim_level=0.01):
    # given ordered vector vec [x_0,...,x_{n-1}] this returns probabilities
    # [p0,...,p_{n-1}] such that p_i = P[d(Z,x_i) is minimal among i], where
    # Z is standard normal ranodm variable
    
    assert np.all(np.diff(vec)>0), "vec must be ordered and increasing!"
    
    vm = np.concatenate( ([-np.inf],vec[:-1]) )
    vp = np.concatenate( (vec[1:],[np.inf]) )
    v_up   = 0.5*(vec + vp)
    v_down = 0.5*(vec+vm)
    assert np.all(v_up>v_down)
    
    p = norm.cdf(v_up) - norm.cdf(v_down)
    
    if trim:
        p[np.where(p<trim_level)] = 0
        p = p / np.sum(p)
    
    
    #ap = np.argpartition(p,[-1,-2])
    assert(np.abs(np.sum(p) - 1) < 1e-8)
    
    return p#, ap, p[ap[-2]], p[ap[-1]]


def int_prob(vec,mu=0,sig=1,trim=True,trim_level=0.01):
    # this works like int_prob_standard, but assumes that Z = mu + sig*N(0,1)
    vec_adj = (np.array(vec)-mu)/sig
    return int_prob_standard(vec_adj,trim,trim_level)

