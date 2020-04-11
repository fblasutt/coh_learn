#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This collects auxiliary functions

"""
import numpy as np
from platform import system



def num_to_nparray(*things_in):
    # makes everything numpy array even if passed as number
    
    out = ( 
            x if isinstance(x,np.ndarray) else np.array([x]) 
            for x in things_in
          ) # this is generator not tuple
    return tuple(out)


def unify_sizes(*things_in):
    # makes all the inputs the same size. basically broadcasts all singletones
    # to be the same size as the first non-singleton array. Also checks is all
    # non-singleton arrays are of the same size
    
    shp = 1
    for x in things_in:
        if x.size != 1: 
            shp = x.shape
            break
        
    if shp != 1:
        assert all([(x.size==1 or x.shape==shp) for x in things_in]), 'Wrong shape, it is {}, but we broadcast it to {}'.format([x.shape for x in things_in],shp)
        things_out = tuple( ( x if x.shape==shp else x*np.ones(shp) for x in things_in))
    else:
        things_out = things_in
        
    return things_out




def first_true(mask, axis=None, invalid_val=-1):
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_true(mask, axis=None, invalid_val=-1):
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)



#if system() != 'Darwin' and system() != 'Windows':  import cupy as cp



def zero_hit_mat(Vin,trim=True,test_monotonicity=False,test_sc=True,return_loc=False):
    # for a vector Vin it retunrs the "location" k of zero between coordiates
    # i and i+1. Precisely, it returns vector k where k[i] is such that 
    # x[i]*(1-k[i]) + x[i+1]*k[i] = 0. 
    # Size of k is size of Vin-1. 
    # If Vin is array, it performs the same thing operation over the LAST 
    # dimension of matrix (i.e. for matrix -- it works over column dimension)
    # 
    # If trim = True:
    #    k[i] are trimed to be between 0 and 1, so if k[i] is 1 this indicates
    #    that both x[i] and x[i+1] are (weakly) negative, 
    #    if k[i] is 0 then both are positive, so k[i] is between 0 and 1 only
    #    in case when x[i] and x[i+1] are of the opposite sign
    # It also can test monotonicity and single crossing (uniquness of hitting 
    # zero). Monotonicity guarantees single crossing.
    # F
    
    dV = np.diff(Vin,axis=-1)    
    if test_monotonicity: assert np.all(dV>0)
    
    k = - Vin[...,:-1] / dV
    
    if test_sc:
        sum_hit = np.sum( (k>=0)*(k<=1), axis=-1)
        assert np.all( (sum_hit==1) | (sum_hit==0) )
        
    if trim: k = np.maximum(np.minimum(k,1.0),0.0)
        
    if not return_loc:
        return k
    else:
        
        # now we have to figure out the location of k that is between 0 and 1
        k_01 = (k>0)*(k<1)
        # this requires single-crossing
        # but if violated this returns the location of the first crossing
        all_neg = np.all(k==1,axis=-1)
        all_pos = np.all(k==0,axis=-1)
        loc = np.argmax(k_01,axis=-1)
        loc[all_neg] = k.shape[-1]+1      
        loc[all_pos] = -1
        
        
        loc_adj = np.minimum(np.maximum(loc,0),k.shape[-1]-1)
        
        #print((loc))
        
        
        k_flat = np.take_along_axis(k,np.expand_dims(loc_adj,-1),-1)   
        
        
        assert np.all( k_flat[(~all_neg) & (~all_pos)] < 1)
        assert np.all( k_flat[(~all_neg) & (~all_pos)] > 0)
        
        
        return k, loc, k_flat