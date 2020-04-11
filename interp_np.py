#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:54:40 2019

@author: egorkozlov
"""

import numpy as np


def interp(grid,xnew,return_wnext=True,use_cp=False,check_sorted=False,trim=False):
    # this returns indices j and weights w of where to put xnew relative to
    # grid. 
    #
    # If return_wnext is True, it holds that
    # xnew = grid[j]*(1-w) + grid[j+1]*w.
    # If return_wnext is False, it holds that 
    # xnew = grid[j]*w + grid[j+1]*(1-w).
    #
    # This assumes that grid is a sorted array.
    #
    # trim can specify whether to trim the values at the top and bottom 
    # (so they do not exceed the top and bottom points of the grid)
    
    if check_sorted: assert np.all(np.diff(grid) > 0)
    
    if not use_cp:
        return interp_np(grid,xnew,return_wnext=return_wnext,trim=trim)
    else:
        return interp_tu(grid,xnew,return_wnext=return_wnext,tirm=trim)


def interp_np(grid,xnew,return_wnext=True,trim=False):    
    # this finds grid positions and weights for performing linear interpolation
    # this implementation uses numpy
    
    if trim: xnew = np.minimum(grid[-1], np.maximum(grid[0],xnew) )
    
    j = np.minimum( np.searchsorted(grid,xnew,side='left')-1, grid.size-2 )
    wnext = (xnew - grid[j])/(grid[j+1] - grid[j])
    
    return j, (wnext if return_wnext else 1-wnext) 


def interp_tu(grid,xnew,return_wnext=True,trim=True):
    # this is based on trans_unif.py code
    # this is kept as this can be ran on GPU using cupy
    
    if not isinstance(xnew,np.ndarray): xnew = np.array([xnew])
    
    
    assert grid.ndim==1, "grid should be 1-dimensional array"
    assert xnew.ndim==1, "xnew should be 1-dimensional array"
    
    j_to = np.empty(xnew.shape,np.int32)
    w_to = np.empty(xnew.shape,np.float32)
    
    value_to = np.minimum(grid[-1], np.maximum(grid[0], xnew) ) if trim else xnew
    
    
    
    j_to[:] = np.maximum( np.minimum(  np.sum(value_to[:,np.newaxis] > grid, axis=1) - 1, grid.size - 2 ), 0 )
    
    w_next = (value_to - grid[j_to]) / (grid[j_to+1] - grid[j_to])
    
    w_to[:] = w_next if return_wnext else 1 - w_next
    
    assert (np.all(w_to<=1) and np.all(w_to>=0))
    
    return j_to, w_to



if __name__ == "__main__":

    grid = np.array([1,2,3,5,6,9])
    xnew = np.array([2.5,3,1.2,9,9])

    j, p = interp_tu(grid,xnew,return_wnext=False)
    j2, p2 = interp_np(grid,xnew,return_wnext=False)

    #assert np.all(u[0] == f[0]) and np.all( np.abs(u[1]-f[1]) < 1e-6 ), "Different results?"
    
    assert np.all(j==j2) and np.all(np.abs(p-p2)<1e-5)
    print( (j,p,j2,p2) )   