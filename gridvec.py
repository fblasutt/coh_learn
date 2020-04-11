#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:13:57 2019

@author: egorkozlov
"""

from interp_np import interp            
from aux_routines import num_to_nparray
import numpy as np

# this is griddend linear interpolant that uses interpolation routines from
# interp_np. It can be applied to arrays specifying values of function to get
# their interpolated values along an arbitrary dimesnion (see method apply)

# TODO: this should have nan-like values and throw errors

class VecOnGrid(object):
    def __init__(self,grid,values,iwn=None,trim=True):
        # this assumes grid is strictly increasing o/w unpredictable
        self.val = values
        self.val_trimmed = np.clip(values,grid[0],grid[-1])
        self.grid = grid
        
        if iwn is None:
            self.i, self.wnext = interp(self.grid,self.val,return_wnext=True,trim=trim)
        else:
            self.i, self.wnext = iwn
           
        if np.any(self.i<0):
            # manual correction to avoid -1
            ichange = (self.i<0)
            assert np.allclose(self.wnext[ichange],1.0)
            self.i[ichange] = 0
            self.wnext[ichange] = 0.0
        
            
        self.wnext = self.wnext.astype(grid.dtype)
        
        self.n = self.i.size
        
        self.one = np.array(1).astype(grid.dtype) # sorry
        
        self.wthis = self.one-self.wnext
        self.trim = trim
        
        
        
        
        assert np.allclose(self.val_trimmed,self.apply_crude(grid))
        
    
    def apply_crude(self,xin):
        # crude version of apply
        # has no options, assumes xin is 1-dimensional
        return xin[self.i]*self.wthis + xin[self.i+1]*self.wnext
    
    def apply_preserve_shape(self,xin,axis=0):
        nd = xin.ndim
        shp0 = xin.shape
        shp1 = shp0[:axis] + (self.n,) + shp0[axis+1:]
        shpw = (1,)*axis + (self.n,) + (1,)*(nd-axis-1)
        
        ithis = (slice(None),)*axis + (self.i,) + (slice(None),)*(nd-axis-1)
        inext = (slice(None),)*axis + (self.i+1,) + (slice(None),)*(nd-axis-1)
        wthis = self.wthis.astype(xin.dtype,copy=False).reshape(shpw)       
        wnext = self.wnext.astype(xin.dtype,copy=False).reshape(shpw)
        xout = wthis*xin[ithis] + wnext*xin[inext]
        assert xout.dtype == xin.dtype
        assert xout.shape == shp1
        return xout
        
        
        
        
    def apply(self,xin,axis=0,take=None,pick=None,reshape_i=True):
        # this applies interpolator to array xin along dimension axis
        # and additionally takes indices specified in take list
        # take's elements are assumed to be 2-element tuple where take[0] 
        # is axis and take[1] is indices. 
        
        typein = xin.dtype
        
        nd = xin.ndim
        assert axis < nd
        
        if isinstance(take,tuple):
            take = [take]
        
        ithis = [slice(None)]*nd
        inext = [slice(None)]*nd
        
        
        if pick is None:        
            ithis[axis] = self.i
            inext[axis] = self.i + 1
            wthis = self.wthis
            wnext = self.wnext
            n = self.n
        else:
            ithis[axis] = self.i[pick]
            inext[axis] = self.i[pick] + 1
            wthis = self.wthis[pick]
            wnext = self.wnext[pick]
            n = pick.size
            
            
        
        shp_i = (1,)*axis + (n,) + (1-axis)*(1,)
        
        # TODO: this is not general but let's see if we need more
        # this creates 2-dimensional thing
        shp_w = (1,)*axis + (n,) + (nd-1-axis)*(1,)
        
        
        if reshape_i:
            ithis[axis] = ithis[axis].reshape(shp_i)
            inext[axis] = inext[axis].reshape(shp_i)
            
        
        if take is not None:
            
            dimextra = [t[0] for t in take]
            iextra = [t[1] for t in take]
            
            for d, i in zip(dimextra,iextra):
                
                ithis[d] = i
                inext[d] = i
                
            if not reshape_i: # remove one dimension from w
                # if reshape_i is false it assumes that each index in interpolant
                # corresponds to each index in take[1], and therefore the
                # dimensionality is reduced accordingly (I know this seems hard to 
                # comprehend but something like V[[0,1],[0,1],:] returns 2-dimensional
                # array, and V[ [[0],[1]], [[0,1]], :] returns 3-dimensional array, so
                # reshape_i=False leads to dimensionality reduction 
                #assert all([i.shape == self.i.shape for i in iextra])
                shp_w = list(shp_w)
                shp_w = [s for i, s in enumerate(shp_w) if i not in dimextra] # remove extra elements
                shp_w = tuple(shp_w)
                
                
        wthis = wthis.reshape(shp_w)
        wnext = wnext.reshape(shp_w)
        
        out = wthis*xin[tuple(ithis)] + wnext*xin[tuple(inext)]
        
        # TODO: check if this hurts dimensionality
        # FIXME: yes it does
        
        return (np.atleast_1d(out.astype(typein).squeeze()))
    
    
    def apply_2dim(self,xin,*,apply_first,axis_first,axis_this=0,take=None,pick=None,reshape_i=True):
        # this is experimental, use at your own risk        
        # may not react on some options nicely
        
        assert not reshape_i, 'This does not work'            
        assert isinstance(apply_first,VecOnGrid)
        
        
        # this first applies apply_first and then applies current VecOnGrid
        
        if take is None: take = list()
        if type(take) is not list: take = list(take)
        
        _ithis = self.i     if pick is None else  self.i[pick]
        _inext = self.i+1   if pick is None else  self.i[pick] + 1
        _wthis = self.wthis if pick is None else  self.wthis[pick]
        _wnext = self.wnext if pick is None else  self.wnext[pick]
        
        take_this = take + [(axis_this,_ithis)]
        take_next = take + [(axis_this,_inext)]
        
        xthis = apply_first.apply(xin,axis=axis_first,take=take_this,pick=pick,reshape_i=reshape_i)
        xnext = apply_first.apply(xin,axis=axis_first,take=take_next,pick=pick,reshape_i=reshape_i)
        
        return _wthis*xthis + _wnext*xnext
        
        
        
            
        
    
    
    def update(self,where,values):
        # this replaces values at positions where with values passed to the
        # function and recomputes interpolation weights
        
        where, values = num_to_nparray(where,values) # safety check in case of singletons
        
        assert ( where.size == values.size ) or (values.size == 1)
        
        if values.size == 1:
            values = values*np.ones(where.shape)
        
        self.val[where] = values
        self.i[where], self.wnext[where] = \
            interp(self.grid,self.val[where],return_wnext=True,trim=self.trim)
        self.wnext[where] =self.wnext[where].astype(self.dtype)
        self.wthis[where] = self.one - self.wnext[where]
        
        
    def roll(self,shocks=None):
        # this draws a random vector of grid poisitions such that probability
        # of self.i is self.wthis and probability of self.i+1 is self.wnext
        
        if shocks is None:
            print('Warning: fix the seed please')
            shocks = np.random.random_sample(self.val.shape)
            
        out = self.i
        out[shocks>self.wthis] += 1
        return out
        