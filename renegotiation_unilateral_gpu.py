#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:56:40 2020

@author: egorkozlov
"""

# this is to be used with renegotiation_unilateral

from numba import cuda, f4, f8, i2, b1
import numpy as np

use_f32 = False

if use_f32:
    gpu_type = f4
    cpu_type = np.float32
else:
    gpu_type = f8
    cpu_type = np.float64

from math import ceil

def v_ren_gpu_oneopt(v_y_ni, vf_y_ni, vm_y_ni, vf_n_ni, vm_n_ni, itht, wntht, thtgrid, 
                          rescale = True):
    
                    
    na, ne, nt_coarse = v_y_ni.shape
    nt = thtgrid.size
    assert rescale, 'no rescale is not implemented'
    
    assert nt < 500 
    
    
    
    
    v_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    vm_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    vf_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    itheta_out = cuda.device_array((na,ne,nt),dtype=np.int16)
    
    
    
    
    thtgrid = cuda.to_device(thtgrid)
    

    threadsperblock = (16, 64)
        
    b_a = ceil(na/threadsperblock[0])
    b_exo = ceil(ne/threadsperblock[1])
    
    blockspergrid = (b_a, b_exo)
    
    v_y, vf_y, vm_y = [cuda.to_device(
                            np.ascontiguousarray(x)
                                        ) for x in (v_y_ni, vf_y_ni, vm_y_ni)]
    
    vf_n, vm_n = [cuda.to_device(
                                    np.ascontiguousarray(x)
                                  ) for x in (vf_n_ni,vm_n_ni)]
    
    
    #itht, wntht = (cuda.const.array_like(x) for x in (itht, wntht))
    
    
    cuda_ker_one_opt[blockspergrid, threadsperblock](v_y, vf_y, vm_y, vf_n, vm_n, 
                                    itht, wntht, thtgrid,  
                                    v_out, vm_out, vf_out, itheta_out)
    
    v_out, vm_out, vf_out, itheta_out = (x.copy_to_host() 
                            for x in (v_out, vm_out, vf_out, itheta_out))
    
    return v_out, vf_out, vm_out, itheta_out
    



@cuda.jit   
def cuda_ker_one_opt(v_y_ni, vf_y_ni, vm_y_ni, vf_n, vm_n, itht, wntht, thtgrid, v_out, vm_out, vf_out, itheta_out):
    # this assumes block is for the same a and theta
    ia, ie  = cuda.grid(2)
    
    
    
    na = v_y_ni.shape[0]
    ne = v_y_ni.shape[1]
    nt_crude = v_y_ni.shape[2]
    nt = thtgrid.size
    
    
    f1 = gpu_type(1.0)
    
    if ia < na and ie < ne:
        
        vf_no = vf_n[ia,ie]
        vm_no = vm_n[ia,ie]
        
        v_in_store  = cuda.local.array((500,),gpu_type)
        vf_in_store = cuda.local.array((500,),gpu_type)
        vm_in_store = cuda.local.array((500,),gpu_type)
        
        
        is_good_store = cuda.local.array((500,),b1)
        it_left_store = cuda.local.array((500,),i2)
        it_right_store = cuda.local.array((500,),i2)
        it_best_store = cuda.local.array((500,),i2)
        
        
        ittc = 0
        any_good = False
        for it in range(nt):
            
            it_left_store[it] = -1
            it_right_store[it] = -1
            it_best_store[it] = -1
            
            
            it_int = itht[it]
            for ittc in range(ittc,nt_crude):
                if ittc==it_int: break
            
            
            ittp = ittc + 1
            wttp = wntht[it]
            wttc = f1 - wttp
            
            
            v_in_store[it]  = wttc*v_y_ni[ia,ie,ittc]  + wttp*v_y_ni[ia,ie,ittp]
            vf_in_store[it] = wttc*vf_y_ni[ia,ie,ittc] + wttp*vf_y_ni[ia,ie,ittp]
            vm_in_store[it] = wttc*vm_y_ni[ia,ie,ittc] + wttp*vm_y_ni[ia,ie,ittp]
            
            
            if vf_in_store[it] >= vf_no and vm_in_store[it] >= vm_no:
                is_good_store[it] = True
                any_good = True
            else:
                is_good_store[it] = False
            
        # everything is in local mem now
        
        # go from the right
        
        
        if not any_good:            
            for it in range(nt):
                tht = thtgrid[it]
                v_out[ia,ie,it] = tht*vf_no + (f1-tht)*vm_no
                vf_out[ia,ie,it] = vf_no
                vm_out[ia,ie,it] = vm_no
                itheta_out[ia,ie,it] = -1
            return
        
        assert any_good
        
        if is_good_store[0]: it_left_store[0] = 0        
        for it in range(1,nt):
            it_left_store[it] = it if is_good_store[it] else it_left_store[it-1]
        
        if is_good_store[nt-1]: it_right_store[nt-1] = nt-1
        for it in range(nt-2,-1,-1):
            it_right_store[it] = it if is_good_store[it] else it_right_store[it+1]
        
        # find the best number
        for it in range(nt):
            if is_good_store[it]:
                it_best_store[it] = it
            else:
                if it_right_store[it] >= 0 and it_left_store[it] >= 0:
                    dist_right = it_right_store[it] - it
                    dist_left = it - it_left_store[it]
                    assert dist_right>0
                    assert dist_left>0
                    
                    if dist_right < dist_left:
                        it_best_store[it] = it_right_store[it]
                    elif dist_right > dist_left:
                        it_best_store[it] = it_left_store[it]
                    else:                                
                        # tie breaker
                        drc = 2*it_right_store[it] - nt
                        if drc<0: drc = -drc
                        dlc = 2*it_left_store[it] - nt
                        if dlc<0: dlc = -dlc                                
                        it_best_store[it] = it_left_store[it] if \
                            dlc <= drc else it_right_store[it]
                elif it_right_store[it] >= 0:
                    it_best_store[it] = it_right_store[it]
                elif it_left_store[it] >= 0:
                    it_best_store[it] = it_left_store[it]
                else:
                    assert False, 'this should not happen'
            
            itb = it_best_store[it]
            
        
            tht_old = thtgrid[it]
            tht_new = thtgrid[itb]
            if tht_old>tht_new:
                factor = tht_old/tht_new
            else:
                factor = (1-tht_old)/(1-tht_new)
            
            
            v_out[ia,ie,it] = factor*v_in_store[itb]
            vf_out[ia,ie,it] = vf_in_store[itb]
            vm_out[ia,ie,it] = vm_in_store[itb]
            itheta_out[ia,ie,it] = itb
            
            assert vf_out[ia,ie,it] >= vf_no
            assert vm_out[ia,ie,it] >= vm_no
            
            

def v_ren_gpu_twoopt(v_y_ni0, v_y_ni1, vf_y_ni0, vf_y_ni1, vm_y_ni0, vm_y_ni1, vf_n_ni, vm_n_ni, itht, wntht, thtgrid, 
                          rescale = True):
    
    
    na, ne, nt_coarse = v_y_ni0.shape
    nt = thtgrid.size
    assert rescale, 'no rescale is not implemented'
    
    assert nt < 500 
    
    
    
    
    v_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    vm_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    vf_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    itheta_out = cuda.device_array((na,ne,nt),dtype=np.int16)
    switch_out = cuda.device_array((na,ne,nt),dtype=np.bool_)
    
    
    thtgrid = cuda.to_device(thtgrid)
    

    threadsperblock = (16, 64)
        
    b_a = ceil(na/threadsperblock[0])
    b_exo = ceil(ne/threadsperblock[1])
    
    blockspergrid = (b_a, b_exo)
    
    v_y0, vf_y0, vm_y0 = [cuda.to_device(
                            np.ascontiguousarray(x)
                                        ) for x in (v_y_ni0, vf_y_ni0, vm_y_ni0)]

    v_y1, vf_y1, vm_y1 = [cuda.to_device(
                            np.ascontiguousarray(x)
                                        ) for x in (v_y_ni1, vf_y_ni1, vm_y_ni1)]
    
    vf_n, vm_n = [cuda.to_device(
                                    np.ascontiguousarray(x)
                                  ) for x in (vf_n_ni,vm_n_ni)]
    
    
    
    
    
    
    #itht, wntht = (cuda.const.array_like(x) for x in (itht, wntht))
    
    
    cuda_ker_two_opt[blockspergrid, threadsperblock](v_y0, v_y1, vf_y0, vf_y1, vm_y0, vm_y1, vf_n, vm_n, 
                                    itht, wntht, thtgrid,  
                                    v_out, vm_out, vf_out, itheta_out, switch_out)
    
    v_out, vm_out, vf_out, itheta_out, switch_out = (x.copy_to_host() 
                            for x in (v_out, vm_out, vf_out, itheta_out, switch_out))
    
    return v_out, vf_out, vm_out, itheta_out, switch_out   
            
            


@cuda.jit   
def cuda_ker_two_opt(v_y_ni0, v_y_ni1, vf_y_ni0, vf_y_ni1, vm_y_ni0, vm_y_ni1, vf_n, vm_n, itht, wntht, thtgrid, v_out, vm_out, vf_out, itheta_out, switch_out):
    # this assumes block is for the same a and theta
    ia, ie  = cuda.grid(2)
    
    
    
    na = v_y_ni0.shape[0]
    ne = v_y_ni0.shape[1]
    nt_crude = v_y_ni0.shape[2]
    nt = thtgrid.size
    
    
    f1 = gpu_type(1.0)
    
    if ia < na and ie < ne:
        
        vf_no = vf_n[ia,ie]
        vm_no = vm_n[ia,ie]
        
        v_in_store  = cuda.local.array((500,),gpu_type)
        vf_in_store = cuda.local.array((500,),gpu_type)
        vm_in_store = cuda.local.array((500,),gpu_type)
        
        
        is_good_store = cuda.local.array((500,),b1)
        it_left_store = cuda.local.array((500,),i2)
        it_right_store = cuda.local.array((500,),i2)
        it_best_store = cuda.local.array((500,),i2)
        
        
        ittc = 0
        any_good = False
        
        for it in range(nt):
            
            it_left_store[it] = -1
            it_right_store[it] = -1
            it_best_store[it] = -1
            
            
            it_int = itht[it]
            for ittc in range(ittc,nt_crude):
                if ittc==it_int: break
            
            
            ittp = ittc + 1
            wttp = wntht[it]
            wttc = f1 - wttp
            
            
            
            vy_0 = wttc*v_y_ni0[ia,ie,ittc] + wttp*v_y_ni0[ia,ie,ittp]
            vy_1 = wttc*v_y_ni1[ia,ie,ittc] + wttp*v_y_ni1[ia,ie,ittp]
            
            pick1 = (vy_1 > vy_0)       
            switch_out[ia,ie,it] = pick1
            
            if pick1:
                v_in_store[it]  = vy_1
                vf_in_store[it] = wttc*vf_y_ni1[ia,ie,ittc] + wttp*vf_y_ni1[ia,ie,ittp]
                vm_in_store[it] = wttc*vm_y_ni1[ia,ie,ittc] + wttp*vm_y_ni1[ia,ie,ittp]
            else:
                v_in_store[it]  = vy_0
                vf_in_store[it] = wttc*vf_y_ni0[ia,ie,ittc] + wttp*vf_y_ni0[ia,ie,ittp]
                vm_in_store[it] = wttc*vm_y_ni0[ia,ie,ittc] + wttp*vm_y_ni0[ia,ie,ittp]
            
            
            if vf_in_store[it] >= vf_no and vm_in_store[it] >= vm_no:
                is_good_store[it] = True
                any_good = True
            else:
                is_good_store[it] = False
            
        # everything is in local mem now
        
        # go from the right
        
        
        if not any_good:            
            for it in range(nt):
                tht = thtgrid[it]
                v_out[ia,ie,it] = tht*vf_no + (f1-tht)*vm_no
                vf_out[ia,ie,it] = vf_no
                vm_out[ia,ie,it] = vm_no
                itheta_out[ia,ie,it] = -1
            return
        
        assert any_good
        
        if is_good_store[0]: it_left_store[0] = 0        
        for it in range(1,nt):
            it_left_store[it] = it if is_good_store[it] else it_left_store[it-1]
        
        if is_good_store[nt-1]: it_right_store[nt-1] = nt-1
        for it in range(nt-2,-1,-1):
            it_right_store[it] = it if is_good_store[it] else it_right_store[it+1]
        
        # find the best number
        for it in range(nt):
            if is_good_store[it]:
                it_best_store[it] = it
            else:
                if it_right_store[it] >= 0 and it_left_store[it] >= 0:
                    dist_right = it_right_store[it] - it
                    dist_left = it - it_left_store[it]
                    assert dist_right>0
                    assert dist_left>0
                    
                    if dist_right < dist_left:
                        it_best_store[it] = it_right_store[it]
                    elif dist_right > dist_left:
                        it_best_store[it] = it_left_store[it]
                    else:                                
                        # tie breaker
                        drc = 2*it_right_store[it] - nt
                        if drc<0: drc = -drc
                        dlc = 2*it_left_store[it] - nt
                        if dlc<0: dlc = -dlc                                
                        it_best_store[it] = it_left_store[it] if \
                            dlc <= drc else it_right_store[it]
                elif it_right_store[it] >= 0:
                    it_best_store[it] = it_right_store[it]
                elif it_left_store[it] >= 0:
                    it_best_store[it] = it_left_store[it]
                else:
                    assert False, 'this should not happen'
            
            itb = it_best_store[it]
            
        
            tht_old = thtgrid[it]
            tht_new = thtgrid[itb]
            if tht_old>tht_new:
                factor = tht_old/tht_new
            else:
                factor = (1-tht_old)/(1-tht_new)
            
            v_out[ia,ie,it] = factor*v_in_store[itb]
            vf_out[ia,ie,it] = vf_in_store[itb]
            vm_out[ia,ie,it] = vm_in_store[itb]
            itheta_out[ia,ie,it] = itb
            
            assert vf_out[ia,ie,it] >= vf_no
            assert vm_out[ia,ie,it] >= vm_no
               
