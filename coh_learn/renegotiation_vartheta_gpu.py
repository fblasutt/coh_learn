#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:56:40 2020

@author: egorkozlov
"""

from numba import cuda, f4, f8
import numpy as np

use_f32 = False

if use_f32:
    gpu_type = f4
    cpu_type = np.float32
else:
    gpu_type = f8
    cpu_type = np.float64


def v_ren_gpu_oneopt(v_y_ni, vf_y_ni, vm_y_ni, vf_n_ni, vm_n_ni, itht, wntht, thtgrid):
    
                    
    na, ne, nt_coarse = v_y_ni.shape
    nt = thtgrid.size
    
    assert nt < 500 
    
    
    
    
    
    
    v_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    vm_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    vf_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    itheta_out = cuda.device_array((na,ne,nt),dtype=np.int16)
    
    thtgrid = cuda.to_device(thtgrid)
    

    threadsperblock = (nt, 1, 1)
        
    b_a = na
    b_exo = ne
    b_theta = 1
    
    blockspergrid = (b_theta, b_a, b_exo)
    
    v_y, vf_y, vm_y = [cuda.to_device(
                            np.ascontiguousarray(x)
                                        ) for x in (v_y_ni, vf_y_ni, vm_y_ni)]
    
    vf_n, vm_n = [cuda.to_device(
                                    np.ascontiguousarray(x)
                                  ) for x in (vf_n_ni,vm_n_ni)]
    
    
    
    
    cuda_ker_one_opt[blockspergrid, threadsperblock](v_y, vf_y, vm_y, vf_n, vm_n, 
                                    itht, wntht, thtgrid,  
                                    v_out, vm_out, vf_out, itheta_out)
    
    v_out, vm_out, vf_out, itheta_out = (x.copy_to_host() 
                            for x in (v_out, vm_out, vf_out, itheta_out))
    
    return v_out, vf_out, vm_out, itheta_out
    



@cuda.jit   
def cuda_ker_one_opt(v_y_ni, vf_y_ni, vm_y_ni, vf_n_ni, vm_n_ni, itht, wntht, thtgrid, v_out, vm_out, vf_out, itheta_out):
    # this assumes block is for the same a and theta
    it, ia, ie  = cuda.grid(3)
    
    v_in_store  = cuda.shared.array((500,),gpu_type)
    vf_in_store = cuda.shared.array((500,),gpu_type)
    vm_in_store = cuda.shared.array((500,),gpu_type)
    
    vf_no_store = cuda.shared.array((500,),gpu_type)
    vm_no_store = cuda.shared.array((500,),gpu_type)
    
    
    na = v_y_ni.shape[0]
    ne = v_y_ni.shape[1]
    nt_crude = v_y_ni.shape[2]
    nt = thtgrid.size
    
    
    f1 = gpu_type(1.0)
    
    if ia < na and ie < ne and it < nt:
        
        it_int = itht[it]
        for ittc in range(nt_crude):
            if ittc==it_int: break
        
        
        ittp = ittc + 1
        wttp = wntht[it]
        wttc = f1 - wttp
        
        
        v_in_store[it]  = wttc*v_y_ni[ia,ie,ittc]  + wttp*v_y_ni[ia,ie,ittp]
        vf_in_store[it] = wttc*vf_y_ni[ia,ie,ittc] + wttp*vf_y_ni[ia,ie,ittp]
        vm_in_store[it] = wttc*vm_y_ni[ia,ie,ittc] + wttp*vm_y_ni[ia,ie,ittp]
        
        vf_no_store[it] = wttc*vf_n_ni[ia,ie,ittc] + wttp*vf_n_ni[ia,ie,ittp]
        vm_no_store[it] = wttc*vm_n_ni[ia,ie,ittc] + wttp*vm_n_ni[ia,ie,ittp]
        
        
        v_out[ia,ie,it] = v_in_store[it]
        vf_out[ia,ie,it] = vf_in_store[it] 
        vm_out[ia,ie,it] = vm_in_store[it] 
        
        cuda.syncthreads()
        
        vf_no = vf_no_store[it]
        vm_no = vm_no_store[it]
        
        
        
        
        
        # fill default values
        
        
        
        if vf_in_store[it] >= vf_no and vm_in_store[it] >= vm_no:
        #if vf_y[ia,ie,it] >= vf_no and vm_y[ia,ie,it] >= vm_no:
            itheta_out[ia,ie,it] = it
            return
        
        if vf_in_store[it] < vf_no and vm_in_store[it] < vm_no:
        #if vf_y[ia,ie,it] < vf_no and vm_y[ia,ie,it] < vm_no:
            itheta_out[ia,ie,it] = -1
            tht = thtgrid[it]
            v_out[ia,ie,it] = tht*vf_no + (1-tht)*vm_no
            vf_out[ia,ie,it] = vf_no
            vm_out[ia,ie,it] = vm_no
            return
        
        
       
        it_ren = -1
        
        found_increase = False
        found_decrease = False
         
         
        for it_inc in range(it+1,nt):
            if (vf_in_store[it_inc] >= vf_no and vm_in_store[it_inc] >= vm_no):
                found_increase = True
                break
         
        for it_dec in range(it-1,-1,-1):
            if (vf_in_store[it_dec] >= vf_no and vm_in_store[it_dec] >= vm_no):
                found_decrease = True
                break
#            
#        
        if found_increase and found_decrease:
            dist_increase = it_inc - it
            dist_decrease = it - it_dec
#            
            if dist_increase != dist_decrease:
                it_ren = it_inc if dist_increase < dist_decrease else it_dec
            else:
                # tie breaker
                # numba-cuda does not do abs so we do these dumb things
                dist_mid_inc = it_inc - (nt/2)                
                if dist_mid_inc < 0: dist_mid_inc = -dist_mid_inc
                dist_mid_dec = it_dec - (nt/2)
                if dist_mid_dec < 0: dist_mid_dec = -dist_mid_dec
                it_ren = it_inc if dist_mid_inc < dist_mid_dec else it_dec
            
        elif found_increase and not found_decrease:
            it_ren = it_inc
        elif found_decrease and not found_increase:
            it_ren = it_dec
        else:
            it_ren = -1 # check this!
             
         # finally fill the values    
             
        if it_ren == -1:
            tht = thtgrid[it]
            v_out[ia,ie,it] = tht*vf_no + (1-tht)*vm_no
            vf_out[ia,ie,it] = vf_no
            vm_out[ia,ie,it] = vm_no
            itheta_out[ia,ie,it] = -1
        else:
             
            v_out[ia,ie,it] = v_in_store[it_ren]
            vf_out[ia,ie,it] = vf_in_store[it_ren]
            vm_out[ia,ie,it] = vm_in_store[it_ren]
            itheta_out[ia,ie,it] = it_ren





def v_ren_gpu_twoopt(v_y_ni0, v_y_ni1, vf_y_ni0, vf_y_ni1, vm_y_ni0, vm_y_ni1, vf_n_ni, vm_n_ni, itht, wntht, thtgrid):
    
                    
    na, ne, nt_coarse = v_y_ni0.shape
    nt = thtgrid.size
    
    assert nt < 500 
    
    
    
    
    v_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    vm_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    vf_out = cuda.device_array((na,ne,nt),dtype=cpu_type)
    itheta_out = cuda.device_array((na,ne,nt),dtype=np.int16)
    switch_out = cuda.device_array((na,ne,nt),dtype=np.bool_)
    
    
    thtgrid = cuda.to_device(thtgrid)
    


    threadsperblock = (nt, 1, 1)
        
    b_exo = ne
    b_a = na
    b_theta = 1
    
    blockspergrid = (b_theta, b_a, b_exo)
    
    v_y0, vf_y0, vm_y0 = [cuda.to_device(
                            np.ascontiguousarray(x)
                                        ) for x in (v_y_ni0, vf_y_ni0, vm_y_ni0)]

    v_y1, vf_y1, vm_y1 = [cuda.to_device(
                            np.ascontiguousarray(x)
                                        ) for x in (v_y_ni1, vf_y_ni1, vm_y_ni1)]
    
    vf_n, vm_n = [cuda.to_device(
                                    np.ascontiguousarray(x)
                                  ) for x in (vf_n_ni,vm_n_ni)]
    
    
    
                     
    
    cuda_ker_two_opt[blockspergrid, threadsperblock](v_y0, v_y1, vf_y0, vf_y1, vm_y0, vm_y1, vf_n, vm_n, 
                                    itht, wntht, thtgrid,  
                                    v_out, vm_out, vf_out, itheta_out, switch_out)
    
    
   
    
    v_out, vm_out, vf_out, itheta_out, switch_out = (x.copy_to_host() 
                    for x in (v_out, vm_out, vf_out, itheta_out, switch_out))
    
    return v_out, vf_out, vm_out, itheta_out, switch_out
    


            
            

@cuda.jit   
def cuda_ker_two_opt(v_y_ni0, v_y_ni1, vf_y_ni0, vf_y_ni1, vm_y_ni0, vm_y_ni1, vf_n_ni, vm_n_ni, itht, wntht, thtgrid, v_out, vm_out, vf_out, itheta_out, switch_out):
    # this assumes block is for the same a and theta
    it, ia, ie = cuda.grid(3)
    
    v_in_store  = cuda.shared.array((500,),gpu_type)
    vf_in_store = cuda.shared.array((500,),gpu_type)
    vm_in_store = cuda.shared.array((500,),gpu_type)
    
    vf_no_store = cuda.shared.array((500,),gpu_type)
    vm_no_store = cuda.shared.array((500,),gpu_type)
    
    
    na = v_y_ni0.shape[0]
    ne = v_y_ni0.shape[1]
    nt_crude = v_y_ni0.shape[2]
    nt = thtgrid.size
    
    
    f1 = f8(1.0)
    
    if ia < na and ie < ne and it < nt:
        
        it_int = itht[it]
        for ittc in range(nt_crude):
            if ittc==it_int: break
        
        
        ittp = ittc + 1
        wttp = wntht[it]
        wttc = f1 - wttp
        
        
        vy_0 = wttc*v_y_ni0[ia,ie,ittc] + wttp*v_y_ni0[ia,ie,ittp]
        vy_1 = wttc*v_y_ni1[ia,ie,ittc] + wttp*v_y_ni1[ia,ie,ittp]
        
        pick1 = (vy_1 > vy_0)
        #print(vy_1-vy_0)
        
        switch_out[ia,ie,it] = pick1
        
        if pick1:
            v_in_store[it]  = vy_1
            vf_in_store[it] = wttc*vf_y_ni1[ia,ie,ittc] + wttp*vf_y_ni1[ia,ie,ittp]
            vm_in_store[it] = wttc*vm_y_ni1[ia,ie,ittc] + wttp*vm_y_ni1[ia,ie,ittp]
        else:
            v_in_store[it]  = vy_0
            vf_in_store[it] = wttc*vf_y_ni0[ia,ie,ittc] + wttp*vf_y_ni0[ia,ie,ittp]
            vm_in_store[it] = wttc*vm_y_ni0[ia,ie,ittc] + wttp*vm_y_ni0[ia,ie,ittp]
        
        
        vf_no_store[it] = wttc*vf_n_ni[ia,ie,ittc] + wttp*vf_n_ni[ia,ie,ittp]
        vm_no_store[it] = wttc*vm_n_ni[ia,ie,ittc] + wttp*vm_n_ni[ia,ie,ittp]
        
        
        
        v_out[ia,ie,it] = v_in_store[it]
        vf_out[ia,ie,it] = vf_in_store[it] 
        vm_out[ia,ie,it] = vm_in_store[it] 
        
        cuda.syncthreads()
        
        vf_no = vf_no_store[it]
        vm_no = vm_no_store[it]
        
        
        
        if vf_in_store[it] >= vf_no and vm_in_store[it] >= vm_no:
            itheta_out[ia,ie,it] = it
            return
        
        if vf_in_store[it] < vf_no and vm_in_store[it] < vm_no:
            itheta_out[ia,ie,it] = -1
            tht = thtgrid[it]
            v_out[ia,ie,it] = tht*vf_no + (1-tht)*vm_no
            vf_out[ia,ie,it] = vf_no
            vm_out[ia,ie,it] = vm_no
            return
        
        
       
        it_ren = -1
        
        found_increase = False
        found_decrease = False
         
         
        for it_inc in range(it+1,nt):
            if (vf_in_store[it_inc] >= vf_no and vm_in_store[it_inc] >= vm_no):
                found_increase = True
                break
         
        for it_dec in range(it-1,-1,-1):
            if (vf_in_store[it_dec] >= vf_no and vm_in_store[it_dec] >= vm_no):
                found_decrease = True
                break
#            
#        
        if found_increase and found_decrease:
            dist_increase = it_inc - it
            dist_decrease = it - it_dec
#            
            if dist_increase != dist_decrease:
                it_ren = it_inc if dist_increase < dist_decrease else it_dec
            else:
                # tie breaker
                # numba-cuda does not do abs so we do these dumb things
                dist_mid_inc = it_inc - (nt/2)                
                if dist_mid_inc < 0: dist_mid_inc = -dist_mid_inc
                dist_mid_dec = it_dec - (nt/2)
                if dist_mid_dec < 0: dist_mid_dec = -dist_mid_dec
                it_ren = it_inc if dist_mid_inc < dist_mid_dec else it_dec
            
        elif found_increase and not found_decrease:
            it_ren = it_inc
        elif found_decrease and not found_increase:
            it_ren = it_dec
        else:
            it_ren = -1 # check this!
             
         # finally fill the values    
             
        if it_ren == -1:
            tht = thtgrid[it]
            v_out[ia,ie,it] = tht*vf_no + (1-tht)*vm_no
            vf_out[ia,ie,it] = vf_no
            vm_out[ia,ie,it] = vm_no
            itheta_out[ia,ie,it] = -1
        else:
             
            v_out[ia,ie,it] = v_in_store[it_ren]
            vf_out[ia,ie,it] = vf_in_store[it_ren]
            vm_out[ia,ie,it] = vm_in_store[it_ren]
            itheta_out[ia,ie,it] = it_ren