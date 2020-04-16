#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This collects selected routines from opt_test.py

The aim is keep each function readable, and either njit-able or using cuda

Both versions are needed as it is not clear whether we'll actually use cuda

"""



import numpy as np
from numba import njit, prange
from platform import system



if system() != 'Darwin' and system() != 'Windows' and system()!='Linux':
#if system() != 'Darwin':
    import cupy as cp
    ugpu = True
else:
    ugpu = False
    
def v_optimize_couple(money_in,sgrid,EV,mgrid,utilint,xint,ls,beta,ushift,
                              use_gpu=ugpu,dtype=np.float32,mt=0.0):
    
    # This optimizer avoids creating big arrays and uses parallel-CPU on 
    # machines without NUMBA-CUDA codes otherwise
    

    nls = len(ls)
    
        
    assert isinstance(money_in,tuple)
    assert len(money_in)==3
    
    asset_income, wf, wm = money_in
    
    
    nexo = wf.size
    na = asset_income.size
    
    wf = wf.reshape((1,wf.size))
    wm = wm.reshape((1,wm.size))
    asset_income = asset_income.reshape((asset_income.size,1))
    money = wf + wm + asset_income
        
    
    if isinstance(EV,tuple):
        assert len(EV) == 2
        vsgrid,EVin = EV
        EVin = EVin.astype(dtype,copy=False)
        EV_by_l = vsgrid.apply_preserve_shape(EVin)
        assert EVin.shape[1:] == EV_by_l.shape[1:]
        assert EVin.dtype == EV_by_l.dtype

        
    
    ntheta = EV_by_l.shape[-2]
    ns = sgrid.size
    nm = mgrid.size
    
    tal = np.take_along_axis
    
    i_opt_arr = np.empty((na,nexo,ntheta,nls),dtype=np.int16)
    c_opt_arr = np.empty((na,nexo,ntheta,nls),dtype=dtype)
    x_opt_arr = np.empty((na,nexo,ntheta,nls),dtype=dtype)
    s_opt_arr = np.empty((na,nexo,ntheta,nls),dtype=dtype)
    V_opt_arr = np.empty((na,nexo,ntheta,nls),dtype=dtype)
    
    
    
    for i, lval in enumerate(ls):
        
        EV_here = EV_by_l[...,i]
        util = utilint[...,i]
        xvals = xint[...,i]
        
        money_left = (money - (1-lval)*wf.reshape((1,nexo))- mt*wm.reshape((1,nexo))).astype(dtype,copy=False)
        
        if not use_gpu:
            
            # preallocation helps a bit here          
            V, c, x, s = np.empty((4,na,nexo,ntheta),dtype=dtype)
            i_opt = -np.ones((na,nexo,ntheta),dtype=np.int16)                 
            v_couple_par(money_left,sgrid,EV_here,mgrid,util,xvals,beta,ushift,V,i_opt,c,x,s)            
            
            
        else:
            #print((EV_here.shape,(na,nexo,ntheta,nls)))
            V, i_opt, c, x, s = v_couple_gpu(money_left,sgrid,EV_here,mgrid,util,xvals,beta,ushift)


        
                
        i_opt_arr[...,i] = i_opt
        c_opt_arr[...,i] = c
        x_opt_arr[...,i] = x
        s_opt_arr[...,i] = s
        V_opt_arr[...,i] = V # discard this V later as it is not very precise
        
        
        
    i_ls = np.expand_dims(V_opt_arr.argmax(axis=3),3)
    
    V = tal(V_opt_arr,i_ls,axis=3).squeeze(axis=3)
    c = tal(c_opt_arr,i_ls,axis=3).squeeze(axis=3)
    s = tal(s_opt_arr,i_ls,axis=3).squeeze(axis=3)
    i_opt = tal(i_opt_arr,i_ls,axis=3).squeeze(axis=3)
    i_ls = i_ls.squeeze(axis=3)
        
    
    assert V.dtype == dtype
    assert c.dtype == dtype
    assert s.dtype == dtype
    
    
    ret = lambda q : q
        
    V_all = V_opt_arr
    
    
    
    return ret(V), ret(c), ret(x), ret(s), ret(i_opt), ret(i_ls), ret(V_all)




@njit(parallel=True)
def v_couple_par(money,sgrid,EV,mgrid,u_on_mgrid,x_on_mgrid,beta,uadd,V_opt,i_opt,c_opt,x_opt,s_opt):
    # this is a looped version of the optimizer
    # the last two things are outputs
    
    na, nexo, ntheta = money.shape[0], money.shape[1], EV.shape[2]
    
    ns = sgrid.size
    
    #nm = mgrid.size
    
    assert money.shape == (na,nexo)
    assert V_opt.shape == (na,nexo,ntheta) == i_opt.shape
    assert EV.shape == (ns,nexo,ntheta)
    
    

    
    for ind_exo in prange(nexo):
        EV_slice = EV[:,ind_exo,:]
        for ind_a in prange(na):
            # finds index of maximum savings
            money_i = money[ind_a,ind_exo]
            i_opt_i, V_opt_i, x_opt_i, c_opt_i, s_opt_i = \
                v_couple_par_int(money_i,sgrid,EV_slice,mgrid,u_on_mgrid,x_on_mgrid,beta,uadd)
            
            i_opt[ind_a,ind_exo,:] = i_opt_i
            V_opt[ind_a,ind_exo,:] = V_opt_i
            x_opt[ind_a,ind_exo,:] = x_opt_i
            c_opt[ind_a,ind_exo,:] = c_opt_i
            s_opt[ind_a,ind_exo,:] = s_opt_i
            

@njit
def v_couple_par_int(money_i,sgrid,EV_slice,mgrid,u_on_mgrid,x_on_mgrid,beta,uadd):
    
    ns = sgrid.size
    nm = mgrid.size
    ntheta = EV_slice.shape[1]
    
    i_opt_i = np.empty((ntheta,),dtype=np.int16)
    V_opt_i = np.empty((ntheta,),dtype=np.float64)
    x_opt_i = np.empty((ntheta,),dtype=np.float64)
    c_opt_i = np.empty((ntheta,),dtype=np.float64)
    s_opt_i = np.empty((ntheta,),dtype=np.float64)
    
    
    money_minus_coh = money_i - mgrid[0]
            
    ind_s = np.minimum( np.searchsorted(sgrid,money_minus_coh)-1,ns-1)
    
    i_m = np.minimum( np.searchsorted(mgrid,money_i)-1,nm-1)
    
    i_m_all = np.zeros((ind_s+1,),np.int16)
    
    i_m_all[0] = i_m
    
    for i_cand in range(1,ind_s+1):
        m_after_s = money_i - sgrid[i_cand]
        while mgrid[i_m] > m_after_s:
            i_m -= 1                    
        assert i_m >= 0
        i_m_all[i_cand] = i_m
        
    
    
    
    for ind_theta in range(ntheta):
        
        ugrid_ce = u_on_mgrid[:,ind_theta]
        EVval = EV_slice[:,ind_theta]
        
        io = 0
        Vo = -1e20
        
        
        for i_cand in range(ind_s+1):
            
            i_m = i_m_all[i_cand]
            u_cand = ugrid_ce[i_m]
            
            V_cand = u_cand + beta*EVval[i_cand]
            
            if i_cand == 0 or V_cand > Vo:
                io, Vo = i_cand, V_cand                     
            
        
        i_opt_i[ind_theta] = io
        V_opt_i[ind_theta] = Vo + uadd# NB: this V is imprecise
        # you can recover V from optimal savings & consumption later
        x_opt_i[ind_theta] = x_on_mgrid[i_m_all[io],ind_theta]
        c_opt_i[ind_theta] = money_i - x_opt_i[ind_theta] - sgrid[io]
        s_opt_i[ind_theta] = sgrid[io]        
       
        assert Vo > -1e20
    return i_opt_i, V_opt_i, x_opt_i, c_opt_i, s_opt_i

    

from math import ceil
from numba import cuda



def v_couple_gpu(money,sgrid,EV,mgrid,util,xvals,beta,uadd,use_kernel_pool=False):
    
    
    na, nexo, ntheta = money.shape[0], money.shape[1], EV.shape[2]
    
    ns = sgrid.size
    
    
    #assert ns < 2000, 'Please alter the array size in cuda_ker_pool'
    
    assert money.shape == (na,nexo)
    assert EV.shape == (ns,nexo,ntheta)
    
    V_opt_g = cuda.device_array((na,nexo,ntheta),dtype=EV.dtype)  
    x_opt_g = cuda.device_array((na,nexo,ntheta),dtype=EV.dtype)  
    i_opt_g = cuda.device_array((na,nexo,ntheta),dtype=np.int16)
    
    bEV = beta*EV
    
    
    money_g, sgrid_g, bEV_g, mgrid_g, util_g, xvals_g = (cuda.to_device(np.ascontiguousarray(x)) for x in (money, sgrid, bEV, mgrid, util, xvals))
    
    
    threadsperblock = (8, 16, 8)
    # this is a tunning parameter. 8*16*8=1024 is the number of threads per 
    # block. This is GPU specific, on Quests's GPU the maximum is 1024, on 
    # different machines it can be lower
    
    b_a = int(ceil(na / threadsperblock[0]))
    b_exo = int(ceil(nexo / threadsperblock[1]))
    b_theta = int(ceil(ntheta / threadsperblock[2]))
    blockspergrid = (b_a, b_exo, b_theta)
    
    cuda_ker[blockspergrid, threadsperblock](money_g, sgrid_g, bEV_g, mgrid_g, util_g, xvals_g, uadd,
                                                V_opt_g,i_opt_g,x_opt_g)
    
    V_opt, i_opt, x_opt = (x.copy_to_host() for x in (V_opt_g,i_opt_g,x_opt_g))
    
    s_opt = sgrid[i_opt]
    c_opt = money[:,:,None] - x_opt - s_opt
    
    return V_opt,i_opt,c_opt,x_opt,s_opt


#from numba import f4

@cuda.jit
def cuda_ker(money_g, sgrid_g, bEV_g, mgrid_g, util_g, xvals_g, uadd, V_opt_g, i_opt_g, x_opt_g):
    ind_a, ind_exo, ind_theta = cuda.grid(3)
        
    na = money_g.shape[0]
    nexo = money_g.shape[1]
    ntheta = bEV_g.shape[2]
    ns = sgrid_g.size
    nm = mgrid_g.size
    
    if (ind_a < na) and (ind_exo < nexo) and (ind_theta < ntheta):
        # finds index of maximum savings
        money_i = money_g[ind_a,ind_exo]
        bEV_of_s = bEV_g[:,ind_exo,ind_theta]  
        util_here = util_g[:,ind_theta]
        xvals_here = xvals_g[:,ind_theta]
        
        mmin = mgrid_g[0]
        
            
        for im in range(nm-1,-1,-1):
            if mgrid_g[im] < money_i: break
        
        iobest = 0
        vbest = util_here[im] + bEV_of_s[0]
        xbest = xvals_here[im]
        
        for io in range(1,ns):
            money_left = money_i - sgrid_g[io]
            if money_left < mmin: break
            
            while mgrid_g[im] > money_left:
                im -= 1
            
            assert im >= 0            
            
            vnow = util_here[im] + bEV_of_s[io] 
            if vnow > vbest:
                iobest = io
                vbest = vnow
                xbest = xvals_here[im] 
            
        i_opt_g[ind_a,ind_exo,ind_theta] = iobest
        x_opt_g[ind_a,ind_exo,ind_theta] = xbest
        V_opt_g[ind_a,ind_exo,ind_theta] = vbest + uadd
        
    
    
