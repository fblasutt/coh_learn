#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is solver for those who are couples at period 0
"""
import numpy as np
from timeit import default_timer


from optimizers import v_optimize_couple

from platform import system

if system() != 'Darwin' and system() != 'Windows':    
    nbatch_def = 500    
else:    
    nbatch_def = 17

def v_iter_couple(setup,t,EV_tuple,ushift,nbatch=nbatch_def,verbose=False,
                              force_f32 = False):
    
    if verbose: start = default_timer()
    
    agrid = setup.agrid_c
    sgrid = setup.sgrid_c
    
    dtype = setup.dtype
    
    ls = setup.ls_levels
    nls = len(ls)
    
    
    # type conversion is here
    
    zf  = setup.exogrid.all_t[t][:,0]
    zm  = setup.exogrid.all_t[t][:,1]
    zftrend = setup.pars['f_wage_trend'][t]
    zmtrend = setup.pars['m_wage_trend'][t]

    psi = setup.exogrid.all_t[t][:,2]
    beta = setup.pars['beta_t'][t]
    sigma = setup.pars['crra_power']
    R = setup.pars['R_t'][t]


    
    nexo = setup.pars['nexo_t'][t]
    shp = (setup.na,nexo,setup.ntheta)
    
    
    wf = np.exp(zf + zftrend)
    wm = np.exp(zm + zmtrend)
    
    
    dtype_here = np.float32 if force_f32 else dtype
    
    if EV_tuple is None:
        EV_by_l, EV_fem_by_l, EV_mal_by_l = np.zeros(((3,) + shp + (nls,)), dtype=dtype )
    else:
        EV_by_l, EV_fem_by_l, EV_mal_by_l = EV_tuple
    
    
    
    
    # type conversion
    sgrid,sigma,beta = (dtype(x) for x in (sgrid,sigma,beta))
    
    V_couple, c_opt, s_opt, x_opt = np.empty((4,)+shp,dtype)
    i_opt, il_opt = np.empty(shp,np.int16), np.empty(shp,np.int16)
    
    V_all_l = np.empty(shp+(nls,),dtype=dtype)
    
    theta_val = dtype(setup.thetagrid)
    
    # the original problem is max{umult*u(c) + beta*EV}
    # we need to rescale the problem to max{u(c) + beta*EV_resc}
    
    istart = 0
    ifinish = nbatch if nbatch < nexo else nexo
    
    # this natually splits everything onto slices
    
    
    
    for ibatch in range(int(np.ceil(nexo/nbatch))):
        #money_i = money[:,istart:ifinish]
        assert ifinish > istart
        
        money_t = (R*agrid, wf[istart:ifinish], wm[istart:ifinish])
        EV_t = (setup.vsgrid_c,EV_by_l[:,istart:ifinish,:,:])
        
        
        V_pure_i, c_opt_i, x_opt_i, s_opt_i, i_opt_i, il_opt_i, V_all_l_i = \
           v_optimize_couple(money_t,sgrid,EV_t,setup.mgrid,
                             setup.ucouple_precomputed_u,setup.ucouple_precomputed_x,
                                 ls,beta,ushift,dtype=dtype_here)
           
        V_ret_i = V_pure_i + psi[None,istart:ifinish,None]
        
        # if dtype_here != dtype type conversion happens here
        
        V_couple[:,istart:ifinish,:] = V_ret_i # this estimate of V can be improved
        c_opt[:,istart:ifinish,:] = c_opt_i 
        s_opt[:,istart:ifinish,:] = s_opt_i
        i_opt[:,istart:ifinish,:] = i_opt_i
        x_opt[:,istart:ifinish,:] = x_opt_i
        il_opt[:,istart:ifinish,:] = il_opt_i
        V_all_l[:,istart:ifinish,:,:] = V_all_l_i # we need this for l choice so it is ok
        
        
        istart = ifinish
        ifinish = ifinish+nbatch if ifinish+nbatch < nexo else nexo
        
        if verbose: print('Batch {} done at {} sec'.format(ibatch,default_timer()-start))
    
    
    assert np.all(c_opt > 0)
    
    psi_r = psi[None,:,None].astype(dtype)
    
    # finally obtain value functions of partners
    uf, um = setup.u_part(c_opt,x_opt,il_opt,theta_val[None,None,:],ushift,psi_r)
    uc = setup.u_couple(c_opt,x_opt,il_opt,theta_val[None,None,:],ushift,psi_r)
    
    
    EVf_all, EVm_all, EV_all  = (setup.vsgrid_c.apply_preserve_shape(x) for x in (EV_fem_by_l, EV_mal_by_l,EV_by_l))
    
    
    
    V_fem = uf + beta*np.take_along_axis(np.take_along_axis(EVf_all,i_opt[...,None],0),il_opt[...,None],3).squeeze(axis=3)
    V_mal = um + beta*np.take_along_axis(np.take_along_axis(EVm_all,i_opt[...,None],0),il_opt[...,None],3).squeeze(axis=3)
    V_all = uc + beta*np.take_along_axis(np.take_along_axis(EV_all,i_opt[...,None],0),il_opt[...,None],3).squeeze(axis=3)
    #def r(x): return x.astype(dtype)
    
    def r(x): return x
    
    assert V_all.dtype == dtype
    assert V_fem.dtype == dtype
    assert V_mal.dtype == dtype
    assert c_opt.dtype == dtype
    assert x_opt.dtype == dtype
    assert s_opt.dtype == dtype
    
    try:
        assert np.allclose(V_all,V_couple,atol=1e-4,rtol=1e-3)
    except:
        #print('max difference in V is {}'.format(np.max(np.abs(V_all-V_couple))))
        pass
    
    return r(V_all), r(V_fem), r(V_mal), r(c_opt), r(x_opt), r(s_opt), il_opt, r(V_all_l)




