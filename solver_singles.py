#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This collects solver for single agents
"""

import numpy as np
#from scipy.optimize import fminbound

#from opt_test import build_s_grid, sgrid_on_agrid, get_EVM
from optimizers import v_optimize_couple
from intratemporal import int_sol


def v_iter_single(setup,dd,t,EV,female,edu,ushift,force_f32=False):
    
    agrid_s = setup.agrid_s
    sgrid_s = setup.sgrid_s
    
    
    dtype = setup.dtype
    
    
    using_pre_u=setup.usinglef_precomputed_u if female else setup.usinglem_precomputed_u
    using_pre_x=setup.usinglef_precomputed_x if female else setup.usinglem_precomputed_x
    zvals = setup.exogrid.zf_t[edu][t] if female else setup.exogrid.zm_t[edu][t]
    ztrend = setup.pars['wtrend']['f'][edu][t] if female else setup.pars['wtrend']['m'][edu][t]
    #sigma = setup.pars['crra_power']
    beta = setup.pars['beta_t'][t]
    R = setup.pars['R_t'][t]
    
    
    dtype_here = np.float32 if force_f32 else dtype

    
    
    
    ls = np.array([setup.ls_levels[-1]],dtype=dtype) if female else np.array([setup.mlevel],dtype=dtype)
    money_t = (R*agrid_s,np.exp(zvals + ztrend),np.zeros_like(zvals))
    
    
    if EV is None:
        EV = np.zeros((agrid_s.size,zvals.size),dtype=dtype_here)
    else:
        EV = EV.astype(dtype_here,copy=False)
    
    assert EV.dtype == dtype_here
    
    # V_01, c_opt1, x_opt1, s_opt1, i_opt1, _, _ = \
    #     v_optimize_couple(money_t,sgrid_s,(setup.vsgrid_s,EV[:,:,None,None]),setup.mgrid,
    #                           using_pre_u[:,None,None],
    #                           using_pre_x[:,None,None],
    #                               ls,beta,ushift,dtype=dtype)
    
    
    
    # V_01, c_opt1, x_opt1, s_opt1, i_opt1 =  \
    #     (x.squeeze(axis=2) for x in [V_01, c_opt1, x_opt1, s_opt1, i_opt1])
    
    A=setup.pars['A']
    sig = setup.pars['crra_power']
    alp = setup.pars['util_alp_m']
    xi = setup.pars['util_xi']
    lam = setup.pars['util_lam']
    kap = setup.pars['util_kap_m']
    x_opt,c_opt,u=int_sol(money_t[1]*ls[0],A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=ls,mt=1.0-ls[0])
    i_opt=np.zeros(x_opt.shape,dtype=np.int16)
    s_opt=np.zeros(x_opt.shape,dtype=np.float64)
    
    EVexp =  EV if isinstance(setup.vsgrid_s,(np.ndarray)) else setup.vsgrid_s.apply_preserve_shape(EV)
    #V_ret1 = setup.u_single_pub(c_opt1,x_opt1,ls) + ushift + beta*np.take_along_axis(EVexp,i_opt1,0)
    V_ret = setup.u_single_pub(c_opt,x_opt,ls) + ushift + beta*np.take_along_axis(EVexp,np.expand_dims(i_opt,axis=0),0)
    
    #print(1111,np.min(V_ret-V_ret1))
    assert V_ret.dtype==dtype
    
    def r(x): return x
    
    return r(V_ret), r(np.expand_dims(c_opt,axis=0)), r(np.expand_dims(x_opt,axis=0)), r(np.expand_dims(s_opt,axis=0))

