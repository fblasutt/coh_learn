#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is solver for those who are couples at period 0
"""
import numpy as np
from timeit import default_timer
from intratemporal import int_sol

from optimizers import v_optimize_couple

from platform import system

if system() != 'Darwin' and system() != 'Windows':    
    nbatch_def = 500    
else:    
    nbatch_def = 17

def v_iter_couple(setup,dd,t,EV_tuple,edu,ushift,nbatch=nbatch_def,verbose=False,
                              force_f32 = False):
    
    if verbose: start = default_timer()
    
    agrid = setup.agrid_c
    sgrid = setup.sgrid_c
    
    dtype = setup.dtype
    
    ls = setup.ls_levels
    nls = len(ls)
 
    #Get education
    e=edu[0] #Female
    eo=edu[1] #Male
    
    # type conversion is here   
    zf  = setup.exogrid.all_t[e][eo][dd][t][:,0]
    zm  = setup.exogrid.all_t[e][eo][dd][t][:,1]
    zftrend = setup.pars['wtrend']['f'][e][t]
    zmtrend = setup.pars['wtrend']['m'][eo][t]

    psi = setup.exogrid.all_t[e][eo][dd][t][:,2]
    beta = setup.pars['beta_t'][t]
    sigma = setup.pars['crra_power']
    R = setup.pars['R_t'][t]


    
    nexo = setup.pars['nexo_t'][t]
    shp = (setup.na,nexo,setup.ntheta)
    
    
    wf = np.exp(zf + zftrend)
    wm = np.exp(zm + zmtrend)
    
    
    dtype_here = np.float32 if force_f32 else dtype
    
    if EV_tuple is None:
        EVr_by_l, EVc_by_l, EV_fem_by_l, EV_mal_by_l = np.zeros(((4,) + shp + (nls,)), dtype=dtype )
    else:
        EVr_by_l, EVc_by_l, EV_fem_by_l, EV_mal_by_l = EV_tuple
    
    
    
    
    
    
    # type conversion
    sgrid,sigma,beta = (dtype(x) for x in (sgrid,sigma,beta))
    
    if not isinstance(sgrid,np.ndarray):sgrid=np.array([0.0],dtype=dtype(sgrid))
    
    #Attention to type of sgrid
    
    
    V_couple, c_opt, s_opt, x_opt = np.empty((4,)+shp,dtype)
    i_opt, il_opt = np.empty(shp,np.int16), np.empty(shp,np.int16)
    
    V_all_l = np.empty(shp+(nls,),dtype=dtype)
    
    theta_val = dtype(setup.thetagrid)
    
    # the original problem is max{umult*u(c) + beta*EV}
    # we need to rescale the problem to max{u(c) + beta*EV_resc}
    
    istart = 0
    ifinish = nbatch if nbatch < nexo else nexo
    
    #Time husband contribute to build Q
    mt=1.0-setup.mlevel
    
    # this natually splits everything onto slices
    
    
    
#    for ibatch in range(int(np.ceil(nexo/nbatch))):
#        #money_i = money[:,istart:ifinish]
#        assert ifinish > istart
#       
#        money_t = (R*agrid, wf[istart:ifinish], wm[istart:ifinish])
#        EV_t = (setup.vsgrid_c,EVr_by_l[:,istart:ifinish,:,:])
#       
#       
#        V_pure_i, c_opt_i, x_opt_i, s_opt_i, i_opt_i, il_opt_i, V_all_l_i = \
#            v_optimize_couple(money_t,sgrid,EV_t,setup.mgrid,
#                              setup.ucouple_precomputed_u,setup.ucouple_precomputed_x,
#                                  ls,beta,ushift,dtype=dtype_here,mt=mt)
#          
#        V_ret_i = V_pure_i + psi[None,istart:ifinish,None]*setup.pars['multpsi']
#       
#        # if dtype_here != dtype type conversion happens here
#       
#        V_couple[:,istart:ifinish,:] = V_ret_i # this estimate of V can be improved
#        c_opt[:,istart:ifinish,:] = c_opt_i 
#        s_opt[:,istart:ifinish,:] = s_opt_i
#        i_opt[:,istart:ifinish,:] = i_opt_i
#        x_opt[:,istart:ifinish,:] = x_opt_i
#        il_opt[:,istart:ifinish,:] = il_opt_i
#        V_all_l[:,istart:ifinish,:,:] = V_all_l_i # we need this for l choice so it is ok
#       
#     
#        istart = ifinish
#        ifinish = ifinish+nbatch if ifinish+nbatch < nexo else nexo
#       
#        if verbose: print('Batch {} done at {} sec'.format(ibatch,default_timer()-start))
#   
#
#
#    #x_opt_n,c_opt_n,x_opt_w,c_opt_w=x_opt_n,c_opt_n,x_opt_w,c_opt_w
#   
#    assert np.all(c_opt > 0)
#   
#    psi_r = psi[None,:,None].astype(setup.dtype,copy=False)*setup.pars['multpsi']
#   
# 
#    # finally obtain value functions of partners
#    uf, um = setup.u_part(c_opt,x_opt,il_opt,theta_val[None,None,:],ushift,psi_r)
#    uc = setup.u_couple(c_opt,x_opt,il_opt,theta_val[None,None,:],ushift,psi_r)
#   
#    if isinstance(setup.vsgrid_c,(np.ndarray)):
#        EVf_all, EVm_all, EVc_all=EV_fem_by_l, EV_mal_by_l,EVc_by_l
#    else:
#        EVf_all, EVm_all, EVc_all  = (setup.vsgrid_c.apply_preserve_shape(x) for x in (EV_fem_by_l, EV_mal_by_l,EVc_by_l))
#   
#    V_fem = uf + beta*np.take_along_axis(np.take_along_axis(EVf_all,i_opt[...,None],0),il_opt[...,None],3).squeeze(axis=3)
#    V_mal = um + beta*np.take_along_axis(np.take_along_axis(EVm_all,i_opt[...,None],0),il_opt[...,None],3).squeeze(axis=3)
#    V_all = uc + beta*np.take_along_axis(np.take_along_axis(EVc_all,i_opt[...,None],0),il_opt[...,None],3).squeeze(axis=3)
#    def r(x): return x
    
     ###########################################
     #New Part
     ###########################################
    A=setup.pars['A']
    sig = setup.pars['crra_power']
    alp = setup.pars['util_alp_m']
    xi = setup.pars['util_xi']
    lam = setup.pars['util_lam']
    kap = setup.pars['util_kap_m']
    
    #Get money
    money_w=wm*setup.mlevel+wf*ls[1]
    money_nw=wm*setup.mlevel+wf*ls[0]
    #moneyo=money_nw[None,...,None]*(1-il_opt)+il_opt*money_w[None,...,None]  
    x_opt_w,c_opt_w,u_w,x_opt_n,c_opt_n,u_n=np.zeros(shp),np.zeros(shp),np.zeros(shp),np.zeros(shp),np.zeros(shp),np.zeros(shp)
    for itheta in range(len(setup.thetagrid)):
        x_opt_w[0,:,itheta],c_opt_w[0,:,itheta],u_w[0,:,itheta]=int_sol(money_w,A= setup.u_mult(setup.thetagrid[itheta]),alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=ls[1],mt=1.0-setup.mlevel)
        x_opt_n[0,:,itheta],c_opt_n[0,:,itheta],u_n[0,:,itheta]=int_sol(money_nw,A=setup.u_mult(setup.thetagrid[itheta]),alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=ls[0],mt=1.0-setup.mlevel)
       
    ilw=np.ones(shp,dtype=np.int8)
    iln=np.zeros(shp,dtype=np.int8)
    
    psi_r = psi[None,:,None].astype(setup.dtype,copy=False)*setup.pars['multpsi']
    
    uw = setup.u_couple(c_opt_w,x_opt_w,ilw,theta_val[None,None,:],ushift,psi_r)
    un = setup.u_couple(c_opt_n,x_opt_n,iln,theta_val[None,None,:],ushift,psi_r)
       
    
    if isinstance(setup.vsgrid_c,(np.ndarray)):
        EVf_all, EVm_all, EVc_all=EV_fem_by_l, EV_mal_by_l,EVc_by_l
    else:
        EVf_all, EVm_all, EVc_all  = (setup.vsgrid_c.apply_preserve_shape(x) for x in (EV_fem_by_l, EV_mal_by_l,EVc_by_l))
    
    #Ger labor supply
    ilb=np.array(un + beta*EVr_by_l[:,:,:,0]<uw + beta*EVr_by_l[:,:,:,1],dtype=np.int8)
    
    #Get everything now
    
    money=money_nw[None,...,None]*(1-ilb)+ilb*money_w[None,...,None]
    c_opt_nw=c_opt_n*(1.0-ilb)+(ilb)*c_opt_w
    x_opt_nw=x_opt_n*(1.0-ilb)+(ilb)*x_opt_w
    s_opt_nw=np.zeros(c_opt_n.shape,dtype=dtype)
    uf_nw, um_nw = setup.u_part(c_opt_nw,x_opt_nw,ilb,theta_val[None,None,:],ushift,psi_r)
    uc_nw = setup.u_couple(c_opt_nw,x_opt_nw,ilb,theta_val[None,None,:],ushift,psi_r)
    #il_opt=ilb.copy()
    
    
    V_fem_nw=uf_nw+beta*(EVf_all[:,:,:,0]*(1.0-ilb)+(ilb)*EVf_all[:,:,:,1])
    V_mal_nw=um_nw+beta*(EVm_all[:,:,:,0]*(1.0-ilb)+(ilb)*EVm_all[:,:,:,1])
    V_all_nw=uc_nw+beta*(EVc_all[:,:,:,0]*(1.0-ilb)+(ilb)*EVc_all[:,:,:,1])
    
       
    # #def r(x): return x.astype(dtype)
    
    def r(x): return x
    
    assert V_all_nw.dtype == dtype
    assert V_fem_nw.dtype == dtype
    assert V_mal_nw.dtype == dtype
    assert c_opt_nw.dtype == dtype
    assert x_opt_nw.dtype == dtype
    assert s_opt_nw.dtype == dtype
    
    V_all_l_nw=np.stack((un + beta*EVc_all[:,:,:,0],uw + beta*EVc_all[:,:,:,1]),axis=3)
    
    # try:
    # assert np.allclose(V_all,V_couple,atol=1e-4,rtol=1e-3)
    # except:
    # #print('max difference in V is {}'.format(np.max(np.abs(V_all-V_couple))))
    # pass
   
    return r(V_all_nw), r(V_fem_nw), r(V_mal_nw), r(c_opt_nw), r(x_opt_nw), r(s_opt_nw), ilb, r(V_all_l_nw)
    #return r(V_all), r(V_fem), r(V_mal), r(c_opt), r(x_opt), r(s_opt), il_opt, r(V_all_l)
    


