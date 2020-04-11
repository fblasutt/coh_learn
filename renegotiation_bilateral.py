#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:49:58 2020

@author: egorkozlov
"""
import numpy as np
from aux_routines import first_true, last_true
from numba import njit, vectorize
from gridvec import VecOnGrid


def v_ren_bil(setup,V,marriage,t,return_extra=False,return_vdiv_only=False,rescale=True):
    # this returns value functions for couple that entered the period with
    # (s,Z,theta) from the grid and is allowed to renegotiate them or breakup
    # 
    # combine = True creates matrix (n_sc-by-n_inds)
    # combine = False assumed that n_sc is the same shape as n_inds and creates
    # a flat array.
     
    #Get Divorce or Separation Costs
    if marriage:
        dc = setup.div_costs
        is_unil = dc.unilateral_divorce # whether to do unilateral divorce at all
    else:
        dc = setup.sep_costs
        is_unil = dc.unilateral_divorce # whether to do unilateral divorce at all
    
    assert ~is_unil
    
    ind, izf, izm, ipsi = setup.all_indices(t+1)
    
    zfgrid = setup.exo_grids['Female, single'][t+1]
    zmgrid = setup.exo_grids['Male, single'][t+1]
    
    share=(np.exp(zfgrid[izf]+setup.pars['f_wage_trend'][t+1]) / ( np.exp(zmgrid[izm]+setup.pars['m_wage_trend'][t+1]) + np.exp(zfgrid[izf]+setup.pars['f_wage_trend'][t+1]) ) )
    relat=np.ones(share.shape)*0.5
    income_share_f=(1.0*share+0.0*relat).squeeze()
    income_share_f =(np.exp(zfgrid[izf]+setup.pars['f_wage_trend'][t+1]) / ( np.exp(zmgrid[izm]+setup.pars['m_wage_trend'][t+1]) + np.exp(zfgrid[izf]+setup.pars['f_wage_trend'][t+1]) ) ).squeeze()
    
    share_f, share_m = dc.shares_if_split(income_share_f)
   
    
    
    
    # this is the part for bilateral divorce
    a_fem, a_mal = share_f[None,:]*setup.agrid_c[:,None], share_m[None,:]*setup.agrid_c[:,None]
    aleft_c = a_fem + a_mal
    iadiv_fem, iadiv_mal = [np.minimum(np.searchsorted(setup.agrid_s,x),setup.na-1)
                                        for x in [a_fem, a_mal]]
    na_s, nexo, ntheta = setup.na, ind.size, setup.ntheta_fine
    iadiv_fem_full, iadiv_mal_full = [np.broadcast_to(x[...,None],(na_s,nexo,ntheta)) 
                                        for x in [iadiv_fem, iadiv_mal]]
    aleft_c_full = np.broadcast_to(aleft_c[...,None],(na_s,nexo,ntheta))
    
    vf_all_s = V['Female, single']['V'][:,izf]
    vm_all_s = V['Male, single']['V'][:,izm]
        
    
    
    
    sc = setup.agrid_c
    
    from renegotiation_unilateral import v_div_byshare
    # values of divorce
    vf_n, vm_n = v_div_byshare(
        setup, dc, t, sc, share_f, share_m,
        V['Male, single']['V'], V['Female, single']['V'],
        izf, izm, cost_fem=dc.money_lost_f, cost_mal=dc.money_lost_m)
    
    
    
    
    if return_vdiv_only:
        return {'Value of Divorce, male': vm_n,
                'Value of Divorce, female': vf_n}
    
    
    assert vf_n.ndim == vm_n.ndim == 2
    
    
    
    
    expnd = lambda x : setup.v_thetagrid_fine.apply(x,axis=2)
    
    if marriage:
        # if couple is married already
        v_y = expnd(V['Couple, M']['V'])
        vf_y = expnd(V['Couple, M']['VF'])
        vm_y = expnd(V['Couple, M']['VM'])
    else:
        # stay in cohabitation
        v_y_coh = expnd(V['Couple, C']['V'])
        vf_y_coh = expnd(V['Couple, C']['VF'])
        vm_y_coh = expnd(V['Couple, C']['VM'])
        # switch to marriage
        v_y_mar = expnd(V['Couple, M']['V'])
        vf_y_mar = expnd(V['Couple, M']['VF'])
        vm_y_mar = expnd(V['Couple, M']['VM'])
        # switching criterion
        #switch = (vf_y_mar>vf_y_coh) & (vm_y_mar>vm_y_coh)
        switch = (v_y_mar>= v_y_coh)
        
        v_y = switch*v_y_mar + (~switch)*v_y_coh
        vf_y = switch*vf_y_mar + (~switch)*vf_y_coh
        vm_y = switch*vm_y_mar + (~switch)*vm_y_coh
        
    
    result = ren_bilateral_wrap(setup,v_y,vf_y,vm_y,vf_n,vm_n,vf_all_s,vm_all_s,aleft_c_full,                       
                       iadiv_fem_full,iadiv_mal_full,rescale=True)
    
    
    
    if not marriage:
        result['Cohabitation preferred to Marriage'] = ~switch
        
        
    
        
    extra = {'Values':result['Values'],
             'Value of Divorce, male': vm_n, 'Value of Divorce, female': vf_n}
    
    if not return_extra:
        return result
    else:
        return result, extra
    
def ren_bilateral_wrap(setup,vy,vfy,vmy,vfn,vmn,vf_all_s,vm_all_s,aleft_c,                       
                       ia_div_fem,ia_div_mal,rescale=True):
    # v_ren_core_interp(setup,vy,vfy,vmy,vf_n,vm_n,unilateral,show_sc=False,rescale=False)
    tgrid = setup.thetagrid_fine
    
    
    vout, vfout, vmout, thetaout, yes, ithetaout, bribe, iaout_f, iaout_m = \
        ren_loop_bilateral(vy,vfy,vmy,vfn,vmn,
                           vf_all_s,vm_all_s,aleft_c,
                           ia_div_fem,ia_div_mal,
                           setup.agrid_s,
                           tgrid)
    
    #if np.any(bribe):
    #    print('Bribing happens in {}% of divorces'.format(round(100*np.mean(~yes & bribe)/np.mean(~yes))))
    
    def r(x): return x      
    
   # print(aleft_c[bribe]-setup.agrid_s[iaout_f[bribe]]-setup.agrid_s[iaout_m[bribe]])
    return {'Decision': yes, 'thetas': ithetaout,
            'Values': (r(vout), r(vfout), r(vmout)),'Divorce':(vfn,vmn),
            'Bribing':(bribe,iaout_f,iaout_m)}
            #'Bribing':(bribe,ia_div_fem,ia_div_mal)}
    
    
                    

@njit
def ren_loop_bilateral(vy,vfy,vmy,vfn,vmn,vfn_as,vmn_as,aleft_c,ia_f_def_s,ia_m_def_s,agrid_s,thtgrid):
    #print('bilateral hi!')


    #vfn = vfn
    #vmn = vmn
    
    sf = vfy - vfn.reshape(vfn.shape+(1,))
    sm = vmy - vmn.reshape(vmn.shape+(1,))
    
    na, nexo, nt = vy.shape
    
    vout = vy.copy()
    vfout = vfy.copy()
    vmout = vmy.copy()
    
    yes = np.zeros((na,nexo,nt),dtype=np.bool_)
    bribe = np.zeros((na,nexo,nt),dtype=np.bool_)
    
    
    thetaout = -1*np.ones(vout.shape,dtype=np.float32)
    ithetaout = -1*np.ones(vout.shape,dtype=np.int16)
    
    iaout_f = -1*np.ones(vout.shape,dtype=np.int16)
    iaout_m = -1*np.ones(vout.shape,dtype=np.int16)
   
    
    na_s = agrid_s.size
    
    for ia in range(na):
        for ie in range(nexo):
            for it in range(nt):
                
                sf_i = sf[ia,ie,it]
                sm_i = sm[ia,ie,it]
                
                tht = thtgrid[it]
                
                
                if sf_i >= 0 and sm_i >= 0:
                    yes[ia,ie,it] = True
                    thetaout[ia,ie,it] = tht
                    ithetaout[ia,ie,it] = it
                    continue
                else:
                    
                    vout_div_def = tht*vfn[ia,ie] +  (1-tht)*vmn[ia,ie]
                    vfout_div_def = vfn[ia,ie]
                    vmout_div_def = vmn[ia,ie]
                    
                    if sf_i < 0 and sm_i < 0:                        
                        vout[ia,ie,it] = vout_div_def
                        vfout[ia,ie,it] = vfout_div_def
                        vmout[ia,ie,it] = vmout_div_def
                        continue
                    else:
                        # if only one person wants do divorce -- possible
                        # to find reallocation of assets such that both could
                        # agree.
                        ia_m_def = ia_m_def_s[ia,ie,it]
                        ia_f_def = ia_f_def_s[ia,ie,it]
                        
                        ia_m_new = ia_m_def
                        ia_f_new = ia_f_def
                        
                        a_left = aleft_c[ia,ie,it]
                        
                        
                        do_divorce = False
                        found = False
                        

                        if sf_i < 0 and sm_i > 0:
                            # f bribes out
                            #print('at point {} m bribes out'.format((ia,ie,it)))
                            # increase ia_m
                            for ia_m_new in range(ia_m_def+1,na_s):
                                if agrid_s[ia_m_new] > a_left:
                                    break
                                
                                found = False
                                for ia_f_new in range(ia_f_new,-1,-1):
                                    if agrid_s[ia_f_new] + agrid_s[ia_m_new] <= a_left:
                                        found=True
                                        break
                                    
                                if found:
                                    sf_i_new  = vfy[ia,ie,it] - vfn_as[ia_f_new,ie]
                                    sm_i_new  = vmy[ia,ie,it] - vmn_as[ia_m_new,ie]
                                    if sf_i_new < 0 and sm_i_new < 0:
                                        do_divorce = True
                                        #print('divorce happens: f bribes out!')
                                        #print((ia_m_def,sm_i,ia_f_def,sf_i))
                                        #print((ia_f_new,sf_i_new,ia_m_new,sm_i_new))                                       
                                        break
                        
                        
                        if sm_i < 0 and sf_i > 0:
                            # m bribes out
                            # increase ia_f
                            for ia_f_new in range(ia_f_def+1,na_s):
                                if agrid_s[ia_f_new] > a_left:
                                    break
                                
                                found = False
                                for ia_m_new in range(ia_m_new,-1,-1):
                                    if agrid_s[ia_m_new] + agrid_s[ia_f_new] <= a_left:
                                        found=True
                                        break
                                    
                                if found:
                                    sf_i_new  = vfy[ia,ie,it] - vfn_as[ia_f_new,ie]
                                    sm_i_new  = vmy[ia,ie,it] - vmn_as[ia_m_new,ie]
                                    if sf_i_new < 0 and sm_i_new < 0:
                                        do_divorce = True
                                        #print('divorce happens: m bribes out!')
                                        #print((ia_m_def,sm_i,ia_f_def,sf_i))
                                        #print((ia_f_new,sf_i_new,ia_m_new,sm_i_new))
                                        break
                        
                                
                        if not do_divorce:
                            yes[ia,ie,it] = True
                            thetaout[ia,ie,it] = thtgrid[it]
                            ithetaout[ia,ie,it] = it
                            continue
                        
                        # else we do_divorce   
                        assert found
                        bribe[ia,ie,it] = True
                        vfout[ia,ie,it] = vfn_as[ia_f_new,ie]
                        vmout[ia,ie,it] = vmn_as[ia_m_new,ie]
                        vout[ia,ie,it] = tht*vfn_as[ia_f_new,ie] + \
                                        (1-tht)*vmn_as[ia_m_new,ie]
                        
                        iaout_f[ia,ie,it] = ia_f_new
                        iaout_m[ia,ie,it] = ia_m_new
                        
                        
                        continue
                        
  #  print(aleft_c[bribe]-agrid_s[iaout_f[bribe]]-agrid_s[iaout_m[bribe]])
    return vout, vfout, vmout, thetaout, yes, ithetaout, bribe, iaout_f, iaout_m
    