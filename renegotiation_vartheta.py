#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:33:42 2020

@author: egorkozlov
"""

import numpy as np
from numba import njit, prange
from gridvec import VecOnGrid


from platform import system
if system() != 'Darwin' and system()!= 'Windows' and system()!= 'Linux':
    ugpu = True
else:
    ugpu = False
    
    

from renegotiation_vartheta_gpu import v_ren_gpu_oneopt, v_ren_gpu_twoopt
        

def v_ren_vt(setup,V,marriage,t,return_extra=False,return_vdiv_only=False,rescale=True,
             thetafun=None):
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
    
    assert is_unil
    
    ind, izf, izm, ipsi = setup.all_indices(t+1)
    
    sc = setup.agrid_c
    
    if thetafun is None:
        def thetafun(tht): return tht, 1-tht
        #def thetafun(tht): return 0.5*np.ones_like(tht), 0.5*np.ones_like(tht)
            
    
    # values of divorce
    vf_n, vm_n = v_div_vartheta(
        setup, dc, t, sc,
        V['Male, single']['V'], V['Female, single']['V'],
        izf, izm, cost_fem=dc.money_lost_f, cost_mal=dc.money_lost_m, fun=thetafun)
    
    assert vf_n.dtype == setup.dtype
    
    
    if return_vdiv_only:
        return {'Value of Divorce, male': vm_n,
                'Value of Divorce, female': vf_n}
    

    
    itht = setup.v_thetagrid_fine.i
    wntht = setup.v_thetagrid_fine.wnext
    thtgrid = setup.thetagrid_fine
        
    if marriage:        
        
        if not ugpu:
            v_out, vf_out, vm_out, itheta_out, _ = \
             v_ren_core_two_opts_with_int(V['Couple, M']['V'][None,...],
                                          V['Couple, M']['VF'][None,...], 
                                          V['Couple, M']['VM'][None,...], 
                                          vf_n, vm_n,
                                          itht, wntht, thtgrid, 
                                          rescale = rescale)
             
        else:
           
            v_out, vf_out, vm_out, itheta_out  = \
                v_ren_gpu_oneopt(V['Couple, M']['V'],
                                 V['Couple, M']['VF'],
                                 V['Couple, M']['VM'],
                              vf_n, vm_n, itht, wntht, thtgrid)
                
               
            
        assert v_out.dtype == setup.dtype
         
    else:
        
        if not ugpu:
            v_out, vf_out, vm_out, itheta_out, switch = \
                v_ren_core_two_opts_with_int(
                           np.stack([V['Couple, C']['V'], V['Couple, M']['V']]),
                           np.stack([V['Couple, C']['VF'],V['Couple, M']['VF']]), 
                           np.stack([V['Couple, C']['VM'],V['Couple, M']['VM']]), 
                                    vf_n, vm_n,
                                    itht, wntht, thtgrid, rescale = rescale)        
            
        else:
            v_out, vf_out, vm_out, itheta_out, switch = \
                v_ren_gpu_twoopt(V['Couple, C']['V'], V['Couple, M']['V'],
                                 V['Couple, C']['VF'], V['Couple, M']['VF'],
                                 V['Couple, C']['VM'], V['Couple, M']['VM'],
                              vf_n, vm_n, itht, wntht, thtgrid)
        
        assert v_out.dtype == setup.dtype
        
        
    def r(x): return x
        
    result =  {'Decision': (itheta_out>=0), 'thetas': itheta_out,
                'Values': (r(v_out), r(vf_out), r(vm_out)),'Divorce':(vf_n,vm_n)}
    
    
    if not marriage:
        result['Cohabitation preferred to Marriage'] = ~switch
        
       
    extra = {'Values':result['Values'],
             'Value of Divorce, male': vm_n, 'Value of Divorce, female': vf_n}
    
    
    if not return_extra:
        return result
    else:
        return result, extra
    
    



from renegotiation_unilateral import v_div_allsplits


def v_div_vartheta(setup,dc,t,sc,Vmale,Vfemale,izf,izm,
                   cost_fem=0.0,cost_mal=0.0, fun=lambda x : (x,1-x) ):
    # this produces value of divorce for gridpoints given possibly different
    # shares of how assets are divided. 
    # Returns Vf_divorce, Vm_divorce -- values of singles in case of divorce
    # matched to the gridpionts for couples
    
    # optional cost_fem and cost_mal are monetary costs of divorce
    
    
    shrs = setup.thetagrid
    
    # these are interpolation points
    
    Vm_divorce_M, Vf_divorce_M = v_div_allsplits(setup,dc,t,sc,
                                                 Vmale,Vfemale,izm,izf,
                                shrs=shrs,cost_fem=cost_fem,cost_mal=cost_mal)
    
    # share of assets that goes to the female
    # this has many repetative values but it turns out it does not matter much
    
    
    share_fem, share_mal = fun(setup.thetagrid)
    fem_gets = VecOnGrid(np.array(shrs),share_fem)
    mal_gets = VecOnGrid(np.array(shrs),share_mal)
    
    i_fem = fem_gets.i
    wn_fem = fem_gets.wnext
    wt_fem = setup.dtype(1) - wn_fem
    
    i_mal = mal_gets.i
    wn_mal = mal_gets.wnext
    wt_mal = setup.dtype(1) - wn_mal
    
    
    Vf_divorce = wt_fem[None,None,:]*Vf_divorce_M[:,:,i_fem] + \
                     wn_fem[None,None,:]*Vf_divorce_M[:,:,i_fem+1]
    
    Vm_divorce = wt_mal[None,None,:]*Vm_divorce_M[:,:,i_mal] + \
                     wn_mal[None,None,:]*Vm_divorce_M[:,:,i_mal+1]
                
    
    assert Vf_divorce.dtype == setup.dtype
    
    return Vf_divorce, Vm_divorce



@njit(parallel=True)
def v_ren_core_two_opts_with_int(v_y_ni, vf_y_ni, vm_y_ni, vf_n_ni, vm_n_ni, itht, wntht, thtgrid, 
                                 rescale=False):
    # this takes values with no interpolation and interpolates inside
    # this also makes a choice of mar / coh
    # choice is based on comparing v_y_ni_0 vs v_y_ni_1 in the interpolated pt


    # this code is not really elegant but @njit requires some dumb things
    # note that v_y_ni has either 1 or 2 elements at 0th dimension
    # (so if two functions are passed, v_y_ni is np.stack(v_y_c,v_y_m)),
    # otherwise it is just v_y_m[None,...]. x[0] is equivalent to x[0,...].
    
    if v_y_ni.shape[0] == 2:
        nochoice = False
        v_y_ni_0, v_y_ni_1 = v_y_ni[0], v_y_ni[1]
        vf_y_ni_0, vf_y_ni_1 = vf_y_ni[0], vf_y_ni[1]
        vm_y_ni_0, vm_y_ni_1 = vm_y_ni[0], vm_y_ni[1]
    else:
        nochoice = True
        v_y_ni_0 = v_y_ni[0]
        vf_y_ni_0 = vf_y_ni[0]
        vm_y_ni_0 = vm_y_ni[0]
        
    
    dtype = v_y_ni.dtype
    
        
    na, ne, nt_coarse = v_y_ni_0.shape
    nt = thtgrid.size
    
    shp = (na,ne,nt)
    
    v_out = np.empty(shp,dtype=dtype)
    vm_out = np.empty(shp,dtype=dtype)
    vf_out = np.empty(shp,dtype=dtype)
    
    itheta_out = np.full(v_out.shape,-1,dtype=np.int16)
    ichoice_out = np.zeros(v_out.shape,dtype=np.bool_)
    
    
    f1 = np.float32(1)
    
    
    for ia in prange(na):
        for ie in prange(ne):
            # first we form value functions and choices
            # then we do renegotiation
            # this saves lots of operations
            
            v_opt = np.empty((nt,),dtype=dtype)
            vf_opt = np.empty((nt,),dtype=dtype)
            vm_opt = np.empty((nt,),dtype=dtype)
            
            vf_no_t = np.empty((nt,),dtype=dtype)
            vm_no_t = np.empty((nt,),dtype=dtype)
            
            # this part does all interpolations and maximization
            for it in range(nt):
                it_c = itht[it]
                it_cp = it_c+1
                wn_c = wntht[it]
                wt_c = f1 - wn_c
                
                def wsum(x):
                    return x[ia,ie,it_c]*wt_c + x[ia,ie,it_cp]*wn_c
                
                v_y_0 = wsum(v_y_ni_0)
                
                vf_no_t[it] = wsum(vf_n_ni)
                vm_no_t[it] = wsum(vm_n_ni)
                
                
                if not nochoice:
                    v_y_1 = wsum(v_y_ni_1)                
                    pick_1 = (v_y_1 > v_y_0)
                    
                    ichoice_out[ia,ie,it] = pick_1 
                    
                    if pick_1:
                        vf_opt[it] = wsum(vf_y_ni_1)
                        vm_opt[it] = wsum(vm_y_ni_1)
                        v_opt[it] = v_y_1
                    else:
                        vf_opt[it] = wsum(vf_y_ni_0)
                        vm_opt[it] = wsum(vm_y_ni_0)
                        v_opt[it] = v_y_0
                        
                else:                    
                    vf_opt[it] = wsum(vf_y_ni_0)
                    vm_opt[it] = wsum(vm_y_ni_0)
                    v_opt[it] = v_y_0
                
                
            
            for it in range(nt):
                
                
                vf_y = vf_opt[it]                
                vm_y = vm_opt[it]
                v_y = v_opt[it]
                
                vf_no = vf_no_t[it]
                vm_no = vm_no_t[it]
                
                if vf_y >= vf_no and vm_y >= vm_no:
                    # no search just fill the value
                    itheta_out[ia,ie,it] = it    
                    vf_out[ia,ie,it] = vf_y
                    vm_out[ia,ie,it] = vm_y
                    v_out[ia,ie,it] = v_y
                    continue
                    
                if vf_y < vf_no and vm_y < vm_no:
                    # no search
                    tht = thtgrid[it]
                    v_out[ia,ie,it] = tht*vf_no + (f1-tht)*vm_no
                    vf_out[ia,ie,it] = vf_no
                    vm_out[ia,ie,it] = vm_no
                    itheta_out[ia,ie,it] = -1
                    continue
                
                
                # in the points left one guy agrees and one disagrees
                
                # run two loops: forward and backward
                # see if there is anything to replace
                
                it_ren = -1
                
                found_increase = False
                found_decrease = False
                
                
                # these loops can be improved by monotonicity
                for it_increase in range(it+1,nt):   
                    if (vf_opt[it_increase] >= vf_no and vm_opt[it_increase] >= vm_no):
                        found_increase = True
                        break
                
                
                
                for it_decrease in range(it-1,-1,-1):
                    if (vf_opt[it_decrease] >= vf_no and vm_opt[it_decrease] >= vm_no):
                        found_decrease = True
                        break
                    
                
                if found_increase and found_decrease:
                    dist_increase = it_increase - it
                    dist_decrease = it - it_decrease
                    
                    if dist_increase != dist_decrease:
                        it_ren = it_increase if dist_increase < dist_decrease else it_decrease
                    else:
                        # tie breaker
                        dist_mid_inc = np.abs(it_increase - (nt/2))
                        dist_mid_dec = np.abs(it_decrease - (nt/2))
                        it_ren = it_increase if dist_mid_inc < dist_mid_dec else it_decrease
                    
                elif found_increase and not found_decrease:
                    it_ren = it_increase
                elif found_decrease and not found_increase:
                    it_ren = it_decrease
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
                    # here we need to rescale
                    
                    if rescale:
                        tht_old = thtgrid[it]
                        tht_new = thtgrid[it_ren]
                        factor = np.maximum( (1-tht_old)/(1-tht_new), tht_old/tht_new )
                    else:
                        factor = 1
                    
                    
                    vf_y = vf_opt[it_ren]              
                    vm_y = vm_opt[it_ren]
                    v_y  =  v_opt[it_ren]
                    
                    v_out[ia,ie,it] = factor*v_y
                    vf_out[ia,ie,it] = vf_y
                    vm_out[ia,ie,it] = vm_y
                    itheta_out[ia,ie,it] = it_ren
                
    
    return v_out, vf_out, vm_out, itheta_out, ichoice_out
