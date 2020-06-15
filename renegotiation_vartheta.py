#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:33:42 2020

@author: egorkozlov
"""

import numpy as np
from numba import njit, prange
from gridvec import VecOnGrid
from intratemporal import int_sol


from platform import system
if system() != 'Darwin' and system()!= 'Windows' and system()!= 'Linux':
    ugpu = True
else:
    ugpu = False
    
    

from renegotiation_vartheta_gpu import v_ren_gpu_oneopt, v_ren_gpu_twoopt
        

def v_ren_vt(setup,V,marriage,dd,edu,desc,t,return_extra=False,return_vdiv_only=False,rescale=False,
             thetafun=None, mixed_rescale=False):
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
      
        
    if marriage:
        moneyf = setup.exogrid.zf_t[edu[0]][t]+setup.pars['wtrend']['f'][edu[0]][t]
        moneym = setup.exogrid.zm_t[edu[1]][t]+setup.pars['wtrend']['m'][edu[1]][t]
        
        money_ataxf=np.exp(np.log(1.0-dc.money_lost_f_ez)+(1.0-dc.prog)*np.log(np.exp(moneyf)))
        money_ataxm=np.exp(np.log(1.0-dc.money_lost_m_ez)+(1.0-dc.prog)*np.log(np.exp(moneym)))
        # whf=money_ataxf>np.exp(moneyf)
        # whm=money_ataxm>np.exp(moneym)
        # money_ataxm[whm]=np.exp(moneym)[whm]
        # money_ataxf[whf]=np.exp(moneyf)[whf]
        
        money_ataxf=np.exp(np.log(1.0-dc.money_lost_f_ez)+(1.0-dc.prog)*np.log(np.exp(moneyf)))#+0.605#+0.65
        money_ataxm=np.exp(np.log(1.0-dc.money_lost_m_ez)+(1.0-dc.prog)*np.log(np.exp(moneym)))#+0.605#+0.65
        # whf=money_ataxf>np.exp(moneyf)
        # whm=money_ataxm>np.exp(moneym)
        # money_ataxm[whm]=np.exp(moneym)[whm]
        # money_ataxf[whf]=np.exp(moneyf)[whf] 

        #money_ataxf=np.exp(moneyf)*(1.0-0.45) if edu[0]=='e' else np.exp(moneyf)*(1.0-0.10)
        #money_ataxm=np.exp(moneym)*(1.0-0.45) if edu[1]=='e' else np.exp(moneym)*(1.0-0.10)
    
        
        A=setup.pars['A']
        sig = setup.pars['crra_power']
        alp = setup.pars['util_alp_m']
        xi = setup.pars['util_xi']
        lam = setup.pars['util_lam']
        kap = setup.pars['util_kap_m']
        xf,cf,uf=int_sol(np.exp(moneyf)*setup.mlevel,A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
        xm,cm,um=int_sol(np.exp(moneym)*setup.mlevel,A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
        xfd,cfd,ufd=int_sol(money_ataxf*setup.mlevel,A=A,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
        xmd,cmd,umd=int_sol(money_ataxm*setup.mlevel,A=A,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
        
        V_df=V[setup.desc_i['f'][edu[0]]]['V']-setup.u_single_pub(cf,xf,setup.mlevel)+setup.u_single_pub(cfd,xfd,setup.mlevel)
        V_dm=V[setup.desc_i['m'][edu[1]]]['V']-setup.u_single_pub(cm,xm,setup.mlevel)+setup.u_single_pub(cmd,xmd,setup.mlevel)
  
    else:
    
        moneyf = setup.exogrid.zf_t[edu[0]][t]+setup.pars['wtrend']['f'][edu[0]][t]
        moneym = setup.exogrid.zm_t[edu[1]][t]+setup.pars['wtrend']['m'][edu[1]][t]
        money_ataxf=np.exp(np.log(1.0-0.0)+(1.0-0.0)*moneyf)
        money_ataxm=np.exp(np.log(1.0-0.0)+(1.0-0.0)*moneym)
    
        
        A=setup.pars['A']
        sig = setup.pars['crra_power']
        alp = setup.pars['util_alp_m']
        xi = setup.pars['util_xi']
        lam = setup.pars['util_lam']
        kap = setup.pars['util_kap_m']
        xf,cf,uf=int_sol(np.exp(moneyf)*setup.mlevel,A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
        xm,cm,um=int_sol(np.exp(moneym)*setup.mlevel,A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
        xfd,cfd,ufd=int_sol(money_ataxf*setup.mlevel,A=A,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
        xmd,cmd,umd=int_sol(money_ataxm*setup.mlevel,A=A,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
        
        V_df=V[setup.desc_i['f'][edu[0]]]['V']-setup.u_single_pub(cf,xf,setup.mlevel)+setup.u_single_pub(cfd,xfd,setup.mlevel)
        V_dm=V[setup.desc_i['m'][edu[1]]]['V']-setup.u_single_pub(cm,xm,setup.mlevel)+setup.u_single_pub(cmd,xmd,setup.mlevel)
  
    # #Get value of divorce for men and women
    # if marriage:
    #     moneyf = np.exp(setup.exogrid.zf_t[edu[0]][t]+setup.pars['wtrend']['m'][edu[0]][t])   
    #     moneym = np.exp(setup.exogrid.zf_t[edu[1]][t]+setup.pars['wtrend']['m'][edu[1]][t])
    
        
    #     A=setup.pars['A']
    #     sig = setup.pars['crra_power']
    #     alp = setup.pars['util_alp_m']
    #     xi = setup.pars['util_xi']
    #     lam = setup.pars['util_lam']
    #     kap = setup.pars['util_kap_m']
    #     xf,cf,uf=int_sol(moneyf*setup.mlevel,A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
    #     xm,cm,um=int_sol(moneym*setup.mlevel,A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
    #     xfd,cfd,ufd=int_sol(moneyf*setup.mlevel*(1.0-dc.money_lost_f_ez),A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
    #     xmd,cmd,umd=int_sol(moneym*setup.mlevel*(1.0-dc.money_lost_m_ez),A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=setup.mlevel,mt=1.0-setup.mlevel)
        
    #     V_df=V[setup.desc_i['f'][edu[0]]]['V']-setup.u_single_pub(cf,xf,setup.mlevel)+setup.u_single_pub(cfd,xfd,setup.mlevel)
    #     V_dm=V[setup.desc_i['m'][edu[1]]]['V']-setup.u_single_pub(cm,xm,setup.mlevel)+setup.u_single_pub(cmd,xmd,setup.mlevel)
  
    # else:
        
        # V_df=V[setup.desc_i['f'][edu[0]]]['V']
        # V_dm=V[setup.desc_i['m'][edu[1]]]['V']
        
    # values of divorce
    vf_n, vm_n = v_div_vartheta(setup, dc, dd,t, sc,V_dm,V_df,izf, izm, cost_fem=dc.money_lost_f, cost_mal=dc.money_lost_m, fun=thetafun)
    
    assert vf_n.dtype == setup.dtype
    
    
    if return_vdiv_only:
        return {'Value of Divorce, male': vm_n,
                'Value of Divorce, female': vf_n}
    

    
    itht = setup.v_thetagrid_fine.i
    wntht = setup.v_thetagrid_fine.wnext
    thtgrid = setup.thetagrid_fine
        
    if marriage:        
        
        #Value of the description
        descr=setup.desc_i[edu[0]][edu[1]]['M']
        if not ugpu:
            v_out_nor, vf_out, vm_out, itheta_out, _ = \
             v_ren_core_two_opts_with_int(V[descr]['V'][None,...],
                                          V[descr]['VF'][None,...], 
                                          V[descr]['VM'][None,...], 
                                          vf_n, vm_n,
                                          itht, wntht, thtgrid)
             
        else:
           
            v_out_nor, vf_out, vm_out, itheta_out  = \
                v_ren_gpu_oneopt(V[descr]['V'],
                                 V[descr]['VF'],
                                 V[descr]['VM'],
                              vf_n, vm_n, itht, wntht, thtgrid)
                
               
            
        assert v_out_nor.dtype == setup.dtype
         
    else:
        #Value of the description
        descrc,descrm=setup.desc_i[edu[0]][edu[1]]['C'],setup.desc_i[edu[0]][edu[1]]['M']
        
        if not ugpu:
            v_out_nor, vf_out, vm_out, itheta_out, switch = \
                v_ren_core_two_opts_with_int(
                            np.stack([V[descrc]['V'], V[descrm]['V']]),
                            np.stack([V[descrc]['VF'],V[descrm]['VF']]), 
                            np.stack([V[descrc]['VM'],V[descrm]['VM']]), 
                                    vf_n, vm_n,
                                    itht, wntht, thtgrid)     
                
            # #First: cohabitation versus separation
            # v_out_nor_1, vf_out_1, vm_out_1, itheta_out_1,_ = \
            #     v_ren_core_two_opts_with_int(
            #                 V[descrc]['V'][None,...],
            #                 V[descrc]['VF'][None,...], 
            #                 V[descrc]['VM'][None,...], 
            #                         vf_n, vm_n,
            #                         itht, wntht, thtgrid)  
                
                
            # #Second: Marriage versus envelop
            # v_out_nor, vf_out, vm_out, itheta_out,_ = \
            #     v_ren_core_two_opts_with_int2(
            #                 V[descrm]['V'][None,...],
            #                 V[descrm]['VF'][None,...], 
            #                 V[descrm]['VM'][None,...], 
            #                         vf_out_1, vm_out_1,
            #                         itht, wntht, thtgrid)  
             
                
            # #Get switch
            # switch=(itheta_out>=0)
            
            # #Adjust thetaout
            # nomar=(itheta_out<0)
            # itheta_out[nomar]=itheta_out_1[nomar]
            

            
            
            
        else:
            v_out_nor, vf_out, vm_out, itheta_out, switch = \
                v_ren_gpu_twoopt(V[descrc]['V'], V[descrm]['V'],
                                 V[descrc]['VF'], V[descrm]['VF'],
                                 V[descrc]['VM'], V[descrm]['VM'],
                              vf_n, vm_n, itht, wntht, thtgrid)
        
        assert v_out_nor.dtype == setup.dtype
        
        
    def v_rescale(v,it_out):
    
        vo = v.copy()
        itheta_in = np.broadcast_to(np.arange(thtgrid.size,dtype=np.int16)[None,None,:],it_out.shape)
        stay = (it_out!=-1)
        
        decrease = (it_out < itheta_in) & stay
        f_dec = ((thtgrid[itheta_in[decrease]])/(thtgrid[it_out[decrease]]))
        vo[decrease] = f_dec*vo[decrease]
        assert np.all(f_dec>1)
        increase = (it_out > itheta_in) & stay
        f_inc = ((1-thtgrid[itheta_in[increase]])/(1-thtgrid[it_out[increase]]))
        assert np.all(f_inc>1)
        vo[increase] = f_inc*vo[increase]
        
        return vo
    
    v_resc = v_rescale(v_out_nor,itheta_out) if rescale else v_out_nor
    v_out = v_out_nor if mixed_rescale else v_resc
    
        
    def r(x): return x
        
    result =  {'Decision': (itheta_out>=0), 'thetas': itheta_out,
                'Values': (r(v_resc), r(v_out), r(vf_out), r(vm_out)),'Divorce':(vf_n,vm_n)}
    
    
    
    if not marriage:
        result['Cohabitation preferred to Marriage'] = ~switch
        
       
    extra = {'Values':result['Values'],
             'Value of Divorce, male': vm_n, 'Value of Divorce, female': vf_n}
    
    
    if not return_extra:
        return result
    else:
        return result, extra
    
    



from renegotiation_unilateral import v_div_allsplits


def v_div_vartheta(setup,dc,dd,t,sc,Vmale,Vfemale,izf,izm,
                   cost_fem=0.0,cost_mal=0.0, fun=lambda x : (x,1-x) ):
    # this produces value of divorce for gridpoints given possibly different
    # shares of how assets are divided. 
    # Returns Vf_divorce, Vm_divorce -- values of singles in case of divorce
    # matched to the gridpionts for couples
    
    # optional cost_fem and cost_mal are monetary costs of divorce
    
    
    shrs = setup.thetagrid if len(setup.agrid_s)>1 else np.array([0.5])
    
    # these are interpolation points
    
    Vm_divorce_M, Vf_divorce_M = v_div_allsplits(setup,dc,dd,t,sc,
                                                 Vmale,Vfemale,izm,izf,
                                shrs=shrs,cost_fem=cost_fem,cost_mal=cost_mal)
    

    
    if len(setup.agrid_s)>1:
        
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
    else:
        Vf_divorce = Vf_divorce_M
        Vm_divorce = Vm_divorce_M
        
        return np.reshape(np.repeat(Vf_divorce,setup.ntheta),(1,len(izf),setup.ntheta)),\
                  np.reshape(np.repeat(Vm_divorce,setup.ntheta),(1,len(izm),setup.ntheta))
                
    
    assert Vf_divorce.dtype == setup.dtype
    
    return Vf_divorce, Vm_divorce



@njit(parallel=True)
def v_ren_core_two_opts_with_int(v_y_ni, vf_y_ni, vm_y_ni, vf_n_ni, vm_n_ni, itht, wntht, thtgrid):
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
                    pick_1 = (v_y_1 >= v_y_0)
                    
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
                    
                    
                    
                    vf_y = vf_opt[it_ren]              
                    vm_y = vm_opt[it_ren]
                    v_y  =  v_opt[it_ren]
                    
                    v_out[ia,ie,it] = v_y
                    vf_out[ia,ie,it] = vf_y
                    vm_out[ia,ie,it] = vm_y
                    itheta_out[ia,ie,it] = it_ren
                
    
    return v_out, vf_out, vm_out, itheta_out, ichoice_out

@njit(parallel=True)
def v_ren_core_two_opts_with_int2(v_y_ni, vf_y_ni, vm_y_ni, vf_n_ni, vm_n_ni, itht, wntht, thtgrid):
    # this takes values with no interpolation and interpolates inside
    # this also makes a choice of mar / coh
    # choice is based on comparing v_y_ni_0 vs v_y_ni_1 in the interpolated pt


    # this code is not really elegant but @njit requires some dumb things
    # note that v_y_ni has either 1 or 2 elements at 0th dimension
    # (so if two functions are passed, v_y_ni is np.stack(v_y_c,v_y_m)),
    # otherwise it is just v_y_m[None,...]. x[0] is equivalent to x[0,...].
    
   
  
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
            vf_no_t[:] = vf_n_ni[ia,ie,:]
            vm_no_t[:] = vm_n_ni[ia,ie,:]
            for it in range(nt):
                it_c = itht[it]
                it_cp = it_c+1
                wn_c = wntht[it]
                wt_c = f1 - wn_c
                
                def wsum(x):
                    return x[ia,ie,it_c]*wt_c + x[ia,ie,it_cp]*wn_c
                
                v_y_0 = wsum(v_y_ni_0)
                

                
            
                  
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
                    
                    
                    
                    vf_y = vf_opt[it_ren]              
                    vm_y = vm_opt[it_ren]
                    v_y  =  v_opt[it_ren]
                    
                    v_out[ia,ie,it] = v_y
                    vf_out[ia,ie,it] = vf_y
                    vm_out[ia,ie,it] = vm_y
                    itheta_out[ia,ie,it] = it_ren
                
    
    return v_out, vf_out, vm_out, itheta_out, ichoice_out
