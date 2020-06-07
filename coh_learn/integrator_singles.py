#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This contains routines for intergation for singles
"""

import numpy as np
#import dill as pickle

#from ren_mar_pareto import v_mar_igrid, v_no_mar
#from ren_mar_alt import v_mar_igrid, v_no_marù
from marriage import v_mar_igrid, v_no_mar
    



def ev_single(setup,V,sown,female,ef,em,desc,dd,t,trim_lvl=0.001,decc=None):
    # expected value of single person meeting a partner with a chance pmeet
    pmeet = setup.dtype( setup.pars['pmeet_t'][t] )
    
    skip_mar = (pmeet < 1e-5)
    
    
    # do test here
#    ev_single_meet_test(setup,V,sown,female,t,
#                                      skip_mar=skip_mar,trim_lvl=trim_lvl)
#    
    EV_meet, dec = ev_single_meet(setup,V,sown,female,ef,em,desc,dd,t,
                                      skip_mar=skip_mar,trim_lvl=trim_lvl,dec_c=decc)
    
    
    
    if female:
        M = setup.exogrid.zf_t_mat[ef][t].T
        
    else:
        M = setup.exogrid.zm_t_mat[em][t].T
        
    EV_nomeet =  np.dot(V[desc]['V'],M).astype(setup.dtype)
    
    assert EV_nomeet.dtype == setup.dtype
    assert EV_meet.dtype   == setup.dtype
    
    
    return (1-pmeet)*EV_nomeet + pmeet*EV_meet, dec
    

def ev_single_meet(setup,V,sown,female,ef,em,desc,dd,t,skip_mar=False,trim_lvl=0.000001,dec_c=None):
    # computes expected value of single person meeting a partner
    
    # this creates potential partners and integrates over them
    # this also removes unlikely combinations of future z and partner's 
    # characteristics so we have to do less bargaining
    
    
    
    nexo = setup.pars['nexo_t'][t]
    ns = sown.size
    
    edu=ef if female else em
    eduo=em if female else ef
    p_mat = setup.part_mats[setup.desc[desc]][edu][eduo][t].T 
    p_mat = p_mat.astype(setup.dtype,copy=False)
        
    V_next = np.ones((ns,nexo),dtype=setup.dtype)*(-1e10)
    inds = np.where( np.any(p_mat>-10,axis=1 ) )[0]
    
    
    
    EV = setup.dtype(0.0)
    
    i_assets_c, p_assets_c = setup.i_a_mat[female], setup.prob_a_mat[female]
    
    npart = i_assets_c.shape[1]
    
    
    matches = setup.matches[setup.desc[desc]][ef][em][t] if female else setup.matches[setup.desc[desc]][em][ef][t] 
    
    
    dec = np.zeros(matches['iexo'].shape,dtype=np.bool)
    morc = np.zeros(matches['iexo'].shape,dtype=np.bool)
    tht = -1*np.ones(matches['iexo'].shape,dtype=np.int32)
    iconv = matches['iconv']
    
    for i in range(npart):
        if not skip_mar:
            # try marriage
            res_m = v_mar_igrid(setup,t,V,i_assets_c[:,i],inds,desc,ef,em,
                                     female=female,marriage=True)
            
            
            res_c = v_mar_igrid(setup,t,V,i_assets_c[:,i],inds,desc,ef,em,
                                     female=female,marriage=False)
        else:
            # try marriage
            res_m = v_no_mar(setup,t,V,i_assets_c[:,i],inds,desc,ef,em,
                                     female=female,marriage=True)
            
            
            res_c = v_no_mar(setup,t,V,i_assets_c[:,i],inds,desc,ef,em,
                                     female=female,marriage=False)
        
        
        
        (vfoutm,vmoutm), nprm, decm, thtm = res_m['Values'], res_m['NBS'], res_m['Decision'], res_m['theta']
        
        # try cohabitation
        (vfoutc, vmoutc), nprc, decc, thtc =  res_c['Values'], res_c['NBS'], res_c['Decision'], res_c['theta']
        
        
        # choice is made based on Nash Surplus value
        i_mar =(nprm>=nprc) #((vfoutm>=vfoutc) & (vmoutm>=vfoutc))#((vfoutm>vfoutc) & (vmoutm>vfoutc))#         
        if female:
            vout = i_mar*vfoutm + (~i_mar)*vfoutc
        else:
            vout = i_mar*vmoutm + (~i_mar)*vmoutc
            
            
        #check whether person would marry right away
      #   if decc not None: 
            
#             #Checks
#    thtm=mdl[0].decisions[0][3]['Female, single']['thetam'][0,:]
#    thtc=mdl[0].decisions[0][3]['Female, single']['thetac'][0,:]
#    iexo=mdl[0].decisions[0][3]['Female, single']['iexo'][0,2,:]
#    moc=mdl[0].decisions[0][3]['Female, single']['M or C'][0,2,:]
#    dec=mdl[0].decisions[0][3]['Female, single']['Decision'][0,2,:]
#    coh=((moc==False) & (dec==True))
#    
#    mocc=mdl[0].decisions[0][3]['Couple, C']['Cohabitation preferred to Marriage']
#    
#    mocc[0,iexo[coh],thtm[coh]]
      

        
        assert vout.dtype == setup.dtype
        
       # if dec_c is not None:
        #coh=((i_mar==False) & ((i_mar*decm + (~i_mar)*decc)==True))
        coh=((i_mar*decm + (~i_mar)*decc)==True)
        if np.any(coh):
            indsi=inds[coh[0,:]]
            thtci=thtc[coh]
            ass=np.ones(i_mar[coh].shape,dtype=np.int32)*0
            i_mar[coh]=(~dec_c['Cohabitation preferred to Marriage'][ass,indsi,thtci])
            
        dec[:,:,iconv[:,i]] = (i_mar*decm + (~i_mar)*decc)[:,None,:]        
        tht[:,:,iconv[:,i]] = (i_mar*thtm + (~i_mar)*thtc)[:,None,:]
        morc[:,:,iconv[:,i]] = i_mar[:,None,:]

            
        V_next[:,inds] = vout
        
        EV += (p_assets_c[:,i][:,None])*np.dot(V_next,p_mat)
    
    assert EV.dtype == setup.dtype
    
    mout = matches.copy()
    mout['Decision'] = dec
    mout['M or C'] = morc
    mout['theta'] = tht
    #mout['thetac']=thtc
    #mout['thetam']=thtm
    
    return EV, mout




def ev_single_meet_test(setup,V,sown,female,t,skip_mar=False,trim_lvl=0.000001):
    # computes expected value of single person meeting a partner
    
    # this creates potential partners and integrates over them
    # this also removes unlikely combinations of future z and partner's 
    # characteristics so we have to do less bargaining
    
    nexo = setup.pars['nexo_t'][t]
    ns = sown.size
    
    
    
    iexo = np.arange(nexo)
    iassets_c = np.arange(ns)
    # this just says that grid position of couple = grid position of single fem
    


    res_m = v_mar_igrid(setup,t,V,iassets_c,iexo,
                             female=female,marriage=True)
    
    
    res_c = v_mar_igrid(setup,t,V,iassets_c,iexo,
                             female=female,marriage=False)
    
    (vfoutm,vmoutm), nprm, decm, thtm = res_m['Values'], res_m['NBS'], res_m['Decision'], res_m['theta']
        
    (vfoutc, vmoutc), nprc, decc, thtc =  res_c['Values'], res_c['NBS'], res_c['Decision'], res_c['theta']
    
    i_mar =((nprm>=nprc) & (nprm>0)) # ((vfoutm>vfoutc) & (vmoutm>vfoutc) & (nprm>0))# 
    
 
            
   
    
    print('worked!')
            