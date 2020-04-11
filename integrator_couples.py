#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is integrator for couples

"""

import numpy as np
#from renegotiation import v_last_period_renegotiated, v_renegotiated_loop
#from ren_mar_pareto import v_ren_new, v_no_ren
#from ren_mar_alt import v_ren_new, v_no_ren
from renegotiation_unilateral import v_no_ren   
from renegotiation_unilateral import v_ren_uni
from renegotiation_bilateral import v_ren_bil
from renegotiation_vartheta import v_ren_vt
#from ren_mar_pareto import v_ren_new as ren_pareto

def ev_couple_m_c(setup,Vpostren,t,marriage,use_sparse=True):
    # computes expected value of couple entering the next period with an option
    # to renegotiate or to break up
    
    can_divorce = setup.pars['can divorce'][t]
    if can_divorce:
        uni_div = setup.div_costs.unilateral_divorce if marriage else setup.sep_costs.unilateral_divorce
        if uni_div:
            # choose your fighter
            #out = v_ren_uni(setup,Vpostren,marriage,t)
            out = v_ren_vt(setup,Vpostren,marriage,t)            
        else:
            out = v_ren_bil(setup,Vpostren,marriage,t)
    else:
        out = v_no_ren(setup,Vpostren,marriage,t)
    _Vren2 = out.pop('Values') 
    #_Vren2=out['Values']
    dec = out
    
    
    tk = lambda x : x[:,:,setup.theta_orig_on_fine]
    
    Vren = {'M':{'V':tk(_Vren2[0]),'VF':tk(_Vren2[1]),'VM':tk(_Vren2[2])},
            'SF':Vpostren['Female, single'],
            'SM':Vpostren['Male, single']}

    
    # accounts for exogenous transitions
    
    EV, EVf, EVm = ev_couple_exo(setup,Vren['M'],t,use_sparse,down=False)
    
    
    assert EV.dtype == setup.dtype
    
    return (EV, EVf, EVm), dec


def ev_couple_exo(setup,Vren,t,use_sparse=True,down=False):
    
 
    # this does dot product along 3rd dimension
    # this takes V that already accounts for renegotiation (so that is e
    # expected pre-negotiation V) and takes expectations wrt exogenous shocks
    
    # floating point math is quirky and can change dtypes occasionally
    def mmult(a,b):
        if use_sparse:
            return (a*b).astype(a.dtype,copy=False)
        else:
            return np.dot(a,b.T).astype(a.dtype,copy=False)
        
    
    nl = len(setup.exogrid.all_t_mat_by_l_spt)
    
    na, nexo, ntheta = setup.na, setup.pars['nexo_t'][t], setup.ntheta 
    
    
    V, Vf, Vm = Vren['V'], Vren['VF'], Vren['VM']
    EV, EVf, EVm = np.zeros((3,na,nexo,ntheta,nl),dtype=setup.dtype)
    
    
    for il in range(nl):
        
        M = setup.exogrid.all_t_mat_by_l_spt[il][t] if use_sparse else setup.exogrid.all_t_mat_by_l[il][t]
        
        
        
        for itheta in range(ntheta):
            EV[...,itheta,il]  = mmult( V[...,itheta],M)
            EVf[...,itheta,il] = mmult(Vf[...,itheta],M)         
            EVm[...,itheta,il] = mmult(Vm[...,itheta],M)            
            

    #assert not np.allclose( EV[...,0], EV[...,1])
    
    
    return EV, EVf, EVm