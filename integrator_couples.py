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

def ev_couple_m_c(setup,Vpostren,edu,desc,dd,t,marriage,use_sparse=True):
    # computes expected value of couple entering the next period with an option
    # to renegotiate or to break up
    
    can_divorce = setup.pars['can divorce'][t]
    if can_divorce:
        uni_div = setup.div_costs.unilateral_divorce if marriage else setup.sep_costs.unilateral_divorce
        if uni_div:
            # choose your fighter
            #out = v_ren_uni(setup,Vpostren,marriage,t)
            out = v_ren_vt(setup,Vpostren,marriage,dd,edu,desc,t)            
        
    else:
        out = v_no_ren(setup,Vpostren,edu,desc,marriage,dd,t)
    _Vren2 = out['Values']#out.pop('Values') 
    #_Vren2=out['Values']
    dec = out
    
    if _Vren2[0].ndim>2:
        tk = lambda x : x[:,:,setup.theta_orig_on_fine]
    else:
        tk = lambda x : x[:,setup.theta_orig_on_fine]
    
    Vren = {'M':{'VR':tk(_Vren2[0]),'VC':tk(_Vren2[1]), 'VF':tk(_Vren2[2]),'VM':tk(_Vren2[3])},
            'SF':Vpostren[setup.desc_i['f'][edu[0]]],
            'SM':Vpostren[setup.desc_i['m'][edu[1]]]}

    
    # accounts for exogenous transitions
    
    EVr, EVc, EVf, EVm = ev_couple_exo(setup,Vren['M'],edu,desc,dd,t,use_sparse,down=False)
    
    
    assert EVr.dtype == setup.dtype
    
    return (EVr, EVc, EVf, EVm), dec


def ev_couple_exo(setup,Vren,edu,desc,dd,t,use_sparse=True,down=False):
    
 
    # this does dot product along 3rd dimension
    # this takes V that already accounts for renegotiation (so that is e
    # expected pre-negotiation V) and takes expectations wrt exogenous shocks
    
    # floating point math is quirky and can change dtypes occasionally
    def mmult(a,b):
        if use_sparse:
            return (a*b).astype(a.dtype,copy=False)
        else:
            return np.dot(a,b.T).astype(a.dtype,copy=False)
        
    
    nl = len(setup.exogrid.all_t_mat_by_l_spt[edu[0]][edu[1]][dd])
    
    na, nexo, ntheta = setup.na, setup.pars['nexo_t'][t], setup.ntheta 
    
    
    Vr, Vc, Vf, Vm = Vren['VR'], Vren['VC'], Vren['VF'], Vren['VM']
    EVr, EVc, EVf, EVm = np.zeros((4,na,nexo,ntheta,nl),dtype=setup.dtype)
    
    
    
    for il in range(nl):
        
        M = setup.exogrid.all_t_mat_by_l_spt[edu[0]][edu[1]][dd][il][t] if use_sparse else setup.exogrid.all_t_mat_by_l[edu[0]][edu[1]][dd][il][t]
        
        
        
        for itheta in range(ntheta):
            EVr[...,itheta,il]  = mmult(Vr[...,itheta],M)
            EVc[...,itheta,il]  = mmult(Vc[...,itheta],M)
            EVf[...,itheta,il]  = mmult(Vf[...,itheta],M)         
            EVm[...,itheta,il]  = mmult(Vm[...,itheta],M)            
            

    #assert not np.allclose( EV[...,0], EV[...,1])
    
    
    return EVr, EVc, EVf, EVm