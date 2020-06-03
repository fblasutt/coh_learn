#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This obtains exact solutions for marriage-cohabitation model.
This allows to flexibly specify divorce costs and allows for savings.

In this code everything is scalar as I do not aim for a great performance

"""

import numpy as np
from scipy.optimize import fminbound   


class DivorceCosts(object):
    # this is something that regulates divorce costs
    # it aims to be fully flexible
    def __init__(self, 
                 unilateral_divorce=True, # whether to allow for unilateral divorce
                 assets_kept = 0.9, # how many assets of couple are splited (the rest disappears)
                 u_lost_m=0.0,u_lost_f=0.0, # pure utility losses b/c of divorce
                 money_lost_m=0.0,money_lost_f=0.0, # pure money (asset) losses b/c of divorce
                 money_lost_m_ez=0.0,money_lost_f_ez=0.0 # money losses proportional to exp(z) b/c of divorce
                 ): # 
        
        self.unilateral_divorce = unilateral_divorce # w
        self.assets_kept = assets_kept
        self.u_lost_m = u_lost_m
        self.u_lost_f = u_lost_f
        self.money_lost_m = money_lost_m
        self.money_lost_f = money_lost_f
        self.money_lost_m_ez = money_lost_m_ez
        self.money_lost_f_ez = money_lost_f_ez
        


def v_couple_t1_renegotiated(setup,a,zf,zm,psi,theta):
    # this is value of couple in the terminal period after theta is determined
    # via renegotiation. This uses a method specified it setup.py
    V, VF, VM = setup.vm_last(a,zm,zf,psi,theta)
    return V, VF, VM

def v_single_t1(setup,a1,z1):
    # this is the value of a single agent in the terminal period. It does not 
    # depend on gender, it only depends on productivity
    return setup.vs_last(a1,z1)

def v_single_t0(setup,a0,z0,female):
    # this is the value of a single agent in period 0. It does depend on 
    # gender, but only through the variance of future producitvity shock.
    # Here I assume that the agent does not meet any partners (the only 
    # possible partner is met before period 0, so this is for those who have
    # chosen to stay single. This problem involves optimal savings and 
    # integration over future z. Sobol sequence of normally distributed shocks
    # is taken from the setup
    
    income = a0 + np.exp(z0)
    beta = setup.pars['beta']
    sig  = setup.pars['crra_power']
    z_nodes = setup.integration['z_nodes']
    sig_z = setup.pars['sig_zf'] if female else setup.pars['sig_zm']
    u = lambda c : c**(1-sig) / (1-sig)
    
    z_next = [z0 + sig_z*z_shock for z_shock in z_nodes]        
        
    def v_of_s(s):
        c = income - s        
        EV = np.mean([v_single_t1(setup,s,zz) for zz in z_next])
        return u(c) + beta*EV
        
    s_min = 0.0
    s_max = 0.9*income
    s_opt = fminbound(lambda s : -v_of_s(s), s_min, s_max,xtol=1e-3)
    c_opt = income - s_opt
    V_opt = v_of_s(s_opt)
    
    
    return V_opt, c_opt, s_opt
    

def v_couple_t1(setup,a,zf,zm,psi,theta,div_costs,assets_share_f,theta_step=0.008,verbose=False):
    # This is the value function of the couple entering the ultimate period, 
    # that has an option to adjust their theta or break up. div_costs specify
    # the divorce protocol, and assets_share_f is how much assets is given to
    # female (supposedly different in marriage and cohabitation and it title-
    # based versus non-title-based regime).
    # This is the slowest part of the codes, lowering theta_step increases
    # precision but slows everything down a lot
    
    V, VF, VM = v_couple_t1_renegotiated(setup,a,zm,zf,psi,theta) # status quo
    
    # in case of divorce
    
    dc = div_costs
    
    
    k_f = dc.assets_kept*assets_share_f
    k_m = dc.assets_kept*(1-assets_share_f)
    
    
    a_f = k_f*a - dc.money_lost_f - dc.money_lost_f_ez*np.exp(zf)
    a_m = k_m*a - dc.money_lost_m - dc.money_lost_m_ez*np.exp(zm)
    
    VF_s = v_single_t1(setup,a_f,zf) - dc.u_lost_f
    VM_s = v_single_t1(setup,a_m,zm) - dc.u_lost_m
    
    if VF > VF_s and VM > VM_s: # if no renegotiation is needed
        if verbose: print('no renegotiation is needed')
        return V, VF, VM, theta
    
    out_div = (theta*VF_s + (1-theta)*VM_s, VF_s, VM_s, None) # value in case of divroce
    
    if VF < VF_s and VM < VM_s: # bilateral divorce
        if verbose: print('bilateral divorce')
        return out_div
    
    if not div_costs.unilateral_divorce:
        if verbose: print('someone is not happy, but no unilateral divorce')
        return V, VF, VM, theta # nothing changes if did not agree bilaterally
        
    
    if (VF-VF_s)*(VM-VM_s) < 0 and div_costs.unilateral_divorce:
        #if VF < VF_s and VM > VM_s:
        #    dtheta = +theta_step
        #else:
        #    dtheta = -theta_step
            
        V_here  = V
        VF_here = VF
        VM_here = VM
        theta_here = theta
        
        theta_min = setup.thetamin
        theta_max = setup.thetamax
            
        if   VF - VF_s < 0: # if initially F is binding
            while VF_here - VF_s < 0 and (theta_here + theta_step < theta_max):
                theta_here = theta_here + theta_step
                V_here, VF_here, VM_here = v_couple_t1_renegotiated(setup,a,zm,zf,psi,theta_here)
        elif VM - VM_s < 0:  # if initially M is binding
            while VM_here - VM_s < 0 and (theta_here - theta_step > theta_min):
                theta_here = theta_here - theta_step
                V_here, VF_here, VM_here = v_couple_t1_renegotiated(setup,a,zm,zf,psi,theta_here)
        else:
            raise Exception('something is wrong')
        
        # check if we got an agreement point
        if (VF_here - VF_s)*(VM_here - VM_s) >= 0:            
            if verbose: print('theta was adjusted! was {}, now {}'.format(theta,theta_here))
            return V_here, VF_here, VM_here, theta_here
        else:
            if VF - VF_s < 0:
                if verbose: print('unilateral divorce: female')
            else:
                if verbose: print('unilateral divorce: male')
                
            return out_div


def v_couple_t0_postmar(setup,a0,zf0,zm0,psi0,theta0,div_costs,assets_share_f):
    # this is value function of couple after it determined theta at marriage.
    # This involves pretty massive integration:
    # we integrate over future zf, zm, psi using Sobol sequence. 
    # number of nodes is chosen in setup.py
    # This also involves choosing optimal savings.
    
    sobol_nodes = setup.integration['large_3dim']
    
    income = a0 + np.exp(zf0) + np.exp(zm0) 
    
    
    beta = setup.pars['beta']
    sig  = setup.pars['crra_power']
    
    umult = setup.u_mult(theta0)
    
    u = lambda c : c**(1-sig) / (1-sig)
    
    sig_zf = setup.pars['sig_zf']
    zf_next = [zf0 + sig_zf*ez for ez in sobol_nodes[:,0]]
    sig_zm = setup.pars['sig_zf']
    zm_next = [zm0 + sig_zm*ez for ez in sobol_nodes[:,1]]
    sig_psi = setup.pars['sigma_psi']
    psi_next = [psi0 + sig_psi*epsi for epsi in sobol_nodes[:,2]]
    
    def v_of_s(s,return_v=False):
        c = income - s        
        vnxt = [v_couple_t1(setup,s,zf,zm,psi,theta0,div_costs,assets_share_f)
                        for zf, zm, psi in zip(zf_next,zm_next,psi_next)]
        EV = np.mean([v[0] for v in vnxt])        
        if not return_v:
            return umult*u(c) + beta*EV
        else:
            return umult*u(c) + beta*EV, vnxt
        
        
    s_min = 0.0
    s_max = 0.9*income
    s_opt = fminbound(lambda s : -v_of_s(s), s_min, s_max,xtol=1e-3)
    c_opt = income - s_opt
    V_opt, vnxt = v_of_s(s_opt,return_v=True)
    V_opt = V_opt + psi0
    
    
    kf, km = setup.c_mult(theta0)
    VF_opt = u(kf*c_opt) + psi0 + beta*np.mean([v[1] for v in vnxt])
    VM_opt = u(km*c_opt) + psi0 + beta*np.mean([v[2] for v in vnxt])   
    assert (theta0*u(kf*c_opt) + (1-theta0)*u(km*c_opt) - umult*u(c_opt)) < 1e-3
    
    
    
    return V_opt, VF_opt, VM_opt, c_opt, s_opt
    
def theta_opt_t0(setup,af,am,zf,zm,psi,div_costs,assets_share_f,gamma_fem=0.5):
    # This determines couple's theta given divorce protocol and rule for 
    # division of assets. It turns out that using fminbound is faster than 
    # computing the surplus for a grid of thetas.
    
    
    
    ac = af + am # couple's assets
    
    
    VF_s = v_single_t0(setup,af,zf,True)[0]
    VM_s = v_single_t0(setup,am,zm,False)[0]
    
    def nbs(theta):
        VF, VM = v_couple_t0_postmar(setup,ac,zf,zm,psi,theta,div_costs,assets_share_f)[1:3]
        SF = VF - VF_s
        SM = VM - VM_s
        if SF < 0 or SM < 0:
            return np.min([SF,SM])
        else:
            return (SF**gamma_fem) * (SM**(1-gamma_fem))
    
    tout = fminbound(lambda t : -nbs(t),setup.thetamin,setup.thetamax,xtol=1e-3)
    if nbs(tout) < 0:
        return None
    else:
        return tout
        
     
def v_partners_t0(setup,af0,am0,zf0,zm0,psi0,div_costs,income_based_share=False,verbose=True):
    # This determines couple's theta given divorce protocol and rule for 
    # division of assets. It turns out that using fminbound is faster than 
    # computing the surplus for a grid of thetas.
    
    
    if income_based_share:
        income_f = af0 + np.exp(zf0)
        income_m = am0 + np.exp(zm0)
        ashare_f = income_f / (income_f + income_m)
    else:
        ashare_f = 0.5
        
    theta = theta_opt_t0(setup,af0,am0,zf0,zm0,psi0,div_costs,ashare_f)
    
    if theta is None:
        # no agreement
        VF = v_single_t0(setup,af0,zf0,True)[0]
        VM = v_single_t0(setup,am0,zm0,False)[0]
        V = -np.inf
        if verbose: print('stayed single')
    else:
        V, VF, VM, c, s = v_couple_t0_postmar(setup,af0+am0,zf0,zm0,psi0,theta,div_costs,assets_share_f=ashare_f)
        if verbose: print('agreed')
        
    return V, VF, VM, theta


if __name__ == '__main__':
    from setup import ModelSetup
    
    
    setup = ModelSetup(nogrid=True,sig_zf=1.2,sig_zm=1.2,sigma_psi=0.0)
    
    div_costs_mar = DivorceCosts(unilateral_divorce=True,u_lost_m=0.5,u_lost_f=0.5)
    div_costs_coh = DivorceCosts(unilateral_divorce=True,u_lost_m=0.0,u_lost_f=0.0)
    
    
    
    #v, vf, vm, c, s = v_couple_t0_postmar(setup,2.0,0.0,0.0,+0.1,0.4,div_costs)
    
    am = 0.0 #1.0
    af = 0.0 #2.0
    zm = 0.5 #1.5
    zf = 0.0 #0.0
    psi = -0.2 #0.2
    
    VF_s, VM_s = v_single_t0(setup,af,zf,True)[0], v_single_t0(setup,am,zm,False)[0]
    
    V_m, VF_m, VM_m, theta_m = v_partners_t0(setup,af,am,zf,zm,psi,div_costs_mar,income_based_share=False)
    V_c, VF_c, VM_c, theta_c = v_partners_t0(setup,af,am,zf,zm,psi,div_costs_coh,income_based_share=True)
    
    print('Marriage:')
    print((V_m, VF_m, VM_m, theta_m))
    print('Cohabitation:')
    print((V_c, VF_c, VM_c, theta_c))
    print('Single:')
    print(('n/a',VF_s,VM_s))
    
    if V_c > V_m: print('couple prefers cohabitation')
    if V_m > V_c: print('couple prefers marriage')
    if VF_c > VF_m: print('female prefers cohabitation')
    if VF_m > VF_c: print('female prefers marriage')
    if VM_c > VM_m: print('male prefers cohabitation')
    if VM_m > VM_c: print('male prefers marriage')
    
    
    
    