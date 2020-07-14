#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This contains things relevant for setting up the model
"""

import numpy as np

from rw_approximations import rouw_nonst, normcdf_tr,tauchen_nonst#,tauchen_nonstm
from rw_approximations import rouw_nonstm as tauchen_nonstm
from mc_tools import mc_simulate,combine_matrices_two_lists,combine_matrices_two_listsf, int_prob,cut_matrix
from scipy.stats import norm
from collections import namedtuple
from gridvec import VecOnGrid
from statutils import kalman
from scipy import optimize
from scipy import sparse
from copy import deepcopy



import gc
import os 
import psutil

gc.disable()


class ModelSetup(object):
    def __init__(self,nogrid=False,divorce_costs='Default',separation_costs='Default',draw=False,**kwargs): 
        p = dict()       
        period_year=1#this can be 1,2,3 or 6
        transform=1#this tells how many periods to pull together for duration moments
        T =int(64/period_year)# int(24/period_year)# 
        Tret = int(48/period_year) #int(18/period_year)#first period when the agent is retired
        Tbef=int(2/period_year)
        Tren =  int(48/period_year) #int(18/period_year)#int(48/period_year)## int(42/period_year) # period starting which people do not renegotiate/divroce
        Tmeet = int(48/period_year) #int(18/period_year)#int(48/period_year)#int(18/period_year)#i int(42/period_year) # period starting which you do not meet anyone
        dm=7#11
        
        #Measure of People
        p['Nfe']=np.array([min(0.32+0.00*t,1.0) for t in range(T)])#*T)
        p['Nfn']=1.0-p['Nfe']#0.3
        p['Nme']=np.array([min(0.264+0.00*t,1.0) for t in range(T)])#np.array([0.25]*T)
        p['Nmn']=1-p['Nme']
        p['ass']=0.53#0.57
        p['dm']=dm
        p['py']=period_year
        p['ty']=transform
        p['T'] = T
        p['Tret'] = Tret
        p['Tren'] = Tren
        p['Tbef'] = Tbef
        p['rho_s']    =0.0#  0.15 hear, 0.127 voena, 
        p['sig_zf_0']  = {'e':.5694464,'n':.6121695}
        p['sig_zf']    = {'e':.0261176**(0.5),'n':.0149161**(0.5)}
        #p['sig_zf']    = {'e':.0141176**(0.5),'n':.0141176**(0.5)}
        #p['sig_zf']    = {'e':.0261176**(0.5),'n':.0269161**(0.5)}
        p['sig_zm_0']  =  {'e':.5673833,'n':.5504325}
        p['sig_zm']    =  {'e':.0316222**(0.5),'n':.0229727**(0.5)}
        #p['sig_zm']    =  {'e':.0226222**(0.5),'n':.0226222**(0.5)}
        #p['sig_zm']    =  {'e':.0316222**(0.5),'n':.0316222**(0.5)}
        p['n_zf_t']      = [6]*Tret + [6]*(T-Tret)
        p['n_zm_t']      = [3]*Tret + [3]*(T-Tret)
        p['n_zf_correct']=3
        p['sigma_psi_mult'] = 0.28
        p['sigma_psi_mu_pre'] = 0.1#1.0#nthe1.1
        p['sigma_psi']   =0.0# 0.11
        p['multpsi']   = 1.0
        p['R_t'] = [1.02**period_year]*T
        p['n_psi_t']     = [21]*T#[21]*T
        p['beta_t'] = [0.98**period_year]*T
        p['A'] =1.0 # consumption in couple: c = (1/A)*[c_f^(1+rho) + c_m^(1+rho)]^(1/(1+rho))
        p['crra_power'] = 1.5
        p['couple_rts'] = 0.0
        p['sigma_psi_init_k']=0.0449626592#1.0
        p['sig_partner_a'] = 0.1#0.5
        p['sig_partner_z'] = 1.2#1.0#0.4 #This is crazy powerful for the diff in diff estimate
        p['sig_partner_mult'] = 1.15#1.15
        p['dump_factor_z'] =0.48# 0.3#0.8#0.78#0.85#0.8
        p['mean_partner_z_female'] =-0.25#
        p['mean_partner_z_male'] = -0.2#-0.3
        p['mean_partner_a_female'] = -0.0#0.1
        p['mean_partner_a_male'] = 0.0#-0.1
        p['m_bargaining_weight'] = 0.5
        p['pmeete'] = 0.2#0.55
        p['pmeetn'] = 0.8
        p['pmeet1'] = -0.0
        p['correction']=0.0
        
        p['z_drift'] = -0.15#-0.2
        
        
        
        p['wage_gap'] = 0.6
        p['wret'] = 0.6#0.5
        p['uls'] = 0.2
        p['pls'] = 1.0
        
        
        
        p['u_shift_mar'] = 0.0
        p['u_shift_coh'] =0.0
        
         

                 # #Wages over time-partner
        p['wtrend']=dict()
        p['wtrend']['f'],p['wtrend']['m']=dict(),dict()       

      
      
        # p['wtrend']['f']['e'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.6171558 +(.03626223)*(t+p['Tbef'])-.0005829*(t+p['Tbef'])**2+.0*(t+p['Tbef'])**3) for t in range(T)]
        # p['wtrend']['f']['n'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.3188577 +(.02013329)*(t+p['Tbef']) -.00023405*(t+p['Tbef'])**2+.0*(t+p['Tbef'])**3) for t in range(T)]
       
        # p['wtrend']['m']['e'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.3854005  +(.11658547)*(t+2+p['Tbef']) -.00314674*(t+2+p['Tbef'])**2+ .00002645*(t+2+p['Tbef'])**3) for t in range(T)]
        # p['wtrend']['m']['n'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.4193351   +(.06240424)*(t+2+p['Tbef']) -.00160054*(t+2+p['Tbef'])**2+ .00001287*(t+2+p['Tbef'])**3) for t in range(T)]
       
        # #Wages over time-partner
        # p['wtrendp']=dict()
        # p['wtrendp']['f'],p['wtrendp']['m']=dict(),dict()
      
        # p['wtrendp']['f']['e'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.5173924 +(.10129988)*(t+p['Tbef'])-.00350539*(t+p['Tbef'])**2+.00003795*(t+p['Tbef'])**3) for t in range(T)]
        # p['wtrendp']['f']['n'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.3182552 +(.0457936)*(t+p['Tbef']) -.00124753*(t+p['Tbef'])**2+.00001292*(t+p['Tbef'])**3) for t in range(T)]
       
        
        # p['wtrendp']['m']['e'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.3982928  +(.1340109)*(t+2+p['Tbef']) -.00502756*(t+2+p['Tbef'])**2+ .00005841*(t+2+p['Tbef'])**3) for t in range(T)]
        # p['wtrendp']['m']['n'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.4705215   +(.05736889)*(t+2+p['Tbef']) -.00202793*(t+2+p['Tbef'])**2+ .00002582*(t+2+p['Tbef'])**3) for t in range(T)]
       

        
        p['wtrend']['f']['e'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.6171558 +(.03626223)*(t+p['Tbef']+1)-.0005829*(t+p['Tbef']+1)**2+.0*(t+p['Tbef']+1)**3) for t in range(T)]
        p['wtrend']['f']['n'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.3188577 +(.02013329)*(t+p['Tbef']+1)-.00023405*(t+p['Tbef']+1)**2+.0*(t+p['Tbef']+1)**3) for t in range(T)]
        #p['wtrend']['f']['n'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.6171558 +(.03626223)*(t+p['Tbef']+1)-.0005829*(t+p['Tbef']+1)**2+.0*(t+p['Tbef']+1)**3) for t in range(T)]
        #p['wtrend']['f']['e'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.3188577 +(.02013329)*(t+p['Tbef']+1)-.00023405*(t+p['Tbef']+1)**2+.0*(t+p['Tbef']+1)**3) for t in range(T)]
        
        p['wtrend']['m']['e'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.3854005  +(.11658547)*(t+p['Tbef']+1) -.00314674*(t+p['Tbef']+1)**2+ .00002645*(t+p['Tbef']+1)**3) for t in range(T)]
        p['wtrend']['m']['n'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.4193351   +(.06240424)*(t+p['Tbef']+1) -.00160054*(t+p['Tbef']+1)**2+ .00001287*(t+p['Tbef']+1)**3) for t in range(T)]
        #p['wtrend']['m']['n'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.3854005  +(.11658547)*(t+p['Tbef']+1) -.00314674*(t+p['Tbef']+1)**2+ .00002645*(t+p['Tbef']+1)**3) for t in range(T)]
        #p['wtrend']['m']['e'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.4193351   +(.06240424)*(t+p['Tbef']+1) -.00160054*(t+p['Tbef']+1)**2+ .00001287*(t+p['Tbef']+1)**3) for t in range(T)]
       
         #Wages over time-partner
        p['wtrendp']=dict()
        p['wtrendp']['f'],p['wtrendp']['m']=dict(),dict()
      
        p['wtrendp']['f']['e'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.5173924 +(.10129988)*(t+p['Tbef']+1)-.00350539*(t+p['Tbef']+1)**2+.00003795*(t+p['Tbef']+1)**3) for t in range(T)]
        p['wtrendp']['f']['n'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.3182552 +(.0457936)*(t+p['Tbef']+1) -.00124753*(t+p['Tbef']+1)**2+.00001292*(t+p['Tbef']+1)**3) for t in range(T)]
        #p['wtrendp']['f']['n'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.5173924 +(.10129988)*(t+p['Tbef']+1)-.00350539*(t+p['Tbef']+1)**2+.00003795*(t+p['Tbef']+1)**3) for t in range(T)]
        #p['wtrendp']['f']['e'] =[t*p['correction']*(t<Tret)+0.0*(t>=Tret)+(t<Tret)*(2.3182552 +(.0457936)*(t+p['Tbef']+1) -.00124753*(t+p['Tbef']+1)**2+.00001292*(t+p['Tbef']+1)**3) for t in range(T)]
        
        p['wtrendp']['m']['e'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.3982928  +(.1340109)*(t+p['Tbef']+1) -.00502756*(t+p['Tbef']+1)**2+ .00005841*(t+p['Tbef']+1)**3) for t in range(T)]
        p['wtrendp']['m']['n'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.4705215   +(.05736889)*(t+p['Tbef']+1) -.00202793*(t+p['Tbef']+1)**2+ .00002582*(t+p['Tbef']+1)**3) for t in range(T)]
       # p['wtrendp']['m']['n'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.3982928  +(.1340109)*(t+p['Tbef']+1) -.00502756*(t+p['Tbef']+1)**2+ .00005841*(t+p['Tbef']+1)**3) for t in range(T)] 
        #p['wtrendp']['m']['e'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.4705215   +(.05736889)*(t+p['Tbef']+1) -.00202793*(t+p['Tbef']+1)**2+ .00002582*(t+p['Tbef']+1)**3) for t in range(T)]
  
        #p['wtrend']['f']['e'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.3854005  +(.11658547)*(t+p['Tbef']+1) -.00314674*(t+p['Tbef']+1)**2+ .00002645*(t+p['Tbef']+1)**3) for t in range(T)]
        #p['wtrend']['f']['n'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.4193351   +(.06240424)*(t+p['Tbef']+1) -.00160054*(t+p['Tbef']+1)**2+ .00001287*(t+p['Tbef']+1)**3) for t in range(T)]
        #p['wtrendp']['f']['e'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.3982928  +(.1340109)*(t+p['Tbef']+1) -.00502756*(t+p['Tbef']+1)**2+ .00005841*(t+p['Tbef']+1)**3) for t in range(T)]
        #p['wtrendp']['f']['n'] = [t*p['correction']*(t<Tret-2)+0.0*(t>=Tret-2)+(t<Tret-2)*(2.4705215   +(.05736889)*(t+p['Tbef']+1) -.00202793*(t+p['Tbef']+1)**2+ .00002582*(t+p['Tbef']+1)**3) for t in range(T)]
    
  
    
  
        p['util_lam'] = 0.189#0.4
        p['util_alp'] = 0.5
        p['util_xi'] = 1.5
        p['util_kap'] = (1-0.206)/(0.206)
        
        #To be consistent with Greenwood divide alpha by (theta=0.26)**((1-xi)/lam)
        p['rprice_durables'] = 1.0#
        p['u_shift']=0.0
        

        
        for key, value in kwargs.items():
            assert (key in p), 'wrong name?'
            p[key] = value
            
            
        p['u_shift_mar'] = p['u_shift']
        p['u_shift_coh'] =p['u_shift']#-0.0016#-0.1
        #Adjust kappa and alpha to make sense of relative prices
        p['util_alp_m']=p['util_alp']*(1.0/(p['rprice_durables'])**(1.0-p['util_xi']))
        p['util_kap_m']=p['util_kap']*p['rprice_durables']**p['util_lam']
        p['sigma_psi_init']=p['sigma_psi_init_k'].copy()
        p['sigma_psi_mu']=p['sigma_psi_init_k']*p['sigma_psi_mu_pre']
            

        #Get the probability of meeting, adjusting for year-period
        p_meete=p['pmeete']
        p_meetn=p['pmeetn']
        p_meet1=p['pmeet1']
        for j in range(period_year-1):
            p_meet=p_meet+(1-p_meet)*p['pmeet']
            
            
        # timing here 
        p['pmeet_t']=dict()
        p['pmeet_t']['n'] =[0.0*(t>=Tmeet)+(t<Tret)*(min(max(p_meetn+p_meet1*t,0),1)) for t in range(T)]#[0.0*(t>=Tmeet)+(t<Tret)*1 for t in range(T)]
        p['pmeet_t']['e'] =[0.0*(t>=Tmeet)+(t<Tret)*(min(max(p_meete+p_meet1*t,0),1)) for t in range(T)]
        p['can divorce'] = [True]*Tren + [False]*(T-Tren)
        

        
        
        def f(x): 
            return (x**(3/2) - (p['sigma_psi_mu']**2+x)*p['sigma_psi_init_k'])#(p['sigma_psi_mult']*p['sigma_psi']))
            #return (np.sqrt(x)*x)-(p['sigma_psi_mu']**2+x)*(p['sigma_psi_mult']*p['sigma_psi'])
        
  
        def ff(x):
            return np.sqrt((x**2+p['sigma_psi_mu']**2)*((x**2+p['sigma_psi_mu']**2)/(x**2+2*p['sigma_psi_mu']**2))**2)
        #root= optimize.brentq(f, 0.0001, 10)
        #sigma0=np.sqrt(root-p['sigma_psi_mu']**2)
        #sigma0=np.sqrt(root)
        #p['sigma_psi_init'] =p['sigma_psi_mult']*p['sigma_psi']
  
        
        #Get Variance of Love shock by Duration using Kalman Filter
        self.K,sigma=kalman(1.0,p['sigma_psi']**2,p['sigma_psi_mu']**2,(p['sigma_psi_init_k'])**2,p['dm']+1)
        #K,sigma=kalman(1.0,p['sigma_psi']**2,p['sigma_psi_mu']**2,(sigma0)**2,p['dm'])
        #Get variance by duration
        self.sigmad=-1*np.ones((p['dm']))
        
        p['sigma_psi_init']=np.sqrt(self.K[0]**2*(sigma[0]**2+p['sigma_psi_mu']**2))
        for i in range(p['dm']):
            self.sigmad[i]=np.sqrt(self.K[i+1]**2*(sigma[i+1]**2+p['sigma_psi_mu']**2))
        
        
        self.pars = p
        
        self.dtype = np.float64 # type for all floats
        
       
        
        # relevant for integration
        self.state_names = ['Female, single e','Male, single e','Female, single n','Male, single n',
                            'Couple, M ee','Couple, M en','Couple, M ne','Couple, M nn',
                            'Couple, C ee','Couple, C en','Couple, C ne','Couple, C nn']
        
        #Function that gives potential partners given state names
        #def get_partners(self, female,educ):
        
        #From new to old decription
        self.desc=dict()
        self.desc['Female, single e']='Female, single'
        self.desc['Female, single n']='Female, single'
        self.desc['Male, single e']='Male, single'
        self.desc['Male, single n']='Male, single'
        self.desc['Couple, M ee']='Couple, M'
        self.desc['Couple, M en']='Couple, M'
        self.desc['Couple, M ne']='Couple, M'
        self.desc['Couple, M nn']='Couple, M'
        self.desc['Couple, C ee']='Couple, C'
        self.desc['Couple, C en']='Couple, C'
        self.desc['Couple, C ne']='Couple, C'
        self.desc['Couple, C nn']='Couple, C'
        
        
        #Education from description
        self.edu=dict()
        self.edu['Female, single e']='e'
        self.edu['Female, single n']='n'
        self.edu['Male, single e']='e'
        self.edu['Male, single n']='n'
        self.edu['Couple, M ee']=['e','e']
        self.edu['Couple, M en']=['e','n']
        self.edu['Couple, M ne']=['n','e']
        self.edu['Couple, M nn']=['n','n']
        self.edu['Couple, C ee']=['e','e']
        self.edu['Couple, C en']=['e','n']
        self.edu['Couple, C ne']=['n','e']
        self.edu['Couple, C nn']=['n','n']
            
        
        #Get relationships description from education and gender
        self.desc_i=dict()
        self.desc_i['f']={'e':'Female, single e','n':'Female, single n'}
        self.desc_i['m']={'e':'Male, single e','n':'Male, single n'}
        self.desc_i['e']={'e':{'M':'Couple, M ee','C':'Couple, C ee'},'n':{'M':'Couple, M en','C':'Couple, C en'}}
        self.desc_i['n']={'e':{'M':'Couple, M ne','C':'Couple, C ne'},'n':{'M':'Couple, M nn','C':'Couple, C nn'}}
        
        
        self.ppart,self.ppartc,self.prob,self.probp=dict(),dict(),dict(),dict()
       
        for sex in ['f','m']:
            
            if sex=='f' :
                    
                fr,f1,f2,f22=dict(),dict(),dict(),dict()
                self.ppart[sex],self.ppartc[sex],self.prob[sex],self.probp[sex]=fr,f1,f2,f22
                
                for edu in ['e','n']:
                    
                    if edu=='e':
                    
                        f3,f4=dict(),dict()
                        self.prob[sex][edu],self.probp[sex][edu]=f3,f4
                    
                        #Possible relationships
                        self.ppart[sex][edu]=['Couple, M ee', 'Couple, M en','Couple, C ee', 'Couple, C en']
                        self.ppartc[sex][edu]=['Couple, M ee', 'Couple, M en']
                        
                        #Probabilities of meeting
                        self.prob[sex][edu]['e']=self.pars['Nme']*(1-p['ass'])+p['ass']
                        self.prob[sex][edu]['n']=1-self.prob[sex][edu]['e']
                        self.probp[sex][edu]['e']={'M':'Couple, M ee','C':'Couple, C ee'}
                        self.probp[sex][edu]['n']={'M':'Couple, M en','C':'Couple, C en'}
                    
                    elif edu=='n':
                    
                        
                        f3,f4=dict(),dict()
                        self.prob[sex][edu],self.probp[sex][edu]=f3,f4
                        
                        #Possible relationships
                        self.ppart[sex][edu]=['Couple, M ne', 'Couple, M nn','Couple, C ne', 'Couple, C nn']
                        self.ppartc[sex][edu]=['Couple, M ne', 'Couple, M nn']
                        
                        #Probabilities of meeting
                        self.prob[sex][edu]['e']=self.pars['Nme']*(1-p['ass'])
                        self.prob[sex][edu]['n']=1-self.prob[sex][edu]['e']
                        self.probp[sex][edu]['e']={'M':'Couple, M ne','C':'Couple, C ne'}
                        self.probp[sex][edu]['n']={'M':'Couple, M nn','C':'Couple, C nn'}
                        
            elif sex=='m':
                    
                    
                fr,f1,f2,f22=dict(),dict(),dict(),dict()
                self.ppart[sex],self.ppartc[sex],self.prob[sex],self.probp[sex]=fr,f1,f2,f22
                
                for edu in ['e','n']:
                    
                    if edu=='e':
                        
                        f3,f4=dict(),dict()
                        self.prob[sex][edu],self.probp[sex][edu]=f3,f4
                        
                        #Possible relationships
                        self.ppart[sex][edu]=['Couple, M ee', 'Couple, M ne','Couple, C ee', 'Couple, C ne']
                        self.ppartc[sex][edu]=['Couple, M ee', 'Couple, M ne']
                        
                        #Probabilities of meeting
                        self.prob[sex][edu]['e']=self.pars['Nfe']*(1-p['ass'])+p['ass']
                        self.prob[sex][edu]['n']=1-self.prob[sex][edu]['e']
                        self.probp[sex][edu]['e']={'M':'Couple, M ee','C':'Couple, C ee'}
                        self.probp[sex][edu]['n']={'M':'Couple, M ne','C':'Couple, C ne'}
                    
                    elif edu=='n':
                    
                        f3,f4=dict(),dict()
                        self.prob[sex][edu],self.probp[sex][edu]=f3,f4
                        
                        #Possible relationships
                        self.ppart[sex][edu]=['Couple, M en', 'Couple, M nn','Couple, C en', 'Couple, C nn']
                        self.ppartc[sex][edu]=['Couple, M en', 'Couple, M nn']
                        
                        
                        #Probabilities of meeting
                        self.prob[sex][edu]['e']=self.pars['Nfe']*(1-p['ass'])
                        self.prob[sex][edu]['n']=1-self.prob[sex][edu]['e']
                        self.probp[sex][edu]['e']={'M':'Couple, M en','C':'Couple, C en'}
                        self.probp[sex][edu]['n']={'M':'Couple, M nn','C':'Couple, C nn'}
                    
                    
                
        
        
        # female labor supply
        #self.ls_levels = np.array([0.0,.357],dtype=self.dtype)
        #self.mlevel=.357
        self.ls_levels = np.array([0.0,0.8038],dtype=self.dtype)
        self.mlevel=0.8038
        #self.ls_utilities = np.array([p['uls'],0.0],dtype=self.dtype)
        self.ls_pdown = np.array([p['pls'],0.0],dtype=self.dtype)
        self.nls = len(self.ls_levels)
        
        
        
        
       
        #Cost of Divorce
        if divorce_costs == 'Default':
            # by default the costs are set in the bottom
            self.div_costs = DivorceCosts(eq_split=1.0,assets_kept=1.0)
        else:
            if isinstance(divorce_costs,dict):
                # you can feed in arguments to DivorceCosts
                self.div_costs = DivorceCosts(**divorce_costs)
            else:
                # or just the output of DivorceCosts
                assert isinstance(divorce_costs,DivorceCosts)
                self.div_costs = divorce_costs
                
        #Cost of Separation
        if separation_costs == 'Default':
            # by default the costs are set in the bottom
            self.sep_costs = DivorceCosts(eq_split=0.0,assets_kept=1.0)
        else:
            if isinstance(separation_costs,dict):
                # you can feed in arguments to DivorceCosts
                self.sep_costs = DivorceCosts(**separation_costs)
            else:
                # or just the output of DivorceCosts
                assert isinstance(separation_costs,DivorceCosts)
                self.sep_costs = separation_costs
            
        # exogrid should be deprecated
        if not nogrid:
        
            exogrid = dict()
            
            
            # let's approximate three Markov chains
            # this sets up exogenous grid
            
            # FIXME: this uses number of points from 0th entry. 
            # in principle we can generalize this
            

            exogrid['zf_t'],  exogrid['zf_t_mat'],zft,zftmat,exogrid['zm_t'],  exogrid['zm_t_mat']=dict(),dict(),dict(),dict(),dict(),dict()
            zft['e'],       zftmat['e']                     = rouw_nonst(p['T'],p['sig_zf']['e']*period_year**0.5,p['sig_zf_0']['e'],p['n_zf_t'][0]-p['n_zf_correct'])
            zft['n'],               zftmat['n']             = rouw_nonst(p['T'],p['sig_zf']['n']*period_year**0.5,p['sig_zf_0']['n'],p['n_zf_t'][0]-p['n_zf_correct'])
            exogrid['zm_t']['e'],  exogrid['zm_t_mat']['e'] = rouw_nonst(p['T'],p['sig_zm']['e']*period_year**0.5,p['sig_zm_0']['e'],p['n_zm_t'][0])
            exogrid['zm_t']['n'],  exogrid['zm_t_mat']['n'] = rouw_nonst(p['T'],p['sig_zm']['n']*period_year**0.5,p['sig_zm_0']['n'],p['n_zm_t'][0])
            
            
            
            
            #Embody the grid for women in a bigger one
            if p['n_zf_correct']>0:
                for edu in ['e','n']:
                    exogrid['zf_t'][edu]=list()
                    exogrid['zf_t_mat'][edu]=list()
                    for t in range(p['T']):
                        
                        
                        #Extend grid
                        h=zft[edu][t][1]-zft[edu][t][0]
                        # dist1=zft[edu][t][0]-h
                        # dist0=zft[edu][t][0]-p['n_zf_correct']*h
                        dist2=zft[edu][t][0]-h
                        dist1=zft[edu][t][0]-(p['n_zf_correct']-1)*h
                        dist0=zft[edu][t][0]-p['n_zf_correct']*h
                        
                        #Copy transition matrix
                        exogrid['zf_t'][edu]=exogrid['zf_t'][edu]+[np.concatenate((np.array([dist0,dist1,dist2]),zft[edu][t]))]
                        #exogrid['zf_t'][edu]=exogrid['zf_t'][edu]+[np.concatenate((np.array([dist0,dist1]),zft[edu][t]))]
                        #exogrid['zf_t'][edu]=exogrid['zf_t'][edu]+[np.concatenate((np.array([dist1]),zft[edu][t]))]
                        exogrid['zf_t_mat'][edu]=exogrid['zf_t_mat'][edu]+[np.zeros((p['n_zf_t'][t],p['n_zf_t'][t]))]
                        exogrid['zf_t_mat'][edu][t][p['n_zf_correct']:,p['n_zf_correct']:]=zftmat[edu][t]
                        
                        #Shift transition matrix to fill values
                        if t<p['T']-1:
                            
                            exogrid['zf_t_mat'][edu][t][0,:-p['n_zf_correct']]=zftmat[edu][t][0,:]
                            exogrid['zf_t_mat'][edu][t][1,:-p['n_zf_correct']]=zftmat[edu][t][1,:]
                            exogrid['zf_t_mat'][edu][t][2,:-p['n_zf_correct']]=zftmat[edu][t][2,:]
                           
                                
                        else:
                            exogrid['zf_t_mat'][edu][t]=None
                           
                    
            else:    

                exogrid['zf_t']=zft
                exogrid['zf_t_mat']=zftmat

                    
                    
            #Drift the grids
            for e in ['e','n']:
                
                for t in range(Tret):
                    exogrid['zf_t'][e][t]=exogrid['zf_t'][e][t]-p['correction']*t
                for t in range(Tret-2):
                    exogrid['zm_t'][e][t]=exogrid['zm_t'][e][t]-p['correction']*t
                    
                    
                    

         
            ################################
            #First mimic US pension system
            ###############################
                      
            #function to compute pension
            def pens(value):
                
                #Get median income before retirement using men model income in Tret-1
                #+ ratio of men and female labor income for the rest
                yret=32.30#(1.73377+(.8427056/1.224638)*1.73377* 0.3246206)/(1+0.3246206)
                thresh1=0.38*yret
                thresh2=1.59*yret
                
                inc1=np.minimum(value,thresh1)
                inc2=np.maximum(np.minimum(value-inc1,thresh2-inc1),0)
                inc3=np.maximum(value-thresh2,0)
                
                return inc1*0.9+inc2*0.32+inc3*0.15            
            
            
            for e in ['e','n']:
                for t in range(Tret,T):
                #    exogrid['zf_t'][e][t] = np.array([np.log(p['wret'])])
                 #   exogrid['zm_t'][e][t] = np.array([np.log(p['wret'])])
                    exogrid['zf_t_mat'][e][t] = np.atleast_2d(1.0)
                 #for t in range(Tret-2,T):                      
                
                
            #     # fix transition from non-retired to retired    
            #     exogrid['zf_t_mat'][Tret-1] = np.ones((p['n_zf_t'][Tret-1],1))
            #     exogrid['zm_t_mat'][Tret-1] = np.ones((p['n_zm_t'][Tret-1],1))
            
                #Tax system as in Wu and Kruger
                # for t in range(0,Tret):
                #     exogrid['zf_t'][e][t] = exogrid['zf_t'][e][t]#*(1-0.1327)+np.log(1-0.1575)
                #     exogrid['zm_t'][e][t] = exogrid['zm_t'][e][t]#*(1-0.1327)+np.log(1-0.1575)  
                
                #Comment out the following if you dont want retirment based on income
                for t in range(Tret,T):
                   
                    exogrid['zf_t'][e][t] = np.log(pens(np.exp(p['wtrend']['f'][e][Tret-1]+exogrid['zf_t'][e][Tret-1])))#np.array([np.log(p['wret'])])                                
                    exogrid['zf_t_mat'][e][t] = np.diag(np.ones(len(exogrid['zf_t'][e][t])))#p.atleast_2d(1.0)
                    
                    
                for t in range(Tret-2,T):
                    exogrid['zm_t'][e][t] = np.log(pens(np.exp(p['wtrend']['m'][e][Tret-3]+exogrid['zm_t'][e][Tret-3]))) 
                    exogrid['zm_t_mat'][e][t] = np.diag(np.ones(len(exogrid['zm_t'][e][t])))
                    
                # fix transition from non-retired to retired    
                exogrid['zf_t_mat'][e][Tret-1] = np.diag(np.ones(len(exogrid['zf_t'][e][Tret-1])))
                exogrid['zm_t_mat'][e][Tret-3] = np.diag(np.ones(len(exogrid['zm_t'][e][Tret-3])))


               
            ###########################
            #Love shock grid
            ###########################
            
            #Idea: first build grid with variance in dmax, then dmax-1 and so on are 
            #created using the Fella routine backwards
            
            print('variances are {}, {}, {}, {}, {}'.format(self.pars['sigma_psi_init'],self.sigmad[0],self.sigmad[1],self.sigmad[2],self.sigmad[3]))
            print(self.K[0],self.K[1],self.K[2],self.K[3])
            #New way of getting transition matrix
            psit, matri=list(np.ones((T))),list(np.ones((T)))
            
            sigmainitial=np.sqrt((self.pars['sigma_psi_init'])**2)+(np.sum(self.sigmad**2)-len(self.sigmad)*self.sigmad[-1]**2)
            sigmainitial=self.pars['sigma_psi_init']
            
            sigmabase=np.sqrt([sigmainitial**2+(t)*self.sigmad[-1]**2 for t in range(T+p['dm']+1)])
            sigmadp=np.concatenate((np.array([0.0]),self.sigmad))
            sigmadi=self.sigmad[::-1]
            for i in range(T):
                
                #base=max(sigmabase[i+p['dm']]**2-np.sum(self.sigmad**2)+0.01,0.001)
                base=max(sigmabase[i+p['dm']]**2-np.sum(self.sigmad**2)+0.000000001,(self.pars['sigma_psi_init']/2.5)**2)
                sigp=np.sqrt([base+np.cumsum(sigmadp**2)[dd] for dd in range(p['dm']+1)])
                #sigp=np.sqrt([base+np.sum(sigmadi[p['dm']-dd:]**2) for dd in range(p['dm']+1)])
                psit[i],matri[i] = tauchen_nonstm(p['dm']+1,0.0,0.0,p['n_psi_t'][0],sd_z=sigp)
                

            exogrid['psi_t'], exogrid['psi_t_mat']=list(np.ones((p['dm']))),list(np.ones((p['dm'])))
            for dd in range(p['dm']):
                
                
                #exogrid['psi_t'][dd], exogrid['psi_t_mat'][dd] = tauchen_nonst(p['T'],self.sigmad[dd],self.sigmad[dd],p['n_psi_t'][0])
                exogrid['psi_t'][dd], exogrid['psi_t_mat'][dd] = tauchen_nonst(p['T'],self.sigmad[dd],sigmabase[0:p['T']],p['n_psi_t'][0])
                for i in range(T):
                    
                    if i<Tret:
                        exogrid['psi_t'][dd][i], exogrid['psi_t_mat'][dd][i]=psit[max(i-dd,0)][dd],matri[min(i-dd,T-1)][dd]

          
            #Here I impose no change in psi from retirement till the end of time 
            for t in range(Tren,T-1):
                for dd in range(p['dm']):
               
                    exogrid['psi_t'][dd][t] = exogrid['psi_t'][dd][Tren-1]#np.array([np.log(p['wret'])])             
                    exogrid['psi_t_mat'][dd][t] = np.diag(np.ones(len(exogrid['psi_t'][dd][t])))

            
            #Get original grid
            self.orig_psi=exogrid['psi_t'][0]
            # #Modify the mean
            # for t in range(Tret):
            #     for d in range(p['dm']):
                    
            #         exogrid['psi_t'][dd][t]=exogrid['psi_t'][dd][t].copy()*p['multpsi']
           #Now the crazy matrix for "true process"
            exogrid['noise_psi_mat'],exogrid['true_psi_mat']=exogrid['psi_t_mat'],exogrid['psi_t_mat']
#            
#            from mc_tools import mc_simulate
#            zero=np.ones((100000),dtype=np.int32)*5
#            s1=mc_simulate(zero,exogrid['psi_t_mat'][0][0])
#            s1e=exogrid['psi_t'][0][1][s1]
#            s2=mc_simulate(s1,exogrid['psi_t_mat'][0][1])
#            s2e=exogrid['psi_t'][1][2][s2]
#            diffe=s2e-s1e
            
            exogrid['all_t_mat_by_l'], exogrid['all_t_mat_by_l_spt'],exogrid['all_t']=dict(),dict(),dict()
            for e in ['e','n']:
                exogrid['all_t_mat_by_l'][e], exogrid['all_t_mat_by_l_spt'][e],exogrid['all_t'][e]=dict(),dict(),dict()
                for eo in ['e','n']:
                    exogrid['all_t_mat_by_l'][e][eo],  exogrid['all_t_mat_by_l_spt'][e][eo],exogrid['all_t'][e][eo]=list(np.ones((p['dm']))),list(np.ones((p['dm']))),list(np.ones((p['dm'])))
            
            
            exogrid['zf_t_mat2']=deepcopy(exogrid['zf_t_mat'])
            exogrid['zf_t_mat2d']=deepcopy(exogrid['zf_t_mat'])
            for e in ['e','n']:
                
                #FBad shock women
                zf_bad = [tauchen_drift(exogrid['zf_t'][e][t].copy(), exogrid['zf_t'][e][t+1].copy(), 
                                                1.0, p['sig_zf'][e], p['z_drift'], exogrid['zf_t_mat'][e][t])
                                    for t in range(self.pars['Tret']-1) ]
                        
                #Account for retirement here
                zf_bad = zf_bad.copy()+[exogrid['zf_t_mat'][e][t] for t in range(self.pars['Tret']-1,self.pars['T']-1)]+ [None]
                
                for eo in ['e','n']:
                    
                    #Preliminaries
                    zfzm, zfzmmat = combine_matrices_two_lists(exogrid['zf_t'][e].copy(), exogrid['zm_t'][eo].copy(), exogrid['zf_t_mat'][e].copy(), exogrid['zm_t_mat'][eo].copy())
                    
                    zf_t_mat_down = zf_bad.copy()
                    zfzm2, zfzmmat2 = combine_matrices_two_lists(exogrid['zf_t'][e].copy(), exogrid['zm_t'][eo].copy(), zf_t_mat_down.copy(), exogrid['zm_t_mat'][eo].copy())
                   
                    
                    # if p['rho_s']>0:
                    #     for t in range(p['Tret']-1):
                    #         for j in range(p['n_zm_t'][t]):
                    #             for ym in range(p['n_zm_t'][t]):
                                
                                    
                    #                 rhom=(1.0-p['rho_s']**2)**0.5
                    #                 prec=exogrid['zm_t'][eo][t][j] if t>0 else 0.0
                    #                 drif=p['rho_s']*p['sig_zf'][e]/p['sig_zm'][eo]*(exogrid['zm_t'][eo][t+1][ym]-prec)
                    #                 mat1=tauchen_drift(exogrid['zf_t'][e][t].copy(), exogrid['zf_t'][e][t+1].copy(), 1.0, rhom*p['sig_zf'][e], drif, exogrid['zf_t_mat'][e][t])
                    #                 mat2=tauchen_drift(exogrid['zf_t'][e][t].copy(), exogrid['zf_t'][e][t+1].copy(), 1.0, rhom*p['sig_zf'][e], drif+p['z_drift'], exogrid['zf_t_mat'][e][t])
                    #                 for i in range(p['n_zf_t'][t]): 
                                
                    #                     #Modify the grid for women
                    #                     exogrid['zf_t_mat2'][e][t][i,:]= mat1[i,:]
    
                    #                     exogrid['zf_t_mat2d'][e][t][i,:]=mat2[i,:]
                                        
                    #                     ##Update the big Matrix
                    #                     for yf in range(p['n_zf_t'][t]):
                                        
                                            
                    #                         zfzmmat[t][i*(p['n_zm_t'][t]-1)+j+i,yf*(p['n_zm_t'][t]-1)+ym+yf]=exogrid['zf_t_mat2'][e][t][i,yf]*exogrid['zm_t_mat'][eo][t][j,ym]
                    #                         zfzmmat2[t][i*(p['n_zm_t'][t]-1)+j+i,yf*(p['n_zm_t'][t]-1)+ym+yf]=exogrid['zf_t_mat2d'][e][t][i,yf]*exogrid['zm_t_mat'][eo][t][j,ym]
                   
                    
                    #Adjust retirement as in Heatcote et al.
                    for t in range(p['Tret'],p['T']):
                        for j in range(len(zfzm[t])):
                            pref=max(np.exp(zfzm[t][j][0])+np.exp(zfzm[t][j][1]),1.5*max(np.exp(zfzm[t][j][0]),np.exp(zfzm[t][j][1])))
                            zfzm[t][j][0]=np.log(pref)
                            zfzm[t][j][1]=-20.0
                            pref=max(np.exp(zfzm2[t][j][0])+np.exp(zfzm2[t][j][1]),1.5*max(np.exp(zfzm2[t][j][0]),np.exp(zfzm2[t][j][1])))
                            zfzm2[t][j][0]=np.log(pref)
                            zfzm2[t][j][1]=-20.0
                   
                   #Modify
                    for dd in range(p['dm']):
                     
                      
                        all_t, all_t_mat = combine_matrices_two_listsf(zfzm.copy(),exogrid['psi_t'][dd].copy(),zfzmmat.copy(),exogrid['psi_t_mat'][dd].copy(),check=False,trim=True)                       
                      
                        all_t_mat_sparse_T = [sparse.csc_matrix(D.T) if D is not None else None for D in all_t_mat.copy()]
                                 
                        all_t_down, all_t_mat_down = combine_matrices_two_listsf(zfzm2.copy(),exogrid['psi_t'][dd].copy(),zfzmmat2.copy(),exogrid['psi_t_mat'][dd].copy(),check=False,trim=True)                        
                        all_t_mat_down_sparse_T = [sparse.csc_matrix(D.T) if D is not None else None for D in all_t_mat_down.copy()]
                        
                        
                        all_t_mat_by_l_spt =[all_t_mat_down_sparse_T.copy(),all_t_mat_sparse_T.copy()] 
                       
                                              
                        exogrid['all_t_mat_by_l_spt'][e][eo][dd] = all_t_mat_by_l_spt.copy()                        
                        exogrid['all_t'][e][eo][dd] = all_t.copy()
                        
                        del all_t_mat_by_l_spt,all_t_mat_down_sparse_T,all_t_down,all_t_mat_down,all_t_mat,all_t_mat_sparse_T
                        
                        
                       
                       
                    del zfzm,zfzmmat,zfzm2,zfzmmat2    
                    gc.collect()

            
            Exogrid_nt = namedtuple('Exogrid_nt',exogrid.keys())
            
            self.exogrid = Exogrid_nt(**exogrid)
            self.pars['nexo_t'] = [v.shape[0] for v in all_t.copy()]
            del all_t
            
            
        
                #Get distribution of productivties by age:
        N=10000
        shokko=np.random.random_sample((2,N,T))
        self.exo=dict()
        self.exo['e']=np.zeros((2,N,T+1),dtype=np.int16)
        self.exo['n']=np.zeros((2,N,T+1),dtype=np.int16)
        
        self.exo['e'][0,:,0]=np.array(np.ones(N)*(self.pars['n_zm_t'][0]-1)/2,dtype=np.int16)
        self.exo['e'][1,:,0]=np.array(np.ones(N)*(self.pars['n_zf_t'][0]-self.pars['n_zf_correct']-1)/2+self.pars['n_zf_correct'],dtype=np.int16)
        self.exo['n'][0,:,0]=np.array(np.ones(N)*(self.pars['n_zm_t'][0]-1)/2,dtype=np.int16)
        self.exo['n'][1,:,0]=np.array(np.ones(N)*(self.pars['n_zf_t'][0]-self.pars['n_zf_correct']-1)/2+self.pars['n_zf_correct'],dtype=np.int16)
        
        #Initial State
        shocks_init = shokko[1,:,0]
        self.exo['e'][0,:,0]=np.sum((shocks_init[:,None] > np.cumsum(int_prob(self.exogrid.zm_t['e'][0],sig=self.pars['sig_zm_0']['e']))[None,:]), axis=1) 
        self.exo['n'][0,:,0]=np.sum((shocks_init[:,None] > np.cumsum(int_prob(self.exogrid.zm_t['n'][0],sig=self.pars['sig_zm_0']['n']))[None,:]), axis=1) 
        self.exo['e'][1,:,0]=np.sum((shocks_init[:,None] > np.cumsum(int_prob(self.exogrid.zf_t['e'][0],sig=self.pars['sig_zf_0']['e']))[None,:]), axis=1) 
        self.exo['n'][1,:,0]=np.sum((shocks_init[:,None] > np.cumsum(int_prob(self.exogrid.zf_t['n'][0],sig=self.pars['sig_zf_0']['n']))[None,:]), axis=1) 
        
       
        for e in ['e','n']:
            for t in range(self.pars['T']-1):
                
                self.exo[e][0,:,t+1]=mc_simulate(self.exo[e][0,:,t],self.exogrid.zm_t_mat[e][t],shocks=shokko[0,:,t])
                self.exo[e][1,:,t+1]=mc_simulate(self.exo[e][1,:,t],self.exogrid.zf_t_mat[e][t],shocks=shokko[1,:,t])
           
        #Get theh distribution
        self.probb=dict()
        self.probb['f'],self.probb['m']=dict(),dict()
        self.probb['f']['e'],self.probb['m']['e'],self.probb['f']['n'],self.probb['m']['n']=list(),list(),list(),list()
        
        zera=np.zeros(self.pars['n_zf_correct'])
        for e in ['e','n']:
            for t in range(self.pars['T']-1):
                self.probb['m'][e]=self.probb['m'][e]+[np.unique(self.exo[e][0,:,t], return_counts=True)[1]/N]
                self.probb['f'][e]=self.probb['f'][e]+[np.unique(self.exo[e][1,:,t], return_counts=True)[1]/N]
            
            
        
        #Grid Couple
        self.na = 40
        self.amin = 0
        self.amax =80#60
        self.amax1 = 180#60
        self.agrid_c = np.linspace(self.amin,self.amax,self.na,dtype=self.dtype)
        tune=2.5
        self.agrid_c = np.geomspace(self.amin+tune,self.amax+tune,num=self.na)-tune
        self.agrid_c[-1]=self.amax1
        self.agrid_c[-2]=120
        # this builds finer grid for potential savings
        s_between = 7 # default numer of points between poitns on agrid
        s_da_min = 0.1 # minimal step (does not create more points)
        s_da_max = 1.0 # maximal step (creates more if not enough)
        
        self.sgrid_c = build_s_grid(self.agrid_c,s_between,s_da_min,s_da_max)
        self.vsgrid_c = VecOnGrid(self.agrid_c,self.sgrid_c)
        
        
         
        #Grid Single
        scale = 1.1
        self.amin_s = 0
        self.amax_s = self.amax/scale
        self.agrid_s = np.linspace(self.amin_s,self.amax_s,self.na,dtype=self.dtype)
        #self.agrid_s[self.na-1]=18#180
        tune_s=2.5
        self.agrid_s = np.geomspace(self.amin_s+tune_s,self.amax_s+tune_s,num=self.na)-tune_s
        self.agrid_s[-1]=self.amax1/scale
        self.agrid_c[-2]=120/scale
        self.sgrid_s = build_s_grid(self.agrid_s,s_between,s_da_min,s_da_max)
        self.vsgrid_s = VecOnGrid(self.agrid_s,self.sgrid_s)
        
#        #No assets stuff
        self.na=1
        self.agrid_s=np.array([0.0])
        self.sgrid_s=np.array([0.0])
        self.vsgrid_s =np.array([0.0])
        self.agrid_c=np.array([0.0])
        self.sgrid_c=np.array([0.0])
        self.vsgrid_c =np.array([0.0])
        self.amin = 0
        self.amax = 0
        self.amax1 = 0
#        
        
        # grid for theta
        self.ntheta = 13
        self.thetamin = 0.02
        self.thetamax = 0.98
        self.thetagrid = np.linspace(self.thetamin,self.thetamax,self.ntheta,dtype=self.dtype)

        
        # construct finer grid for bargaining
        ntheta_fine =9*self.ntheta #7*self.ntheta actual number may be a bit bigger
        self.thetagrid_fine = np.unique(np.concatenate( (self.thetagrid,np.linspace(self.thetamin,self.thetamax,ntheta_fine,dtype=self.dtype)) ))
        self.ntheta_fine = self.thetagrid_fine.size
        
        i_orig = list()
        
        for theta in self.thetagrid:
            assert theta in self.thetagrid_fine
            i_orig.append(np.where(self.thetagrid_fine==theta)[0])
            
        assert len(i_orig) == self.thetagrid.size
        # allows to recover original gird points on the fine grid        
        self.theta_orig_on_fine = np.array(i_orig).flatten()
        self.v_thetagrid_fine = VecOnGrid(self.thetagrid,self.thetagrid_fine)
        # precomputed object for interpolation
        
        #Get indexes from fine back to coarse thetagrid
        cg=VecOnGrid(self.thetagrid,self.thetagrid_fine)
        index_t=cg.i
        index_t1=index_t+1
        wherep=(cg.wthis<0.5)
        self.igridcoarse=index_t
        self.igridcoarse[wherep]=index_t1[wherep]

            
        
        self.exo_grids = {'Female, single e':exogrid['zf_t']['e'],
                          'Male, single e':exogrid['zm_t']['e'],
                          'Female, single n':exogrid['zf_t']['n'],
                          'Male, single n':exogrid['zm_t']['n'],
                          'Couple, M ee':exogrid['all_t']['e']['e'],
                          'Couple, C ee':exogrid['all_t']['e']['e'],
                          'Couple, M en':exogrid['all_t']['e']['n'],
                          'Couple, C en':exogrid['all_t']['e']['n'],
                          'Couple, M ne':exogrid['all_t']['n']['e'],
                          'Couple, C ne':exogrid['all_t']['n']['e'],
                          'Couple, M nn':exogrid['all_t']['n']['n'],
                          'Couple, C nn':exogrid['all_t']['n']['n']}
        
        self.exo_mats = {'Female, single e':exogrid['zf_t_mat']['e'],
                  'Male, single e':exogrid['zm_t_mat']['e'],
                  'Female, single n':exogrid['zf_t_mat']['n'],
                  'Male, single n':exogrid['zm_t_mat']['n'],
                  'Couple, M ee':exogrid['all_t_mat_by_l_spt']['e']['e'],
                  'Couple, C ee':exogrid['all_t_mat_by_l_spt']['e']['e'],
                  'Couple, M en':exogrid['all_t_mat_by_l_spt']['e']['n'],
                  'Couple, C en':exogrid['all_t_mat_by_l_spt']['e']['n'],
                  'Couple, M ne':exogrid['all_t_mat_by_l_spt']['n']['e'],
                  'Couple, C ne':exogrid['all_t_mat_by_l_spt']['n']['e'],
                  'Couple, M nn':exogrid['all_t_mat_by_l_spt']['n']['n'],
                  'Couple, C nn':exogrid['all_t_mat_by_l_spt']['n']['n']}
                          
 
        self.utility_shifters = {'Female, single e':0.0,
          'Male, single e':0.0,
          'Female, single n':0.0,
          'Male, single n':0.0,
          'Couple, M ee':p['u_shift_mar'],
          'Couple, C ee':p['u_shift_coh'],
          'Couple, M en':p['u_shift_mar'],
          'Couple, C en':p['u_shift_coh'],
          'Couple, M ne':p['u_shift_mar'],
          'Couple, C ne':p['u_shift_coh'],
          'Couple, M nn':p['u_shift_mar'],
          'Couple, C nn':p['u_shift_coh']}

        
        # this pre-computes transition matrices for meeting a partner
        zf_t_partmat,zm_t_partmat=dict(),dict()

        for e in ['e','n']:
            zf_t_partmat[e],zm_t_partmat[e]=dict(),dict()
            for eo in ['e','n']:
                zf_t_partmat[e][eo] = [self.mar_mats_iexo(t,e,eo,female=True) if t < p['T'] - 1 else None 
                        for t in range(p['T'])]
                zm_t_partmat[e][eo] = [self.mar_mats_iexo(t,e,eo,female=False) if t < p['T'] - 1 else None 
                        for t in range(p['T'])]
        
        self.part_mats = {'Female, single':zf_t_partmat,
                          'Male, single':  zm_t_partmat,
                          'Couple, M': None,
                          'Couple, C': None} # last is added for consistency
        

        
        self.mar_mats_assets()
        
        self.mar_mats_combine()
        
        
        # building m grid
        ezfmin = min([np.min(self.ls_levels[-1]*np.exp(g+t)) for g,t in zip(exogrid['zf_t']['n'],p['wtrend']['f']['n'])])
        ezmmin = min([np.min(self.mlevel*np.exp(g+t)) for g,t in zip(exogrid['zm_t']['n'],p['wtrend']['m']['n'])])
        ezfmax = max([np.max(self.ls_levels[-1]*np.exp(g+t)) for g,t in zip(exogrid['zf_t']['e'],p['wtrend']['f']['e'])])
        ezmmax = max([np.max(self.mlevel*np.exp(g+t)) for g,t in zip(exogrid['zm_t']['e'],p['wtrend']['m']['e'])])
#        
#        ezfmin = min([np.min(self.ls_levels[-1]*np.exp(g+t)) for g,t in zip(exogrid['zf_t']['e'],p['wtrend']['f']['e'])])
#        ezmmin = min([np.min(self.mlevel*np.exp(g+t)) for g,t in zip(exogrid['zm_t']['e'],p['wtrend']['m']['e'])])
#        ezfmax = max([np.max(self.ls_levels[-1]*np.exp(g+t)) for g,t in zip(exogrid['zf_t']['n'],p['wtrend']['f']['n'])])
#        ezmmax = max([np.max(self.mlevel*np.exp(g+t)) for g,t in zip(exogrid['zm_t']['n'],p['wtrend']['m']['n'])])
#        
        
        
        
        
        self.money_min = 0.95*min(ezmmin,ezfmin) # cause FLS can be up to 0
        #self.mgrid = ezmmin + self.sgrid_c # this can be changed later
        mmin = self.money_min
        mmax = ezfmax + ezmmax + np.max(self.pars['R_t'])*self.amax1
        mint = (ezfmax + ezmmax) # poin where more dense grid begins
        
        ndense = 6000
        nm = 15000
        
        gsparse = np.linspace(mint,mmax,nm-ndense)
        gdense = np.linspace(mmin,mint,ndense+1) # +1 as there is a common pt
        
        self.mgrid = np.zeros(nm,dtype=self.dtype)
        self.mgrid[ndense:] = gsparse
        self.mgrid[:(ndense+1)] = gdense
        assert np.all(np.diff(self.mgrid)>=0)
        
        self.u_precompute()
        
        
                                  
        
        
    def mar_mats_assets(self,npoints=1,abar=0.0001):
        # for each grid point on single's grid it returns npoints positions
        # on (potential) couple's grid's and assets of potential partner 
        # (that can be off grid) and correpsonding probabilities. 
        
        self.prob_a_mat = dict()
        self.i_a_mat = dict()
        
        na = self.agrid_s.size
        
        agrid_s = self.agrid_s
        agrid_c = self.agrid_c
        
        s_a_partner = self.pars['sig_partner_a']
        
        for female in [True,False]:
            prob_a_mat = np.zeros((na,npoints),dtype=self.dtype)
            i_a_mat = np.zeros((na,npoints),dtype=np.int16)
            
            
            
            for ia, a in enumerate(agrid_s):
                lagrid_t = np.zeros_like(agrid_c)
                
                i_neg = (agrid_c <= max(abar,a) - 1e-16) if na>1 else np.array([True],dtype=bool) 
                
                # if a is zero this works a bit weird but does the job
                
                lagrid_t[~i_neg] = np.log(2e-16 + (agrid_c[~i_neg] - a)/max(abar,a))
                lmin = lagrid_t[~i_neg].min()  if na>1 else lagrid_t
                # just fill with very negative values so this is never chosen
                lagrid_t[i_neg] = lmin - s_a_partner*10 - \
                    s_a_partner*np.flip(np.arange(i_neg.sum())) 
                
                # TODO: this needs to be checked
                if female:
                    mean=self.pars['mean_partner_a_female']
                else:
                    mean=self.pars['mean_partner_a_male']
                p_a = int_prob(lagrid_t,mu=mean,sig=s_a_partner,n_points=npoints) if na>1 else np.array([1.0])
                i_pa = (-p_a).argsort()[:npoints] # this is more robust then nonzero
                p_pa = p_a[i_pa]
                prob_a_mat[ia,:] = p_pa
                i_a_mat[ia,:] = i_pa
            
            
            self.prob_a_mat[female] = prob_a_mat
            self.i_a_mat[female] = i_a_mat
            

        
    
    def mar_mats_iexo(self,t,e,eo,female=True,trim_lvl=0.00):
        # TODO: check timing
        # this returns transition matrix for single agents into possible couples
        # rows are single's states
        # columnts are couple's states
        # you have to transpose it if you want to use it for integration
        setup = self
        
        nexo = setup.pars['nexo_t'][t]
        sigma_psi_init = setup.pars['sigma_psi_init']
        #sig_z_partner = setup.pars['sig_partner_z']
        psi_couple = setup.orig_psi[t]#setup.exogrid.psi_t[0][t+1]
        
        g='f' if female else 'm'
        go='m' if female else 'f'
        
        if female:
            nz_single = setup.exogrid.zf_t[e][t].shape[0]
            p_mat = np.empty((nexo,nz_single))
            z_own = setup.exogrid.zf_t[e][t]
            n_zown = z_own.shape[0]
            z_partner = setup.exogrid.zm_t[eo][t]
            zmat_own = setup.exogrid.zf_t_mat[e][t]
            trend=setup.pars['wtrendp'][go][eo][t]
            mean=setup.pars['mean_partner_z_female']-setup.pars['wtrend'][go][eo][t]+setup.pars['wtrendp'][go][eo][t]
            sig_z_partner=(setup.pars['sig_zm_0'][eo]**2+(t+1)*setup.pars['sig_zm'][eo]**2)**0.5
        else:
            nz_single = setup.exogrid.zm_t[e][t].shape[0]
            p_mat = np.empty((nexo,nz_single))
            z_own = setup.exogrid.zm_t[e][t]
            n_zown = z_own.shape[0]
            z_partner = setup.exogrid.zf_t[eo][t]
            zmat_own = setup.exogrid.zm_t_mat[e][t]    
            trend=setup.pars['wtrendp'][go][eo][t]
            mean=setup.pars['mean_partner_z_male']-setup.pars['wtrend'][go][eo][t]+setup.pars['wtrendp'][go][eo][t]
            sig_z_partner=(setup.pars['sig_zf_0'][eo]**2+(t+1)*setup.pars['sig_zf'][eo]**2)**0.5
            
        def ind_conv(a,b,c): return setup.all_indices(t,(a,b,c))[0]
        
        
        for iz in range(n_zown):
            p_psi = int_prob(psi_couple,mu=0.0,sig=sigma_psi_init)
            if female:
              #  p_zm  = (1-setup.pars['dump_factor_z'])*self.probb['m'][eo][t]+setup.pars['dump_factor_z']*int_prob(z_partner, mu=z_own[iz]+mean,sig=0.0001)
                      
                
                p_zm  =int_prob(z_partner, mu=-t*setup.pars['correction']+setup.pars['dump_factor_z']*(z_own[iz]+t*setup.pars['correction'])+
                                  mean,sig=0.0001+(1-setup.pars['dump_factor_z'])**0.5*(sig_z_partner)*setup.pars['sig_partner_mult'])
                                            
                p_zf  = zmat_own[iz,:]
            else:
               # p_zf  = (1-setup.pars['dump_factor_z'])*self.probb['f'][eo][t]+setup.pars['dump_factor_z']*int_prob(z_partner, mu=z_own[iz]+mean,sig=0.0001)
                     

                p_zf  =int_prob(z_partner, mu=-t*setup.pars['correction']+setup.pars['dump_factor_z']*(z_own[iz]+t*setup.pars['correction'])+
                                  mean,sig=0.0001+(1-setup.pars['dump_factor_z'])**0.5*(sig_z_partner)*setup.pars['sig_partner_mult'])
                       
                       
                p_zm  = zmat_own[iz,:]
            #sm = sf
        
            p_vec = np.zeros(nexo)
            
            for izf, p_zf_i in enumerate(p_zf):
                if p_zf_i < trim_lvl: continue
            
                for izm, p_zm_i in enumerate(p_zm):
                    if p_zf_i*p_zm_i < trim_lvl: continue
                
                    for ipsi, p_psi_i in enumerate(p_psi):                    
                        p = p_zf_i*p_zm_i*p_psi_i
                        
                        if p > trim_lvl:
                            p_vec[ind_conv(izf,izm,ipsi)] = p    
                            
            assert np.any(p_vec>trim_lvl), 'Everything is zero?'              
            p_vec = p_vec / np.sum(p_vec)
            p_mat[:,iz] = p_vec
            
        return p_mat.T
    
    
    def mar_mats_combine(self):
        # for time moment and each position in grid for singles (ia,iz)
        # it computes probability distribution over potential matches
        # this is relevant for testing and simulations
        
        

        
        self.matches = dict()
        
        for female in [True,False]:
            
            desc = 'Female, single' if female else 'Male, single'
            g='f' if female else 'm'
            
            self.matches[desc]=dict()
            
            for e in ['e','n']:

                pmat_a = self.prob_a_mat[female]
                imat_a = self.i_a_mat[female]
                
                self.matches[desc][e]=dict()
                
                for eo in ['e','n']:
                    pmats = self.part_mats[desc][e][eo]
                    
                    
                    match_matrix = list()
                    
                    for t in range(self.pars['T']-1):
                        pmat_iexo = pmats[t] # nz X nexo
                        # note that here we do not use transpose
                        
                        nz = pmat_iexo.shape[0]
                        
                        inds = np.where( np.any(pmat_iexo>-10,axis=0) )[0]
                        
                        npos_iexo = inds.size
                        npos_a = pmat_a.shape[1]
                        npos = npos_iexo*npos_a
                        pmatch = np.zeros((self.na,nz,npos),dtype=self.dtype)
                        iamatch = np.zeros((self.na,nz,npos),dtype=np.int32)
                        iexomatch = np.zeros((self.na,nz,npos),dtype=np.int32)
                        
                        i_conv = np.zeros((npos_iexo,npos_a),dtype=np.int32)
                        
                        for ia in range(npos_a):
                            i_conv[:,ia] = np.arange(npos_iexo*ia,npos_iexo*(ia+1))
                         
                        
                        for iz in range(nz):
                            probs = pmat_iexo[iz,inds]
                            
                            for ia in range(npos_a):
                                
                                pmatch[:,iz,(npos_iexo*ia):(npos_iexo*(ia+1))] = \
                                    (pmat_a[:,ia][:,None])*(probs[None,:])
                                iamatch[:,iz,(npos_iexo*ia):(npos_iexo*(ia+1))] = \
                                    imat_a[:,ia][:,None]
                                iexomatch[:,iz,(npos_iexo*ia):(npos_iexo*(ia+1))] = \
                                    inds[None,:]
                                    
                                
                        assert np.allclose(np.sum(pmatch,axis=2),1.0)
                        match_matrix.append({'p':pmatch,'ia':iamatch,'iexo':iexomatch,'iconv':i_conv})
                            
                    self.matches[desc][e][eo] = match_matrix
                 
        
    
    
    def all_indices(self,t,ind_or_inds=None):
        
        # just return ALL indices if no argument is called
        if ind_or_inds is None: 
            ind_or_inds = np.array(range(self.pars['nexo_t'][t]))
        
        if isinstance(ind_or_inds,tuple):
            izf,izm,ipsi = ind_or_inds
            ind = izf*self.pars['n_zm_t'][t]*self.pars['n_psi_t'][t] + izm*self.pars['n_psi_t'][t] + ipsi
        else:
            ind = ind_or_inds
            izf = ind // (self.pars['n_zm_t'][t]*self.pars['n_psi_t'][t])
            izm = (ind - izf*self.pars['n_zm_t'][t]*self.pars['n_psi_t'][t]) // self.pars['n_psi_t'][t]
            ipsi = ind - izf*self.pars['n_zm_t'][t]*self.pars['n_psi_t'][t] - izm*self.pars['n_psi_t'][t]
            
        return ind, izf, izm, ipsi

    
    # functions u_mult and c_mult are meant to be shape-perservings
    
    def u_mult(self,theta):
        assert np.all(theta > 0) and np.all(theta < 1)
        powr = (1+self.pars['couple_rts'])/(self.pars['couple_rts']+self.pars['crra_power'])
        tf = theta
        tm = 1-theta
        ces = (tf**powr + tm**powr)**(1/powr)
        umult = (self.pars['A']**(1-self.pars['crra_power']))*ces
        
        
        
        assert umult.shape == theta.shape
        
        return umult
    
    
    def c_mult(self,theta):
        assert np.all(theta > 0) and np.all(theta < 1)
        powr = (1+self.pars['couple_rts'])/(self.pars['couple_rts']+self.pars['crra_power'])
        irho = 1/(1+self.pars['couple_rts'])
        irs  = 1/(self.pars['couple_rts']+self.pars['crra_power'])
        tf = theta
        tm = 1-theta
        bottom = (tf**(powr) + tm**(powr))**irho 
        
        kf = self.pars['A']*(tf**(irs))/bottom
        km = self.pars['A']*(tm**(irs))/bottom
        
        assert kf.shape == theta.shape
        assert km.shape == theta.shape
        
        return kf, km
    
    def u(self,c):
        return u_aux(c,self.pars['crra_power'])#(c**(1-self.pars['crra_power']))/(1-self.pars['crra_power'])
    
    
    
    
    def u_pub(self,x,l,mt=0.0):
        alp = self.pars['util_alp_m']
        xi = self.pars['util_xi']
        lam = self.pars['util_lam']
        kap = self.pars['util_kap_m']        
        return alp*(x**lam + kap*(1+mt-l)**lam)**((1-xi)/lam)/(1-xi)
    
    
    def u_part(self,c,x,il,theta,ushift,psi): # this returns utility of each partner out of some c
        kf, km = self.c_mult(theta)   
        l = self.ls_levels[il]
        upub = self.u_pub(x,l,mt=1.0-self.mlevel) + ushift + psi*self.pars['multpsi']
        return self.u(kf*c) + upub, self.u(km*c) + upub
    
    def u_couple(self,c,x,il,theta,ushift,psi): # this returns utility of each partner out of some c
        umult = self.u_mult(theta) 
        l = self.ls_levels[il]
        return umult*self.u(c) + self.u_pub(x,l,mt=1.0-self.mlevel) + ushift + psi*self.pars['multpsi']
    
    def u_single_pub(self,c,x,l):
        return self.u(c) + self.u_pub(x,l)
    
        
        
    
    def u_precompute(self):
        from intratemporal import int_sol
        sig = self.pars['crra_power']
        alp = self.pars['util_alp_m']
        xi = self.pars['util_xi']
        lam = self.pars['util_lam']
        kap = self.pars['util_kap_m']
        
        nm = self.mgrid.size
        ntheta = self.ntheta
        nl = self.nls
        
        uout = np.empty((nm,ntheta,nl),dtype=self.dtype)
        xout = np.empty((nm,ntheta,nl),dtype=self.dtype)
        
        for il in range(nl):
            for itheta in range(ntheta):
                A = self.u_mult(self.thetagrid[itheta])
                ls = self.ls_levels[il]
                x, c, u = int_sol(self.mgrid,A=A,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=ls,mt=1.0-self.mlevel)
                uout[:,itheta,il] = u
                xout[:,itheta,il] = x
                
                
        self.ucouple_precomputed_u = uout
        self.ucouple_precomputed_x = xout
                
        
        # singles have just one level of labor supply (work all the time)
        
        xout, cout, uout = int_sol(self.mgrid,A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=self.ls_levels[-1])#self.ls_levels[-1]
        self.usinglef_precomputed_u = uout
        self.usinglef_precomputed_x = xout
        xout, cout, uout = int_sol(self.mgrid,A=1,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=self.mlevel)
        self.usinglem_precomputed_u = uout
        self.usinglem_precomputed_x = xout
    

#from numba import jit
#@jit(nopython=True)
def u_aux(c,sigma):
    # this is pretty important not to have (c^sigma - 1) here as it is hard to 
    # keep everywhere and occasionally this generates nasty mistakes
    if sigma!=1:
        return (c**(1-sigma))/(1-sigma)
    else:
        return np.log(c)

    


class DivorceCosts(object):
    # this is something that regulates divorce costs
    # it aims to be fully flexible
    def __init__(self, 
                 unilateral_divorce=True, # whether to allow for unilateral divorce
                 assets_kept = 1.0, # how many assets of couple are splited (the rest disappears)
                 u_lost_m=0.0,u_lost_f=0.0, # pure utility losses b/c of divorce
                 money_lost_m=0.0,money_lost_f=0.0, # pure money (asset) losses b/c of divorce
                 prog=0.0,
                 money_lost_m_ez=0.0,money_lost_f_ez=0.0, # money losses proportional to exp(z) b/c of divorce
                 eq_split=1.0 #The more of less equal way assets are split within divorce
                 ): # 
        
        self.unilateral_divorce = unilateral_divorce # w
        self.assets_kept = assets_kept
        self.u_lost_m = u_lost_m
        self.u_lost_f = u_lost_f
        self.money_lost_m = money_lost_m
        self.money_lost_f = money_lost_f
        self.money_lost_m_ez = money_lost_m_ez
        self.money_lost_f_ez = money_lost_f_ez
        self.eq_split = eq_split
        self.prog=prog
        
    def shares_if_split(self,income_share_f):
        
        
        shf=(0.5*self.eq_split + income_share_f*(1-self.eq_split))
        share_f = self.assets_kept*shf - self.money_lost_f
        share_m = self.assets_kept*(1-shf) - self.money_lost_m
        
        return share_f, share_m
    
    
    def shares_if_split_theta(self,setup,theta):
        
        #First build the title based sharing rule
        sharef=setup.c_mult(theta)[0]
        shf=(0.5*self.eq_split + sharef*(1-self.eq_split))
        share_f = self.assets_kept*shf
        
        return share_f
       
        
def tauchen_drift(z_now,z_next,rho,sigma,mu,mat):
    z_now = np.atleast_1d(z_now)
    z_next = np.atleast_1d(z_next)
    if z_next.size == 1:
        return np.ones((z_now.size,1),dtype=z_now.dtype)
    
    d = np.diff(z_next)
    assert np.ptp(d) < 1e-5, 'Step size should be fixed'
    
    h_half = d[0]/2
    
    Pi = np.zeros((z_now.size,z_next.size),dtype=z_now.dtype)
    Pii = np.zeros((z_now.size,z_next.size),dtype=z_now.dtype)
    
    ez = rho*z_now + mu
    
    
    def f(x):
        
        pi=int_prob(z_next,mu=x,sig=sigma)
        return np.exp(ez[j])/np.exp(np.sum(z_next*pi))-1.0

    for j in range(z_next.size):
        Pi[j,:]=int_prob(z_next,mu=ez[j],sig=sigma)
        if (abs(ez[j]-np.sum(z_next*Pi[j,:]))>0.001):
            
            if (f(ez[j]-1.0)>0 and f(ez[j]+1.0)<0):
                sol = optimize.root_scalar(f, x0=ez[j],bracket=[ez[j]-1.0, ez[j]+1.0], maxiter=200,xtol=0.0001,method='bisect')
                mu1=sol.root
            #mu1=rho*z_now[j]+mu-(-ez[j]+np.sum(z_next*Pi[j,:]))
                Pi[j,:]=int_prob(z_next,mu=mu1,sig=sigma)
            
        
   
       
            
            # if(-mu1+np.sum(z_next*Pi[j,:])<-0.01):
            #     mu1=mu1-(-mu1+np.sum(z_next*Pi[j,:]))
            #     Pi[j,:]=int_prob(z_next,mu=mu1,sig=sigma)
                
            # if(-mu1+np.sum(z_next*Pi[j,:])>0.01):
            #     mu2=mu1+(-mu1+np.sum(z_next*Pi[j,:]))
            #     Pi[j,:]=int_prob(z_next,mu=mu2,sig=sigma)
        

    
    # Pi[:,0] = normcdf_tr( ( z_next[0] + h_half - ez )/sigma)
    # Pi[:,-1] = 1 - normcdf_tr( (z_next[-1] - h_half - ez ) / sigma )
    # for j in range(1,z_next.size - 1):
    #     Pi[:,j] = normcdf_tr( ( z_next[j] + h_half - ez )/sigma) - \
    #         normcdf_tr( ( z_next[j] - h_half - ez )/sigma)
    #for j in range(z_next.size):print(ez[j],np.sum(z_next*Pi[j,:]))
    return Pi
        

def build_s_grid(agrid,n_between,da_min,da_max):
    sgrid = np.array([0.0],agrid.dtype)
    for j in range(agrid.size-1):
        step = (agrid[j+1] - agrid[j])/n_between
        if step >= da_min and step <= da_max:
            s_add = np.linspace(agrid[j],agrid[j+1],n_between)[:-1]
        elif step < da_min:
            s_add = np.arange(agrid[j],agrid[j+1],da_min)
        elif step > da_max:
            s_add = np.arange(agrid[j],agrid[j+1],da_max)
        sgrid = np.concatenate((sgrid,s_add))
    
    sgrid = np.concatenate((sgrid,np.array([agrid[-1]])))
            
    if sgrid[0] == sgrid[1]: 
        sgrid = sgrid[1:]
        
    return sgrid
