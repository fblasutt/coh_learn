#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This contains things relevant for setting up the model
"""

import numpy as np

from rw_approximations import rouw_nonst, normcdf_tr,tauchen_nonst#,tauchen_nonstm
from rw_approximations import rouw_nonstm as tauchen_nonstm
from mc_tools import combine_matrices_two_lists, int_prob,cut_matrix
from scipy.stats import norm
from collections import namedtuple
from gridvec import VecOnGrid
from statutils import kalman
from scipy import optimize
from scipy import sparse



class ModelSetup(object):
    def __init__(self,nogrid=False,divorce_costs='Default',separation_costs='Default',**kwargs): 
        p = dict()       
        period_year=1#this can be 1,2,3 or 6
        transform=1#this tells how many periods to pull together for duration moments
        T = int(62/period_year)
        Tret = int(47/period_year) # first period when the agent is retired
        Tbef=int(2/period_year)
        Tren  = int(47/period_year)#int(42/period_year) # period starting which people do not renegotiate/divroce
        Tmeet = int(47/period_year)#int(42/period_year) # period starting which you do not meet anyone
        dm=4
        p['dm']=dm
        p['py']=period_year
        p['ty']=transform
        p['T'] = T
        p['Tret'] = Tret
        p['Tren'] = Tren
        p['Tbef'] = Tbef
        p['sig_zf_0']  = .5449176#mk0.4096**(0.5)
        p['sig_zf']    = .0272437**(0.5)#0.0399528**(0.5)
        p['n_zf_t']      = [5]*Tret + [1]*(T-Tret)
        p['sig_zm_0']  = 0.54896510#.405769**(0.5)
        p['sig_zm']    = .025014**(0.5)#0.0417483**(0.5)
        p['n_zm_t']      = [5]*Tret + [1]*(T-Tret)
        p['sigma_psi_mult'] = 0.28
        p['sigma_psi_mu'] = 0.0#1.0#nthe1.1
        p['sigma_psi']   = 0.11
        p['R_t'] = [1.02**period_year]*T
        p['n_psi_t']     = [11]*T
        p['beta_t'] = [0.98**period_year]*T
        p['A'] = 1.0 # consumption in couple: c = (1/A)*[c_f^(1+rho) + c_m^(1+rho)]^(1/(1+rho))
        p['crra_power'] = 1.5
        p['couple_rts'] = 0.0 
        p['sig_partner_a'] = 0.1#0.5
        p['sig_partner_z'] = 1.2#1.0#0.4 #This is crazy powerful for the diff in diff estimate
        p['sig_partner_mult'] = 1.0
        p['dump_factor_z'] = 0.85#0.82
        p['mean_partner_z_female'] = 0.02#+0.03
        p['mean_partner_z_male'] =  -0.02#-0.03
        p['mean_partner_a_female'] = 0.0#0.1
        p['mean_partner_a_male'] = 0.0#-0.1
        p['m_bargaining_weight'] = 0.5
        p['pmeet'] = 0.5
        p['pmeet1'] = 0.0
        
        p['z_drift'] = -0.09#-0.1
        
        
        p['wage_gap'] = 0.6
        p['wret'] = 0.6#0.5
        p['uls'] = 0.2
        p['pls'] = 1.0
        
        
        
        p['u_shift_mar'] = 0.0
        p['u_shift_coh'] =0.00
        
         
        #Wages over time
        p['f_wage_trend'] = [0.0*(t>=Tret)+(t<Tret)*(-.74491918 +.04258303*t -.0016542*t**2+.00001775*t**3) for t in range(T)]
        p['f_wage_trend_single'] = [0.0*(t>=Tret)+(t<Tret)*(-.6805060 +.04629912*t -.00160467*t**2+.00001626*t**3) for t in range(T)]
        p['m_wage_trend'] = [0.0*(t>=Tret)+(t<Tret)*(-0.5620125  +0.06767721*t -0.00192571*t**2+ 0.00001573*t**3) for t in range(T)]
        p['m_wage_trend_single'] = [0.0*(t>=Tret)+(t<Tret)*(-.5960803  +.05829568*t -.00169143*t**2+ .00001446*t**3) for t in range(T)]
   
        
  
        p['util_lam'] = 0.19#0.4
        p['util_alp'] = 0.5
        p['util_xi'] = 1.07
        p['util_kap'] = (1-0.21)/(0.21)
        p['rprice_durables'] = 1.0#
        

        
        for key, value in kwargs.items():
            assert (key in p), 'wrong name?'
            p[key] = value
            
        #Adjust kappa and alpha to make sense of relative prices
        p['util_alp_m']=p['util_alp']*(1.0/(p['rprice_durables'])**(1.0-p['util_xi']))
        p['util_kap_m']=p['util_kap']*p['rprice_durables']**p['util_lam']
            
            

        #Get the probability of meeting, adjusting for year-period
        p_meet=p['pmeet']
        p_meet1=p['pmeet1']
        for j in range(period_year-1):
            p_meet=p_meet+(1-p_meet)*p['pmeet']
            
            
        # timing here    
        p['pmeet_t'] =[0.0*(t>=Tmeet)+(t<Tret)*(min(max(p_meet+p_meet1*t,0),1)) for t in range(T)]
        p['can divorce'] = [True]*Tren + [False]*(T-Tren)
        

        
        
        def f(x): 
            return (x**(3/2) - (p['sigma_psi_mu']**2+x)*(p['sigma_psi_mult']*p['sigma_psi']))
            #return (np.sqrt(x)*x)-(p['sigma_psi_mu']**2+x)*(p['sigma_psi_mult']*p['sigma_psi'])
        
  
        def ff(x):
            return np.sqrt((x**2+p['sigma_psi_mu']**2)*((x**2+p['sigma_psi_mu']**2)/(x**2+2*p['sigma_psi_mu']**2))**2)
        #root= optimize.brentq(f, 0.0001, 10)
        #sigma0=np.sqrt(root-p['sigma_psi_mu']**2)
        #sigma0=np.sqrt(root)
        p['sigma_psi_init'] =p['sigma_psi_mult']*p['sigma_psi']
  
        
        #Get Variance of Love shock by Duration using Kalman Filter
        self.K,sigma=kalman(1.0,p['sigma_psi']**2,p['sigma_psi_mu']**2,(p['sigma_psi_mult']*p['sigma_psi'])**2,p['dm']+1)
        #K,sigma=kalman(1.0,p['sigma_psi']**2,p['sigma_psi_mu']**2,(sigma0)**2,p['dm'])
        #Get variance by duration
        self.sigmad=-1*np.ones((p['dm']))
        
        p['sigma_psi_init']=np.sqrt(self.K[0]**2*(sigma[0]**2+p['sigma_psi_mu']**2))
        for i in range(p['dm']):
            self.sigmad[i]=np.sqrt(self.K[i+1]**2*(sigma[i+1]**2+p['sigma_psi_mu']**2))
        
        
        self.pars = p
        
        self.dtype = np.float64 # type for all floats
        
       
        
        # relevant for integration
        self.state_names = ['Female, single','Male, single','Couple, M', 'Couple, C']
        
        # female labor supply
        self.ls_levels = np.array([0.0,0.8],dtype=self.dtype)
        self.mlevel=0.8
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
            
            p['n_zf_t']      = [5]*Tret + [5]*(T-Tret)
            p['n_zm_t']      = [5]*Tret + [5]*(T-Tret)
            exogrid['zf_t'],  exogrid['zf_t_mat'] = rouw_nonst(p['T'],p['sig_zf']*period_year**0.5,p['sig_zf_0'],p['n_zf_t'][0])
            exogrid['zm_t'],  exogrid['zm_t_mat'] = rouw_nonst(p['T'],p['sig_zm']*period_year**0.5,p['sig_zm_0'],p['n_zm_t'][0])
            
            ################################
            #First mimic US pension system
            ###############################
                      
            #function to compute pension
            def pens(value):
                
                #Get median income before retirement using men model income in Tret-1
                #+ ratio of men and female labor income for the rest
                yret=(1.73377+(.8427056/1.224638)*1.73377* 0.3246206)/(1+0.3246206)
                thresh1=0.38*yret
                thresh2=1.59*yret
                
                inc1=np.minimum(value,thresh1)
                inc2=np.maximum(np.minimum(value-inc1,thresh2-inc1),0)
                inc3=np.maximum(value-thresh2,0)
                
                return inc1*0.9+inc2*0.32+inc3*0.15            
            
            for t in range(Tret,T):
                exogrid['zf_t'][t] = np.array([np.log(p['wret'])])
                exogrid['zm_t'][t] = np.array([np.log(p['wret'])])
                exogrid['zf_t_mat'][t] = np.atleast_2d(1.0)
                exogrid['zm_t_mat'][t] = np.atleast_2d(1.0)
                
                
            # fix transition from non-retired to retired    
            exogrid['zf_t_mat'][Tret-1] = np.ones((p['n_zf_t'][Tret-1],1))
            exogrid['zm_t_mat'][Tret-1] = np.ones((p['n_zm_t'][Tret-1],1))
            
            #Tax system as in Wu and Kruger
            for t in range(0,Tret):
                exogrid['zf_t'][t] = exogrid['zf_t'][t]#*(1-0.1327)+np.log(1-0.1575)
                exogrid['zm_t'][t] = exogrid['zm_t'][t]#*(1-0.1327)+np.log(1-0.1575)  
            
            #Comment out the following if you dont want retirment based on income
            for t in range(Tret,T):
               
                exogrid['zf_t'][t] = np.log(pens(np.exp(p['f_wage_trend'][Tret-1]+exogrid['zf_t'][Tret-1])))#np.array([np.log(p['wret'])])
                exogrid['zm_t'][t] = np.log(pens(np.exp(p['m_wage_trend'][Tret-1]+exogrid['zm_t'][Tret-1])))               
                exogrid['zf_t_mat'][t] = np.diag(np.ones(len(exogrid['zf_t'][t])))#p.atleast_2d(1.0)
                exogrid['zm_t_mat'][t] = np.diag(np.ones(len(exogrid['zm_t'][t])))
                
            # fix transition from non-retired to retired    
            exogrid['zf_t_mat'][Tret-1] = np.diag(np.ones(len(exogrid['zf_t'][Tret-1])))
            exogrid['zm_t_mat'][Tret-1] = np.diag(np.ones(len(exogrid['zm_t'][Tret-1])))

            ###########################
            #Love shock grid
            ###########################
            
            #Idea: first build grid with variance in dmax, then dmax-1 and so on are 
            #created using the Fella routine backwards
            
            print('variances are {}, {}, {}, {}, {}'.format(self.pars['sigma_psi_init'],self.sigmad[0],self.sigmad[1],self.sigmad[2],self.sigmad[3]))
            print(self.K[0],self.K[1],self.K[2],self.K[3])
            #New way of getting transition matrix
            psit, matri=list(np.ones((T))),list(np.ones((T)))
            sigmabase=np.sqrt([self.sigmad[0]**2+(t+1)*self.sigmad[-1]**2 for t in range(T)])
            sigmadp=np.concatenate((np.array([0.0]),self.sigmad))
            sigmadi=self.sigmad[::-1]
            for i in range(T):
                
                base=sigmabase[min(i+p['dm'],T-1)]**2-np.sum(self.sigmad**2)
                sigp=np.sqrt([base+np.sum(sigmadi[p['dm']-dd:]**2) for dd in range(p['dm']+1)])
                psit[i],matri[i] = tauchen_nonstm(p['dm']+1,sigmadp*period_year**0.5,0.0,p['n_psi_t'][0],sd_z=sigp)
                

            exogrid['psi_t'], exogrid['psi_t_mat']=list(np.ones((p['dm']))),list(np.ones((p['dm'])))
            for dd in range(p['dm']):
                
                
                #exogrid['psi_t'][dd], exogrid['psi_t_mat'][dd] = tauchen_nonst(p['T'],self.sigmad[dd],self.sigmad[dd],p['n_psi_t'][0])
                exogrid['psi_t'][dd], exogrid['psi_t_mat'][dd] = tauchen_nonst(p['T'],self.sigmad[dd],np.sqrt(self.sigmad[0]**2+self.sigmad[dd]**2)*period_year**0.5,p['n_psi_t'][0])
                for i in range(T):
                    
                    if i<Tret:
                        exogrid['psi_t'][dd][i], exogrid['psi_t_mat'][dd][i]=psit[max(i-dd,0)][dd],matri[min(i-dd,T-1)][dd]
                        #exogrid['psi_t'][dd][i], exogrid['psi_t_mat'][dd][i]=psit[max(i+dd-p['dm'],0)][dd],matri[min(i+dd-p['dm'],T-1)][dd]
                        #exogrid['psi_t'][dd][i], exogrid['psi_t_mat'][dd][i]=psit[i][dd],matri[i][dd]


            aa,bb= rouw_nonst(p['T'],self.sigmad[dd],np.sqrt(self.sigmad[0]**2+self.sigmad[dd]**2),p['n_psi_t'][0])
            #Here I impose no change in psi from retirement till the end of time 
            for t in range(Tren,T-1):
                for dd in range(p['dm']):
               
                    exogrid['psi_t'][dd][t] = exogrid['psi_t'][dd][Tren-1]#np.array([np.log(p['wret'])])             
                    exogrid['psi_t_mat'][dd][t] = np.diag(np.ones(len(exogrid['psi_t'][dd][t])))

            
           #Now the crazy matrix for "true process"
            exogrid['noise_psi_mat'],exogrid['true_psi_mat']=exogrid['psi_t_mat'],exogrid['psi_t_mat']
            
            from mc_tools import mc_simulate
            zero=np.ones((100000),dtype=np.int32)*5
            s1=mc_simulate(zero,exogrid['psi_t_mat'][0][0])
            s1e=exogrid['psi_t'][0][1][s1]
            s2=mc_simulate(s1,exogrid['psi_t_mat'][0][1])
            s2e=exogrid['psi_t'][1][2][s2]
            diffe=s2e-s1e
#            
#            for dd in range(p['dm']):
#                if p['sigma_psi_mu']>0:
#                    for t in range(T-1):
#                        if t<Tret:
#                            
#                            #True Process
#                            mat=exogrid['psi_t_mat'][dd][t].copy()
#                            for i in range(p['n_psi_t'][0]):
#                                mat[i,:]=int_prob(exogrid['psi_t'][min(dd+1,p['dm']-1)][min(t+1,T-1)],
#                                                           mu=exogrid['psi_t'][dd][t][i],
#                                                           sig=p['sigma_psi'],
#                                                           n_points=p['n_psi_t'][0])
#                            exogrid['true_psi_mat'][dd][t]=mat
#                            
#                            #Noisy Update
#                            mat=exogrid['psi_t_mat'][dd][t].copy()
#                            for i in range(p['n_psi_t'][0]):
#                                mat[i,:]=int_prob(exogrid['psi_t'][min(dd,p['dm']-1)][min(t,T-1)],
#                                                           mu=exogrid['psi_t'][dd][t][i],
#                                                           sig=p['sigma_psi_mu'],
#                                                           n_points=p['n_psi_t'][0])
#                            exogrid['noise_psi_mat'][dd][t]=mat
#                else:
#                    exogrid['noise_psi_mat'][dd][t]=np.diag(np.ones(len(exogrid['psi_t'][dd][t])))
#                        
                   
           # zfzm, zfzmmat = combine_matrices_two_lists(exogrid['zf_t'], exogrid['zm_t'], zf_t_mat_down, exogrid['zm_t_mat'])
            
            exogrid['all_t_mat_by_l'],  exogrid['all_t_mat_by_l_spt'],exogrid['all_t']=list(np.ones((p['dm']))),list(np.ones((p['dm']))),list(np.ones((p['dm'])))
            for dd in range(p['dm']):
                
                zfzm, zfzmmat = combine_matrices_two_lists(exogrid['zf_t'], exogrid['zm_t'], exogrid['zf_t_mat'], exogrid['zm_t_mat'])
                all_t, all_t_mat = combine_matrices_two_lists(zfzm,exogrid['psi_t'][dd],zfzmmat,exogrid['psi_t_mat'][dd])
                all_t_mat_sparse_T = [sparse.csc_matrix(D.T) if D is not None else None for D in all_t_mat]
                
                
                
                
                     
                zf_bad = [tauchen_drift(exogrid['zf_t'][t], exogrid['zf_t'][t+1], 
                                        1.0, p['sig_zf'], p['z_drift'])
                            for t in range(self.pars['Tret']-1) ]
                
                #Account for retirement here
                zf_bad = zf_bad+[exogrid['zf_t_mat'][t] for t in range(self.pars['Tret']-1,self.pars['T']-1)]+ [None]
                
                zf_t_mat_down = zf_bad
                zfzm, zfzmmat = combine_matrices_two_lists(exogrid['zf_t'], exogrid['zm_t'], zf_t_mat_down, exogrid['zm_t_mat'])
                all_t_down, all_t_mat_down = combine_matrices_two_lists(zfzm,exogrid['psi_t'][dd],zfzmmat,exogrid['psi_t_mat'][dd])
                all_t_mat_down_sparse_T = [sparse.csc_matrix(D.T) if D is not None else None for D in all_t_mat_down]
                
                
                
                all_t_mat_by_l = [ [(1-p)*m + p*md if m is not None else None 
                                    for m , md in zip(all_t_mat,all_t_mat_down)]
                                   for p in self.ls_pdown ]
                
                all_t_mat_by_l_spt = [ [(1-p)*m + p*md if m is not None else None
                                        for m, md in zip(all_t_mat_sparse_T,all_t_mat_down_sparse_T)]
                                   for p in self.ls_pdown ]
                
                
                
                exogrid['all_t_mat_by_l'][dd] = all_t_mat_by_l
                exogrid['all_t_mat_by_l_spt'][dd] = all_t_mat_by_l_spt
                
                exogrid['all_t'][dd] = all_t
            
            Exogrid_nt = namedtuple('Exogrid_nt',exogrid.keys())
            
            self.exogrid = Exogrid_nt(**exogrid)
            self.pars['nexo_t'] = [v.shape[0] for v in all_t]
            
            #assert False
            
            
            
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
        self.ntheta = 11
        self.thetamin = 0.02
        self.thetamax = 0.98
        self.thetagrid = np.linspace(self.thetamin,self.thetamax,self.ntheta,dtype=self.dtype)
        
        
        
        
        
        
        # construct finer grid for bargaining
        ntheta_fine = 5*self.ntheta # actual number may be a bit bigger
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

            
        
        
        


        self.exo_grids = {'Female, single':exogrid['zf_t'],
                          'Male, single':exogrid['zm_t'],
                          'Couple, M':exogrid['all_t'],
                          'Couple, C':exogrid['all_t']}
        self.exo_mats = {'Female, single':exogrid['zf_t_mat'],
                          'Male, single':exogrid['zm_t_mat'],
                          'Couple, M':exogrid['all_t_mat_by_l'],
                          'Couple, C':exogrid['all_t_mat_by_l']} # sparse version?
        
        
        self.utility_shifters = {'Female, single':0.0,
                                 'Male, single':0.0,
                                 'Couple, M':p['u_shift_mar'],
                                 'Couple, C':p['u_shift_coh']}
        
        
        # this pre-computes transition matrices for meeting a partner
        zf_t_partmat,zm_t_partmat=list(np.ones((self.pars['dm']))),list(np.ones((self.pars['dm'])))
        for dd in range(self.pars['dm']):
            zf_t_partmat[dd] = [self.mar_mats_iexo(t,dd,female=True) if t < p['T'] - 1 else None 
                            for t in range(p['T'])]
            zm_t_partmat[dd] = [self.mar_mats_iexo(t,dd,female=False) if t < p['T'] - 1 else None 
                            for t in range(p['T'])]
        
        self.part_mats = {'Female, single':zf_t_partmat,
                          'Male, single':  zm_t_partmat,
                          'Couple, M': None,
                          'Couple, C': None} # last is added for consistency
        
        self.mar_mats_assets()
        
        self.mar_mats_combine()
        
        
        # building m grid
        ezfmin = min([np.min(self.ls_levels[-1]*np.exp(g+t)) for g,t in zip(exogrid['zf_t'],p['f_wage_trend'])])
        ezmmin = min([np.min(self.mlevel*np.exp(g+t)) for g,t in zip(exogrid['zm_t'],p['m_wage_trend'])])
        ezfmax = max([np.max(self.ls_levels[-1]*np.exp(g+t)) for g,t in zip(exogrid['zf_t'],p['f_wage_trend'])])
        ezmmax = max([np.max(self.mlevel*np.exp(g+t)) for g,t in zip(exogrid['zm_t'],p['m_wage_trend'])])
        
        
        
        
        self.money_min = 0.95*min(ezmmin,ezfmin) # cause FLS can be up to 0
        #self.mgrid = ezmmin + self.sgrid_c # this can be changed later
        mmin = self.money_min
        mmax = ezfmax + ezmmax + np.max(self.pars['R_t'])*self.amax1
        mint = (ezfmax + ezmmax) # poin where more dense grid begins
        
        ndense = 600
        nm = 1500
        
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
            

        
    
    def mar_mats_iexo(self,t,dd,female=True,trim_lvl=0.00):
        # TODO: check timing
        # this returns transition matrix for single agents into possible couples
        # rows are single's states
        # columnts are couple's states
        # you have to transpose it if you want to use it for integration
        setup = self
        
        nexo = setup.pars['nexo_t'][t]
        sigma_psi_init = setup.pars['sigma_psi_init']
        #sig_z_partner = setup.pars['sig_partner_z']
        psi_couple = setup.exogrid.psi_t[dd][t+1]
        
        
        if female:
            nz_single = setup.exogrid.zf_t[t].shape[0]
            p_mat = np.empty((nexo,nz_single))
            z_own = setup.exogrid.zf_t[t]
            n_zown = z_own.shape[0]
            z_partner = setup.exogrid.zm_t[t]
            zmat_own = setup.exogrid.zf_t_mat[t]
            trend=setup.pars['m_wage_trend_single'][t]
            mean=setup.pars['mean_partner_z_female']-setup.pars['m_wage_trend'][t]+setup.pars['m_wage_trend_single'][t]
            sig_z_partner=(setup.pars['sig_zm_0']**2+(t+1)*setup.pars['sig_zm']**2)**0.5
        else:
            nz_single = setup.exogrid.zm_t[t].shape[0]
            p_mat = np.empty((nexo,nz_single))
            z_own = setup.exogrid.zm_t[t]
            n_zown = z_own.shape[0]
            z_partner = setup.exogrid.zf_t[t]
            zmat_own = setup.exogrid.zm_t_mat[t]    
            trend=setup.pars['f_wage_trend_single'][t]
            mean=setup.pars['mean_partner_z_male']-setup.pars['f_wage_trend'][t]+setup.pars['f_wage_trend_single'][t]
            sig_z_partner=(setup.pars['sig_zf_0']**2+(t+1)*setup.pars['sig_zf']**2)**0.5
            
        def ind_conv(a,b,c): return setup.all_indices(t,(a,b,c))[0]
        
        
        for iz in range(n_zown):
            p_psi = int_prob(psi_couple,mu=0,sig=sigma_psi_init)
            if female:
                p_zm  = int_prob(z_partner, mu=setup.pars['dump_factor_z']*z_partner[iz]+
                                  mean+setup.pars['mean_partner_z_female'],sig=(1-setup.pars['dump_factor_z'])**
                                  0.5*sig_z_partner*setup.pars['sig_partner_mult'])
                p_zf  = zmat_own[iz,:]
            else:
                p_zf  = int_prob(z_partner, mu=setup.pars['dump_factor_z']*z_partner[iz]+ 
                                 mean+setup.pars['mean_partner_z_male'],sig=(1-setup.pars['dump_factor_z'])**
                                 0.5*sig_z_partner*setup.pars['sig_partner_mult'])
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
            
            pmat_a = self.prob_a_mat[female]
            imat_a = self.i_a_mat[female]
            
            pmats = self.part_mats[desc] 
            
            
            match_matrix = list()
            
            for t in range(self.pars['T']-1):
                pmat_iexo = pmats[0][t] # nz X nexo
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
                    
            self.matches[desc] = match_matrix
         
        
    
    
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
        upub = self.u_pub(x,l,mt=1.0-self.mlevel) + ushift + psi
        return self.u(kf*c) + upub, self.u(km*c) + upub
    
    def u_couple(self,c,x,il,theta,ushift,psi): # this returns utility of each partner out of some c
        umult = self.u_mult(theta) 
        l = self.ls_levels[il]
        return umult*self.u(c) + self.u_pub(x,l,mt=1.0-self.mlevel) + ushift + psi
    
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
       
        
def tauchen_drift(z_now,z_next,rho,sigma,mu):
    z_now = np.atleast_1d(z_now)
    z_next = np.atleast_1d(z_next)
    if z_next.size == 1:
        return np.ones((z_now.size,1),dtype=z_now.dtype)
    
    d = np.diff(z_next)
    assert np.ptp(d) < 1e-5, 'Step size should be fixed'
    
    h_half = d[0]/2
    
    Pi = np.zeros((z_now.size,z_next.size),dtype=z_now.dtype)
    
    ez = rho*z_now + mu
    
    Pi[:,0] = normcdf_tr( ( z_next[0] + h_half - ez )/sigma)
    Pi[:,-1] = 1 - normcdf_tr( (z_next[-1] - h_half - ez ) / sigma )
    for j in range(1,z_next.size - 1):
        Pi[:,j] = normcdf_tr( ( z_next[j] + h_half - ez )/sigma) - \
                    normcdf_tr( ( z_next[j] - h_half - ez )/sigma)
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
