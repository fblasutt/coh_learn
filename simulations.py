#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This contains things relevant for simulations
"""

import numpy as np


from mc_tools import mc_simulate, int_prob
from gridvec import VecOnGrid


class Agents:

    def __init__(self,Mlist,age_uni,female=False,pswitchlist=None,N=15000,T=None,verbose=True,nosim=False):
            
            
        np.random.seed(8)
  
        # take the stuff from the model and arguments
        # note that this does not induce any copying just creates links
        
        if type(Mlist) is not list:
            Mlist = [Mlist]
            
        
        
        
        #Unilateral Divorce
        self.Mlist = Mlist
        self.Vlist = [M.V for M in Mlist]
        self.declist = [M.decisions for M in Mlist]
        self.npol = len(Mlist)
        self.transition = len(self.Mlist)>1
        
            
        if T is None:
            T = self.Mlist[0].setup.pars['T']
            
        self.setup = self.Mlist[0].setup
        self.state_names = self.setup.state_names
        self.N = N
        self.T = T
        self.verbose = verbose
        self.timer = self.Mlist[0].time
        
        self.female = female
        self.single_state = 'Female, single' if female else 'Male, single'        
        
        #Divorces
        self.divorces=np.zeros((N,T),bool)
        
        # all the randomness is here
        shokko=np.random.random_sample((9,N,T))
        self.shocks_single_iexo2 =shokko[8,:,:]# np.random.random_sample((N,T))
        self.shocks_single_iexo =shokko[0,:,:]# np.random.random_sample((N,T))
        self.shocks_single_meet =shokko[1,:,:]# np.random.random_sample((N,T))
        self.shocks_couple_iexo =shokko[2,:,:]# np.random.random_sample((N,T))
        self.shocks_single_a =shokko[3,:,:]# np.random.random_sample((N,T))
        self.shocks_couple_a =shokko[4,:,:]# np.random.random_sample((N,T))
      
        self.shocks_div_a = shokko[5,:,:]#np.random.random_sample((N,T))
        
        
        z_t = self.setup.exogrid.zf_t if female else self.setup.exogrid.zm_t
        sig = self.setup.pars['sig_zf_0'] if female else self.setup.pars['sig_zm_0']
        z_prob = int_prob(z_t[0], sig = sig )
        shocks_init = shokko[6,:,0]#np.random.random_sample((N,))        
        i_z = np.sum((shocks_init[:,None] > np.cumsum(z_prob)[None,:]), axis=1)
        iexoinit = i_z # initial state        
        
        
        self.shocks_transition = shokko[7,:,:]#np.random.random_sample((N,T))
        # no randomnes past this line please
        
        # initialize assets
        
        self.iassets = np.zeros((N,T),np.int16)
        self.iassetss = np.zeros((N,T),np.int16)
        self.tempo=VecOnGrid(self.setup.agrid_s,self.iassets[:,0])
        
        # initialize FLS
        #self.ils=np.ones((N,T),np.float64)
        self.ils_i=np.ones((N,T),np.int8)*(len(self.setup.ls_levels)-1)
        
        
        self.ils_i[:,-1] = 5

        # initialize theta
        self.itheta = -np.ones((N,T),np.int16)
        
        # initialize iexo
        self.iexo = np.zeros((N,T),np.int16)
        self.iexos = np.zeros((N,T),np.int16)
        # TODO: look if we can/need fix the shocks here...
        
        
        
        self.iexo[:,0] = iexoinit
        self.iexos[:,0] = iexoinit
        
        
        
        # NB: the last column of these things will not be filled
        # c refers to consumption expenditures (real consumption of couples
        # may be higher b/c of returns to scale)
        self.c = np.zeros((N,T),np.float32)
        self.x = np.zeros((N,T),np.float32)
        self.s = np.zeros((N,T),np.float32)
        

        
        
        
        self.state_codes = dict()
        self.has_theta = list()
        for i, name in enumerate(self.setup.state_names):
            self.state_codes[name] = i
            self.has_theta.append((name=='Couple, C' or name=='Couple, M'))
        
        
        # initialize state
        self.state = np.zeros((N,T),dtype=np.int8)       
        self.state[:,0] = self.state_codes[self.single_state]  # everyone starts as female
      
        
        self.timer('Simulations, creation',verbose=self.verbose)
        self.ils_def = self.setup.nls - 1
        
            
            
        #Create a file with the age of the change foreach person
        
        self.policy_ind = np.zeros((N,T),dtype=np.int8)
        
        if pswitchlist == None:
            pswitchlist = [np.eye(self.npol)]*T
            
        # this simulates "exogenous" transitions of polciy functions
        # policy_ind stands for index of the policies to apply, they are
        # from 0 to (self.npol-1)
        zeros = np.zeros((N,),dtype=np.int8)
        mat_init = pswitchlist[0]
        
        self.policy_ind[:,0] = mc_simulate(zeros,mat_init,shocks=self.shocks_transition[:,0]) # everyone starts with 0
        if self.npol > 1:
            for t in range(T-1):    
                mat = pswitchlist[t+1]
                self.policy_ind[:,t+1] = mc_simulate(self.policy_ind[:,t],mat,shocks=self.shocks_transition[:,t+1])
        else:
            self.policy_ind[:] = 0
            
        if not nosim: self.simulate()
        
        
            
   
    def simulate(self):
        
        #Create Variables that stores varibles of interest
        
        
        for t in range(self.T):
         
            self.anext(t) 
            if t+1<self.T:
                self.iexonext(t)            
                self.statenext(t)
            self.timer('Simulations, iteration',verbose=self.verbose)
        
        
        #return self.gsavings, self.iexo, self.state,self.gtheta
    
    

    
    def anext(self,t):
        # finds savings (potenitally off-grid)
        
        
        for ipol in range(self.npol):
            for ist, sname in enumerate(self.state_codes):
                
                
                is_state_any_pol = (self.state[:,t]==ist)  
                is_pol = (self.policy_ind[:,min(t+1,self.T-1)]==ipol)
                
                is_state = (is_state_any_pol) & (is_pol)
                
                use_theta = self.has_theta[ist]            
                nst = np.sum(is_state)
                
                if nst==0:
                    continue
                
                ind = np.where(is_state)[0]
                
                pol = self.Mlist[ipol].decisions[t][sname]
                
                if not use_theta:
                    

                    #Dictionaries below account for 90% of the time in this function
                
                    anext = pol['s'][self.iassets[ind,t],self.iexo[ind,t]]
                    if t+1<self.T:
                        self.iassets[ind,t+1] = VecOnGrid(self.setup.agrid_s,anext).roll(shocks=self.shocks_single_a[ind,t])
                        self.iassetss[ind,t+1] = self.iassets[ind,t+1].copy()
                    self.s[ind,t] = anext
                    self.c[ind,t] = pol['c'][self.iassets[ind,t],self.iexo[ind,t]]
                    self.x[ind,t] = pol['x'][self.iassets[ind,t],self.iexo[ind,t]]
                   
                else:
                    
                    # interpolate in both assets and theta
                    # function apply_2dim is experimental but I checked it at this setup
                    
                    # apply for couples
                    
                    anext = pol['s'][self.iassets[ind,t],self.iexo[ind,t],self.itheta[ind,t]]
                    self.s[ind,t] = anext
                    self.x[ind,t] = pol['x'][self.iassets[ind,t],self.iexo[ind,t],self.itheta[ind,t]]
                    self.c[ind,t] = pol['c'][self.iassets[ind,t],self.iexo[ind,t],self.itheta[ind,t]]
                    if t+1<self.T:
                        self.iassets[ind,t+1] = VecOnGrid(self.setup.agrid_c,anext).roll(shocks=self.shocks_couple_a[ind,t])
                        self.iassetss[ind,t+1] = self.iassets[ind,t+1].copy()
                    
                assert np.all(anext >= 0)
    
    
    def iexonext(self,t):
        
        # let's find out new exogenous state
        
        for ipol in range(self.npol):
            for ist,sname in enumerate(self.state_names):
                is_state_any_pol = (self.state[:,t]==ist)
                is_pol = (self.policy_ind[:,t+1]==ipol)
                is_state = (is_state_any_pol) & (is_pol)   
                
                nst = np.sum(is_state)
                
                if nst == 0:
                    continue
                
                ind = np.where(is_state)[0]
                sname = self.state_names[ist]
                iexo_now = self.iexo[ind,t].reshape(nst)
                
                
                if sname == 'Couple, C' or sname == 'Couple, M':
                    
                    ls_val = self.ils_i[ind,t] 
                    
                    for ils in range(self.setup.nls):
                        this_ls = (ls_val==ils)                    
                        if not np.any(this_ls): continue
                    
                        cnt = np.sum(this_ls)
                        lvl = self.Mlist[ipol].setup.ls_levels[ils]
        
                        
                        if self.verbose: print('At t = {} for {} {} have LS of {}'.format(t,sname,cnt,lvl))
                        
                        
                        mat = self.Mlist[ipol].setup.exo_mats[sname][ils][t]
                        
                        shks = self.shocks_couple_iexo[ind[this_ls],t]
                        
                        #Following line takes 94% of the time for this funciton
                        iexo_next_this_ls = mc_simulate(iexo_now[this_ls],mat,shocks=shks)
                        self.iexo[ind[this_ls],t+1] = iexo_next_this_ls
                        self.iexos[ind[this_ls],t+1] = iexo_next_this_ls
                        
                else:
                    mat = self.Mlist[ipol].setup.exo_mats[sname][t]
                    shks = self.shocks_single_iexo[ind,t]                    
                    iexo_next = mc_simulate(iexo_now,mat,shocks=shks) # import + add shocks     
                    self.iexo[ind,t+1] = iexo_next
                    self.iexos[ind,t+1] = iexo_next
            
    
    def statenext(self,t):
        
        
        
        
        for ipol in range(self.npol):
            for ist,sname in enumerate(self.state_names):
                is_state_any_pol = (self.state[:,t]==ist)
                is_pol = (self.policy_ind[:,t+1]==ipol)
                is_state = (is_state_any_pol) & (is_pol) 
                
                if self.verbose: print('At t = {} count of {} is {}'.format(t,sname,np.sum(is_state)))
                
                if not np.any(is_state):
                    continue
                
                ind = np.where(is_state)[0]
                
                nind = ind.size
                
                
                
                #if sname == self.single_state:
                
                def single():
                    # !!! You can either simulate males or females but not both
                    
                    ss = self.single_state
                    
                    # meet a partner
                    
                    pmeet = self.Mlist[ipol].setup.pars['pmeet_t'][t] # TODO: check timing
                    
                    
                    matches = self.Mlist[ipol].decisions[t][ss]
                    
                    ia = self.iassets[ind,t+1] # note that timing is slightly inconsistent  
                    
                    # we use iexo from t and savings from t+1
                    # TODO: fix the seed
                    #TODO having iexo in t inconsistent with couple ones, which lool t+1
                    iznow = self.iexo[ind,t]
                    
                    pmat = matches['p'][ia,iznow,:]
                    pmat_cum = pmat.cumsum(axis=1)
                    
                    
                    v = self.shocks_single_iexo2[ind,t] #np.random.random_sample(ind.size) # draw uniform dist
                    #This guy below (unitl it_out) account for 50% of single timem
                    pmat_cum[:,-1]=1.0
                    i_pmat = (v[:,None] > pmat_cum).sum(axis=1)  # index of the position in pmat
                    
                    ic_out = matches['iexo'][ia,iznow,i_pmat]
                    ia_out = matches['ia'][ia,iznow,i_pmat]
                    it_out = matches['theta'][ia,iznow,i_pmat]
                    
                    # potential assets position of couple
                    
                    iall, izf, izm, ipsi = self.Mlist[ipol].setup.all_indices(t,ic_out)
                    
                    iz = izf if self.female else izm
                    
                    
                    # compute for everyone
                    
                    
                    vmeet = self.shocks_single_meet[ind,t]
                    i_nomeet =  np.array( vmeet > pmeet )
                    
                    
                    #those two below account for 20% of the time
                    i_pot_agree = matches['Decision'][ia,iznow,i_pmat]
                    i_m_preferred = matches['M or C'][ia,iznow,i_pmat]
                    
                    i_disagree = (~i_pot_agree)
                    i_disagree_or_nomeet = (i_disagree) | (i_nomeet)
                    i_disagree_and_meet = (i_disagree) & ~(i_nomeet)
                    
                    i_agree = ~i_disagree_or_nomeet
    
                    
                    i_agree_mar = (i_agree) & (i_m_preferred)
                    i_agree_coh = (i_agree) & (~i_m_preferred)
                    
                    assert np.all(~i_nomeet[i_agree])
                    
                    
                    nmar, ncoh, ndis, nnom = np.sum(i_agree_mar),np.sum(i_agree_coh),np.sum(i_disagree_and_meet),np.sum(i_nomeet)
                    ntot = sum((nmar, ncoh, ndis, nnom))
                    
                    if self.verbose: print('{} mar, {} coh,  {} disagreed, {} did not meet ({} total)'.format(nmar,ncoh,ndis,nnom,ntot))
                    #assert np.all(ismar==(i_agree )
                    
                    if np.any(i_agree_mar):
                        
                        self.itheta[ind[i_agree_mar],t+1] = it_out[i_agree_mar]
                        self.iexo[ind[i_agree_mar],t+1] = iall[i_agree_mar]#*0+199
                        self.iexos[ind[i_agree_mar],t+1] = iall[i_agree_mar]#*0+199
                        self.state[ind[i_agree_mar],t+1] = self.state_codes['Couple, M']
                        self.iassets[ind[i_agree_mar],t+1] = ia_out[i_agree_mar]
                        
                        # FLS decision
                        #self.ils_i[ind[i_ren],t+1] = 
                        tg = self.Mlist[ipol].setup.v_thetagrid_fine                    
                        fls_policy = self.Mlist[ipol].decisions[t+1]['Couple, M']['fls']
                        
                        self.ils_i[ind[i_agree_mar],t+1] = \
                            fls_policy[self.iassets[ind[i_agree_mar],t+1],self.iexo[ind[i_agree_mar],t+1],self.itheta[ind[i_agree_mar],t+1]]
                        
                        
                    if np.any(i_agree_coh):
                        
                        self.itheta[ind[i_agree_coh],t+1] = it_out[i_agree_coh]#*0+60
                        self.iexo[ind[i_agree_coh],t+1] = iall[i_agree_coh]#*0+199
                        self.iexos[ind[i_agree_coh],t+1] = iall[i_agree_coh]#*0+199
                        self.state[ind[i_agree_coh],t+1] = self.state_codes['Couple, C']
                        self.iassets[ind[i_agree_coh],t+1] = ia_out[i_agree_coh]
                        
                        # FLS decision
                        tg = self.Mlist[ipol].setup.v_thetagrid_fine
                        #fls_policy = self.V[t+1]['Couple, C']['fls']
                        fls_policy = self.Mlist[ipol].decisions[t+1]['Couple, C']['fls']
                        
                        self.ils_i[ind[i_agree_coh],t+1] = \
                            fls_policy[self.iassets[ind[i_agree_coh],t+1],self.iexo[ind[i_agree_coh],t+1],self.itheta[ind[i_agree_coh],t+1]]
                        
                    
                        
                    if np.any(i_disagree_or_nomeet):
                        # do not touch assets
                        self.iexo[ind[i_disagree_or_nomeet],t+1]  = iz[i_disagree_or_nomeet]
                        self.state[ind[i_disagree_or_nomeet],t+1] = self.state_codes[ss]
                        self.ils_i[ind[i_disagree_or_nomeet],t+1] = self.ils_def
                        
                        
                #elif sname == "Couple, M" or sname == "Couple, C":
                
                def couples():
                    
                    ss = self.single_state
                    
                    decision = self.Mlist[ipol].decisions[t][sname]
    
                    
                    # by default keep the same theta and weights
                    
                    self.itheta[ind,t+1] = self.itheta[ind,t]
                    
                    nt = self.Mlist[ipol].setup.ntheta_fine
                    
                    
                    # initiate renegotiation
                    isc = self.iassets[ind,t+1]
                    iall, izf, izm, ipsi = self.Mlist[ipol].setup.all_indices(t+1,self.iexo[ind,t+1])
                    
                    iz = izf if self.female else izm
                    
                    itht = self.itheta[ind,t+1] 
                    agrid =  self.Mlist[ipol].setup.agrid_c  
                    agrids =  self.Mlist[ipol].setup.agrid_s
                    sc = agrid[isc] # needed only for dividing asssets               
                    
                    thts_all = decision['thetas']
                    thts_orig_all = np.broadcast_to(np.arange(nt)[None,None,:],thts_all.shape)
                    
                    
                    thts = thts_all[isc,iall,itht]
                    thts_orig = thts_orig_all[isc,iall,itht]#this line below takes 43% of the time in coupls
                    
                    dec = decision['Decision']
                    #this guy below account for 24% of the time in couple
                    i_stay = dec[isc,iall] if dec.ndim==2 else dec[isc,iall,itht]#i_stay = dec[isc,iall,itht] 
                    
                    bil_bribing = ('Bribing' in decision)
                    
                    
                    i_div = ~i_stay    
                    
                    #ifem=decision['Divorce'][0][isc,iall][...,None]<self.Mlist[ipol].V[t]['Couple, M']['VF'][isc,iall,:]
                    #imal=decision['Divorce'][1][isc,iall][...,None]<self.Mlist[ipol].V[t]['Couple, M']['VM'][isc,iall,:]
                    #both=~np.max((ifem) & (imal),axis=1)
    
                    i_ren = (i_stay) & (thts_orig != thts)
                    i_renf = (i_stay) & (thts_orig > thts)
                    i_renm = (i_stay) & (thts_orig < thts)
                    i_sq = (i_stay) & (thts_orig == thts)
                        
                    
                    if self.verbose: print('{} divorce, {} ren-f, {} ren-m, {} sq'.format(np.sum(i_div),np.sum(i_renf),np.sum(i_renm),np.sum(i_sq))                     )
                    
                    
                    
                    zf_grid = self.setup.exo_grids['Female, single'][t]
                    zm_grid = self.setup.exo_grids['Male, single'][t]
                    
                    
                    
                    
                    if np.any(i_div):
                        
                        income_fem = np.exp(zf_grid[izf[i_div]]+self.setup.pars['f_wage_trend'][t])
                        income_mal = np.exp(zm_grid[izm[i_div]]+self.setup.pars['m_wage_trend'][t])
                        
                        income_share_fem = (income_fem) / (income_fem + income_mal)
                        
                        # this part determines assets
                        costs = self.Mlist[ipol].setup.div_costs if sname == 'Couple, M' else self.Mlist[ipol].setup.sep_costs
                                   
                        share_f, share_m = costs.shares_if_split(income_share_fem)
                        
                        #Uncomment bnelowe if ren_theta
                        share_f = costs.shares_if_split_theta(self.setup,self.setup.thetagrid[self.setup.v_thetagrid_fine.i[itht]+1])[i_div]
                        share_m=1-share_f
                      
                        #sf = share_f[i_div]*sc[i_div]
                        #assert np.all(share_f[i_div]>=0) and np.all(share_f[i_div]<=1)
                        #sm = share_m[i_div]*sc[i_div]
                        
                        sf = share_f*sc[i_div]
                        assert np.all(share_f>=0) and np.all(share_f<=1)
                        sm = share_m*sc[i_div]
                        
                        s = sf if self.female else sm
                        shks = 1-self.shocks_div_a[ind[i_div],t]

                        # if bribing happens we overwrite this
                        self.iassets[ind[i_div],t+1] = VecOnGrid(self.Mlist[ipol].setup.agrid_s,s).roll(shocks=shks)
                        
                        
                        if bil_bribing:
                            
                            iassets = decision['Bribing'][1] if self.female else decision['Bribing'][2] 
                            do_bribing = decision['Bribing'][0]
                            
                            iassets_ifdiv = iassets[isc[i_div],iall[i_div],itht[i_div]] # assets resulted from bribing
                            do_b = do_bribing[isc[i_div],iall[i_div],itht[i_div]] # True / False if bribing happens
                            assert np.all(iassets_ifdiv[do_b] >= 0)
                            
                            if np.any(do_b):
                                #n_b = np.sum(do_b)
                                #n_tot = np.sum(i_div)
                                #share_b = int(100*n_b/n_tot)
                                #print('bribing happens in {} cases, that is {}% of all divorces'.format(n_b,share_b))
                                self.iassets[ind[i_div][do_b],t+1] = iassets_ifdiv[do_b]
                                
                                #print(np.mean(agrid[isc[i_div][do_b]]/(agrids[decision['Bribing'][1][isc[i_div][do_b],iall[i_div][do_b],itht[i_div][do_b]]]+
                                 #                                      agrids[decision['Bribing'][2][isc[i_div][do_b],iall[i_div][do_b],itht[i_div][do_b]]])))
                                
                                #aaa=self.Mlist[ipol].setup.agrid_c[self.iassets[ind[i_div][do_b],t+1]]/(self.Mlist[ipol].setup.agrid_c[self.iassetss[ind[i_div][do_b],t+1]])
                                #aaa1=(self.Mlist[ipol].setup.agrid_c[self.iassetss[ind[i_div][do_b],t+1]]>0) 
                                #if sname == "Couple, M":print(np.mean(aaa[aaa1]))
                         
                                
                        
                        self.itheta[ind[i_div],t+1] = -1
                        self.iexo[ind[i_div],t+1] = iz[i_div]
                        self.state[ind[i_div],t+1] = self.state_codes[ss]
                        if sname == "Couple, M":self.divorces[ind[i_div],t+1]=True
                       

                        #FLS
                        self.ils_i[ind[i_div],t+1] = self.ils_def
                        
                    if np.any(i_ren):
                        
                        self.itheta[ind[i_ren],t+1] = thts[i_ren]
                        
                        
                        #tg = self.setup.v_thetagrid_fine
                        
                        #Distinguish between marriage and cohabitation
                        if sname == "Couple, M":
                            self.state[ind[i_ren],t+1] = self.state_codes[sname]
                            
                            
                            ipick = (self.iassets[ind[i_ren],t+1],self.iexo[ind[i_ren],t+1],self.itheta[ind[i_ren],t+1])
                            self.ils_i[ind[i_ren],t+1] = self.Mlist[ipol].decisions[t+1][sname]['fls'][ipick]
                        else:
                            i_coh = decision['Cohabitation preferred to Marriage'][isc,iall,thts]
                            i_coh1=i_coh[i_ren]
                            
                            ipick = (self.iassets[ind[i_ren],t+1],self.iexo[ind[i_ren],t+1],self.itheta[ind[i_ren],t+1])
                            ils_if_mar = self.Mlist[ipol].decisions[t+1]["Couple, M"]['fls'][ipick]
                            ils_if_coh = self.Mlist[ipol].decisions[t+1]["Couple, C"]['fls'][ipick]
                            
                            self.ils_i[ind[i_ren],t+1] = i_coh1*ils_if_coh+(1-i_coh1)*ils_if_mar
                            self.state[ind[i_ren],t+1] = i_coh1*self.state_codes["Couple, C"]+(1-i_coh1)*self.state_codes["Couple, M"]
                          
                                
                            
                        
                    if np.any(i_sq):
                        self.state[ind[i_sq],t+1] = self.state_codes[sname]
                        # do not touch theta as already updated
                        
                        #Distinguish between marriage and cohabitation
                        if sname == "Couple, M":
                            self.state[ind[i_sq],t+1] = self.state_codes[sname]
                            
                            ipick = (self.iassets[ind[i_sq],t+1],self.iexo[ind[i_sq],t+1],self.itheta[ind[i_sq],t+1])
                            self.ils_i[ind[i_sq],t+1] = self.Mlist[ipol].decisions[t+1][sname]['fls'][ipick]
                        else:
                            i_coh = decision['Cohabitation preferred to Marriage'][isc,iall,thts]
                            i_coh1=i_coh[i_sq]
                            self.state[ind[i_sq],t+1] = i_coh1*self.state_codes["Couple, C"]+(1-i_coh1)*self.state_codes["Couple, M"]
                            
                            ipick = (self.iassets[ind[i_sq],t+1],self.iexo[ind[i_sq],t+1],self.itheta[ind[i_sq],t+1])
                            
                            ils_if_mar = self.Mlist[ipol].decisions[t+1]["Couple, M"]['fls'][ipick]
                            ils_if_coh = self.Mlist[ipol].decisions[t+1]["Couple, C"]['fls'][ipick]
                           
                            self.ils_i[ind[i_sq],t+1] = i_coh1*ils_if_coh+(1-i_coh1)*ils_if_mar
                            self.state[ind[i_sq],t+1] = i_coh1*self.state_codes["Couple, C"]+(1-i_coh1)*self.state_codes["Couple, M"]
                
                
                if sname == self.single_state:
                    
                    single()
                    
                    
                elif sname == "Couple, M" or sname == "Couple, C":
                    
                    couples()
                    
                else:
                    raise Exception('unsupported state?')
        
        assert not np.any(np.isnan(self.state[:,t+1]))        
        
        
class AgentsPooled:
    def __init__(self,AgentsList):
        
        def combine(mlist):
            return np.concatenate(mlist,axis=0)
        
        self.state = combine([a.state for a in AgentsList])
        self.iexo = combine([a.iexo for a in AgentsList])
        self.iexos = combine([a.iexos for a in AgentsList])
        self.ils_i = combine([a.ils_i for a in AgentsList])
        self.itheta = combine([a.itheta for a in AgentsList])
        self.iassets = combine([a.iassets for a in AgentsList])
        self.iassetss = combine([a.iassetss for a in AgentsList])
        self.divorces = combine([a.divorces for a in AgentsList])
        self.c = combine([a.c for a in AgentsList])
        self.s = combine([a.s for a in AgentsList])
        self.x = combine([a.x for a in AgentsList])
        self.shocks_single_iexo=combine([a.shocks_single_iexo for a in AgentsList])
        self.shocks_couple_a=combine([a.shocks_couple_a for a in AgentsList])
        self.shocks_single_a=combine([a.shocks_single_a for a in AgentsList])
        self.policy_ind = combine([a.policy_ind for a in AgentsList])
        self.agents_ind = combine([i*np.ones_like(a.state) for i, a in enumerate(AgentsList)])
        self.is_female = combine([a.female*np.ones_like(a.state) for a in AgentsList])
        self.T = AgentsList[0].T
        self.N = sum([a.N for a in AgentsList])
        
    def sample(self):
        pass