#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:58:43 2019

@author: Egor
"""


# this defines model residuals
import numpy as np
import pickle, dill
import os
import psutil
import gc
gc.disable()
#import cProfile






# return format is any combination of 'distance', 'all_residuals' and 'models'
# we can add more things too for convenience

def mdl_resid(x=None,save_to=None,load_from=None,return_format=['distance'],
              solve_transition=False,
              store_path = None,
              verbose=False,calibration_report=False,draw=False,graphs=False,
              rel_diff=True,welf=False):
   
    from model import Model
    from setup import DivorceCosts
    from simulations import Agents, AgentsPooled
    from moments import moment
    
  
    
    if type(x) is dict:
        params = x
    else:
        from calibration_params import calibration_params     
        lb, ub, x0, keys, translator = calibration_params()        
        if verbose: print('Calibration adjusts {}'.format(keys))        
        if x is None: x = x0
        params = translator(x) # this converts x to setup parameters according to what is in calibration_params
    
    
    
    
    
   
    try:
        ulost = params.pop('ulost') # ulost does not belong to setup parameters so this pop removes it
    except:
        ulost = 0.0
        
    try:
        alost = params.pop('alost')
    except:
        alost = 0.0
    
    # this is for the default model
    dc = DivorceCosts(unilateral_divorce=True,assets_kept = 1.0,u_lost_m=ulost,u_lost_f=ulost,eq_split=1.0)
    sc = DivorceCosts(unilateral_divorce=True,assets_kept = 1.0,u_lost_m=0.00,u_lost_f=0.00,eq_split=0.0)
    
    
    
    
    iter_name = 'default' if not verbose else 'default-timed'
    
    
    def join_path(name,path):
        return os.path.join(path,name)
    
    
    
    
    if load_from is not None:
        if type(load_from) is not list:
            load_from = [load_from]
        if store_path is not None:
            load_from = [join_path(n,store_path) for n in load_from]
    
   
    if save_to is not None:
        if type(save_to) is not list:
            save_to = [save_to]
        if store_path is not None:
            save_to = [join_path(n,store_path) for n in save_to]
    
    
                
    

    if load_from is None:
        
        
        
        if not solve_transition:
            
            mdl = Model(iterator_name=iter_name,divorce_costs=dc,
                        separation_costs=sc,draw=draw,**params)
            mdl_list = [mdl]
            
        else:
            # specify the changes here manually
            dc_before = DivorceCosts(unilateral_divorce=False,assets_kept = 1.0,u_lost_m=ulost,u_lost_f=ulost,eq_split=1.0)
            dc_after  = DivorceCosts(unilateral_divorce=True,assets_kept = 1.0,u_lost_m=ulost,u_lost_f=ulost,eq_split=1.0)
            
            
            
            mdl_before = Model(iterator_name=iter_name,divorce_costs=dc_before,
                        separation_costs=sc,draw=draw,**params)
            
            mdl_after = Model(iterator_name=iter_name,divorce_costs=dc_after,
                        separation_costs=sc,draw=draw,**params)  
            
            mdl = mdl_before # !!! check if this makes a difference
            # I think that it is not used for anything other than getting 
            # setup for plotting
            
            mdl_list = [mdl_before,mdl_after]
            
    else:       
        mdl_list = [dill.load(open(l,'rb+')) for l in load_from]
        mdl = mdl_list[0]
        
        if solve_transition:
            if len(mdl_list) < 2:
                print('Warning: you supplied only one model, so no transition is computed')
  
    if save_to is not None:
        
        if not solve_transition:
            if len(save_to) > 1:
                print('warning: too much stuff is save_to')
            dill.dump(mdl,open(save_to[0],'wb+'))            
            
        else:            
            if len(save_to) > 1:
                [dill.dump(m_i,open(n_i,'wb+')) 
                    for (m_i,n_i) in zip(mdl_list,save_to)]
            else:
                print('Warning: file names have change to write two models, \
                      please provide the list of names if you do not want this')
                dill.dump(mdl_before,open(save_to[0] + '_before','wb+'))
                dill.dump(mdl_after, open(save_to[0] + '_after','wb+'))
                
    
    ##############################################################
    # Build Markov transition processes for models from the data
    #############################################################
    
    #Import Data
    with open('age_uni.pkl', 'rb') as file:
        age_uni=pickle.load(file)
        

    def get_transition(age_dist,welf=False):
        #Transformation of age at uni from actual age to model periods
        change=-np.ones(1000,np.int32)#the bigger is the size of this array, the more precise the final distribution
       
        summa=0.0
        summa1=0.0
        for i in age_dist:
           
            summa+=age_dist[i]
            change[int(summa1*len(change[:])/sum(age_dist.values())):int(summa*len(change[:])/sum(age_dist.values()))]=(i-18)/mdl.setup.pars['py']
            summa1+=age_dist[i]
        change=np.sort(change, axis=0) 
        
        #Now we compute the actual conditional probabilities
        transition_matricest=list()
        
        #First period treated differently
        pr=np.sum(change<=0)/(np.sum(change<=np.inf))
        transition_matricest=transition_matricest+[np.array([[1-pr,pr],[0,1]])]
        for t in range(mdl.setup.pars['T']-1):
            if not welf:
                pr=np.sum(change==t+1)/(np.sum(change<=np.inf))
                transition_matricest=transition_matricest+[np.array([[1-pr,pr],[0,1]])]
            else:
                transition_matricest=transition_matricest+[np.array([[1,0],[1,0]])]
                
            
            
        return transition_matricest
    
   
    
    transition_matricesf=get_transition(age_uni['female'],welf)
    transition_matricesm=get_transition(age_uni['male'],welf)

    
    
        
    #Get Data Moments
    with open('freq.pkl', 'rb') as file:
        freq=pickle.load(file)
   
    #Get Number of simulated agent, malea and female
    N=15000
    Nf=int(N*freq['fem'])#age_uni['share_female'])
    Nm=N-Nf
    Nme=int(Nm*freq['Nme'])
    Nmn=Nm-int(Nm*mdl.setup.pars['Nme'])
    Nfe=int(Nf*freq['Nfe'])
    Nfn=Nf-int(Nf*mdl.setup.pars['Nfe'])
   
    agents_feme = Agents( mdl_list ,age_uni['female'],female=True,edu='e',pswitchlist=None,verbose=False,N=Nfe,draw=draw)
   
    agents_male = Agents( mdl_list ,age_uni['male'],female=False,edu='e',pswitchlist=None,verbose=False,N=Nme,draw=draw)
   
    agents_femn = Agents( mdl_list ,age_uni['female'],female=True,edu='n',pswitchlist=None,verbose=False,N=Nfn,draw=draw)
   
    agents_maln = Agents( mdl_list ,age_uni['male'],female=False,edu='n',pswitchlist=None,verbose=False,N=Nmn,draw=draw)
   
    agents_pooled = AgentsPooled([agents_feme,agents_male,agents_femn,agents_maln])#AgentsPooled([agents_femn,agents_maln])#
    
    
    #Compute moments
   
    moments = moment(mdl_list,agents_pooled,agents_male,draw=draw)
    
    
    ############################################################
    #Build data moments and compare them with simulated ones
    ###########################################################
    
    
    #Get Data Moments
    with open('moments.pkl', 'rb') as file:
        packed_data=pickle.load(file)
        
    #Unpack Moments (see data_moments.py to check if changes)
    #(hazm,hazs,hazd,mar,coh,fls_ratio,W)
    hazs_d=packed_data['hazs']   
    hazm_d=packed_data['hazm']   
    hazd_d=packed_data['hazd']   
    everc_d=packed_data['everc']   
    everm_d=packed_data['everm']
    everr_e_d=packed_data['everr_e']
    everr_ne_d=packed_data['everr_ne']
    flsc_d=packed_data['flsc']
    flsm_d=packed_data['flsm']
    beta_edu_d=packed_data['beta_edu']
    ref_coh_d=packed_data['ref_coh']
    ratio_mar_d=packed_data['ratio_mar']
    W=packed_data['W']
    dat=np.concatenate((hazm_d,hazs_d,hazd_d,everc_d,everm_d,flsc_d,flsm_d,beta_edu_d*np.ones(1),ref_coh_d,ratio_mar_d*np.ones(1)),axis=0)
    

    #Get Simulated Data

    hazs_s=moments['hazard sep'][0:len(hazm_d)]   
    hazm_s=moments['hazard mar'][0:len(hazs_d)]   
    hazd_s=moments['hazard div'][0:len(hazd_d)]   
    everc_s=moments['everc'][0:len(everc_d)]
    everm_s=moments['everm'][0:len(everm_d)]
    everr_e_s=moments['everr_e'][0:len(everr_e_d)]
    everr_ne_s=moments['everr_ne'][0:len(everr_ne_d)]
    flsc_s=moments['flsc']
    flsm_s=moments['flsm']
    beta_edu_s=moments['beta_edu']
    ref_coh_s=moments['ref_coh']  
    ratio_mar_s=moments['ratio_mar'] 
    sim=np.concatenate((hazm_s,hazs_s,hazd_s,everc_s,everm_s,flsc_s,flsm_s,beta_edu_s*np.ones(1),ref_coh_s,ratio_mar_s*np.ones(1)),axis=0)



    if len(dat) != len(sim):
        sim = np.full_like(dat,1.0e6)
        
   
    res_all=(dat-sim)
  
 
    
    if verbose:
        print('data moments are {}'.format(dat))
        print('simulated moments are {}'.format(sim))
    
    resid_all = np.array([x if (not np.isnan(x) and not np.isinf(x)) else 1e6 for x in res_all])
    
    
    
    resid_sc = resid_all*np.sqrt(np.diag(W)) # all residuals scaled
    
    dist = np.dot(np.dot(resid_all,W),resid_all)


    print('Distance is {}'.format(dist))
    
    
    
   
    
    
    
    out_dict = {'distance':dist,'all residuals':resid_all,
                'scaled residuals':resid_sc,'models':mdl_list,'agents':agents_pooled}
    out = [out_dict[key] for key in return_format]
    
    
    
    del(out_dict)
    
    # memory management
    if ('models' not in return_format) | (draw==False):
        for m in mdl_list:
            del(m)
        del mdl_list
        
    if ('agents' not in return_format) | (draw==False):
        del(agents_pooled,agents_feme,agents_male,agents_femn,agents_maln)
        
    gc.collect()
  
            
    return out
