# -*- coding: utf-8 -*-    
"""    
Created on Thu Nov 14 10:26:49 2019    
     
This file comupte simulated moments + optionally    
plots some graphs    
     
@author: Fabio    
"""    
     
import numpy as np    
import matplotlib.pyplot as plt    
import matplotlib.backends.backend_pdf  
import statsmodels.formula.api as smf    

  
#Avoid error for having too many figures 
plt.rcParams.update({'figure.max_open_warning': 0}) 
from statutils import strata_sample 
from welfare_comp import welf_dec 
  
#For nice graphs with matplotlib do the following  
matplotlib.use("pgf")  
matplotlib.rcParams.update({  
    "pgf.texsystem": "pdflatex",  
    'font.family': 'serif',  
    'font.size' : 11,  
    'text.usetex': True,  
    'pgf.rcfonts': False, 
})  
  
  
import pickle    
import pandas as pd    

   
     
def moment(mdl_list,agents,agents_male,draw=True,validation=False):    
#This function compute moments coming from the simulation    
#Optionally it can also plot graphs about them. It is feeded with    
#matrixes coming from simulations    
     
 
 
    mdl=mdl_list[0] 
 
    
    #Import simulated values   
    state_true=agents.state  
    educ=np.reshape(agents.education.ravel(),state_true.shape)
    edup=np.reshape(agents.partneredu.ravel(),state_true.shape)
    
    #Modify state names
    state=state_true.copy()
    state[(state_true==0) | (state_true==2)]=0
    state[(state_true==1) | (state_true==3)]=1
    state[(state_true==4) | (state_true==5)| (state_true==6 )| (state_true==7)]=2
    state[(state_true==8) | (state_true==9)| (state_true==10 )| (state_true==11)]=3
    
    assets_t=mdl.setup.agrid_c[agents.iassets] 
    assets_t[agents.state<=1]=mdl.setup.agrid_s[agents.iassets[agents.state<=1]] 
    iexo=agents.iexo      
    theta_t=mdl.setup.thetagrid_fine[agents.itheta]    
    setup = mdl.setup   
    female=agents.is_female 
    labor=agents.ils_i 
    durf=agents.duf 
    psi_check=np.zeros(state.shape) 
    psim=-1000*np.ones(state.shape) 
    single=np.array((state<=1),dtype=bool) 
   
     
    #Fill psi and ushift here 
    for i in range(len(state[0,:])): 
        for dd in range(setup.pars['dm']):
            
            pos=(agents.ipsim[:,i]>-500)
            psi_check[:,i]=((setup.exogrid.psi_t[dd][i][(setup.all_indices(i,iexo[:,i]))[3]])) 
            if np.any(pos):psim[:,i][pos]=((setup.exogrid.psi_t[0][i][(setup.all_indices(i,agents.ipsim[:,i][pos]))[3]])) 
     
   
    psi_check[:,0]=psim[:,1]  
    psi_check[single]=0.0  
    state_psid=agents_male.state  
    labor_psid=agents_male.ils_i  
    iexo_psid=agents_male.iexo  
    change_psid=agents_male.policy_ind 
  
    if draw: 
        #Import values for female labor supply (simulated men only) 

       
        labor_w=agents.ils_i 
        labor_w=labor_w[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']]
        divorces_w=agents.divorces 
        theta_w=mdl.setup.thetagrid_fine[agents.itheta]  
        assets_w=mdl.setup.agrid_c[agents.iassets] 
        assets_w[agents.state<=1]=mdl.setup.agrid_s[agents.iassets[agents.state<=1]] 
        assetss_w=mdl.setup.agrid_c[agents.iassetss] 
        assetss_w[agents.state<=1]=mdl.setup.agrid_s[agents.iassetss[agents.state<=1]] 
        changep_w=agents.policy_ind  
 
    moments=dict() 
         
    ########################################## 
    #WELFARE DECOMPOSITION HERE 
    ######################################### 
    #if draw:
      #  if len(mdl_list) > 1: 
       #     welf_dec(mdl_list,agents) 
         
    ##########################################    
    #START COMPUTATION OF SIMULATED MOMENTS    
    #########################################    
       
         
    #Create a file with the age of the change foreach person    
    changep=agents.policy_ind   
        
         
    #Get states codes    
    state_codes = {'Female, single': 0, 'Male, single': 1, 'Couple, M': 2,'Couple, C': 3} 
    state_codes_full = {name: i for i, name in enumerate(mdl.setup.state_names)} 
    
     ###########################################    
    #Sample selection - get age at censoring   
    ###########################################    
    with open('freq.pkl', 'rb') as file:     
        freq_raw=pickle.load(file)     
             
    freq=freq_raw['freq_c']
    aged=np.ones((state.shape))    
        
     
    summa=0.0    
    summa1=0.0    
    for i in freq:    
        summa+=freq[int(i)]    
        aged[int(summa1*len(aged[:])/sum(freq.values())):int(summa*len(aged[:])/sum(freq.values()))]=round((i-20)/mdl.setup.pars['py'],0)    
        summa1+=freq[int(i)]    
        
    aged=np.array(aged,dtype=np.int16)     

   
    ###########################################    
    #Moments: Construction of Spells    
    ###########################################    
    nspells = (state[:,1:]!=state[:,:-1]).astype(np.int).sum(axis=1).max() + 1   
    index=np.array(np.linspace(1,len(state[:,0]),len(state[:,0]))-1,dtype=np.int16)   
    N=len(iexo[:,0])   
    state_beg = -1*np.ones((N,nspells),dtype=np.int8)    
    time_beg = -1*np.ones((N,nspells),dtype=np.bool)    
    did_end = np.zeros((N,nspells),dtype=np.bool)    
    state_end = -1*np.ones((N,nspells),dtype=np.int8)       
    sp_length = -1*np.ones((N,nspells),dtype=np.int16)    
    sp_dur = -1*np.ones((N,nspells),dtype=np.int16)    
    is_unid = -1*np.ones((N,nspells),dtype=np.int16)    
    is_unid_end = -1*np.ones((N,nspells),dtype=np.int16)    
    n_spell = -1*np.ones((N,nspells),dtype=np.int16)    
    is_spell = np.zeros((N,nspells),dtype=np.bool)    
    sp_edu=np.ones((N,nspells),dtype='<U1') 
    sp_edup=np.ones((N,nspells),dtype='<U1')
       
         
    state_beg[:,0] = 0 # THIS ASSUMES EVERYONE STARTS AS SINGLE   #TODO consistent with men stuff? 
    time_beg[:,0] = 0    
    sp_length[:,0] = 1    
    is_spell[:,0] = True    
    ispell = np.zeros((N,),dtype=np.int8)    
         
    for t in range(1,mdl.setup.pars['T']):    
        ichange = ((state[:,t-1] != state[:,t]))    
        ifinish=(~ichange) & (t==mdl.setup.pars['T']-1) 
        sp_length[((~ichange)),ispell[((~ichange))]] += 1    
        sp_dur[ifinish,ispell[ifinish]] = durf[ifinish,t] 
        #ichange = ((state[:,t-1] != state[:,t]) & (t<=aged[:,t]))    
        #sp_length[((~ichange) & (t<=aged[:,t])),ispell[((~ichange) & (t<=aged[:,t]))]] += 1    
             
        if not np.any(ichange): continue    
             
        did_end[ichange,ispell[ichange]] = True    
             
        is_spell[ichange,ispell[ichange]+1] = True    
        sp_length[ichange,ispell[ichange]+1] = 1 # if change then 1 year right    
        state_end[ichange,ispell[ichange]] = state[ichange,t]    
        sp_dur[ichange,ispell[ichange]] = durf[ichange,t-1] 
        state_beg[ichange,ispell[ichange]+1] = state[ichange,t] 
        sp_edu[ichange,ispell[ichange]+1] = educ[ichange,t] 
        sp_edup[ichange,ispell[ichange]+1] = edup[ichange,t] 
        time_beg[ichange,ispell[ichange]+1] = t    
        n_spell[ichange,ispell[ichange]+1]=ispell[ichange]+1   
        is_unid[ichange,ispell[ichange]+1]=changep[ichange,t]   
        is_unid_end[ichange,ispell[ichange]]=changep[ichange,t-1]   
           
             
        ispell[ichange] = ispell[ichange]+1    
             
             
    allspells_beg = state_beg[is_spell]    
    allspells_len = sp_length[is_spell]    
    allspells_end = state_end[is_spell] # may be -1 if not ended    
    allspells_timeb = time_beg[is_spell]   
    allspells_isunid=is_unid[is_spell]    
    allspells_du=sp_dur[is_spell]   
    allspells_nspells=n_spell[is_spell]  
    allspells_edu=sp_edu[is_spell]
    allspells_edup=sp_edup[is_spell]
    

         
    # If the spell did not end mark it as ended with the state at its start    
    allspells_end[allspells_end==-1] = allspells_beg[allspells_end==-1]      
    allspells_nspells[allspells_nspells==-1]=0   
    allspells_nspells=allspells_nspells+1  
    allspells_prec=allspells_du+1-allspells_len
        
    #Modify education to be 0/1
    wedu=(allspells_edu=='e')
    educ_rels=np.zeros(allspells_edu.shape)
    educ_rels[wedu]=1.0
    
    wedu=(allspells_edup=='e')
    educ_relsp=np.zeros(allspells_edup.shape)
    educ_relsp[wedu]=1.0
    
    #Use this to construct hazards   
    spells = np.stack((allspells_beg,allspells_len,allspells_end,educ_rels),axis=1)    
       
    #Use this for empirical analysis   
    spells_empiricalt=np.stack((allspells_beg,allspells_timeb,allspells_len,
                                allspells_end,allspells_nspells,allspells_isunid,
                                allspells_prec,educ_rels,educ_relsp),axis=1)   

         
    #Now divide spells by relationship nature    
    all_spells=dict()    
    for ist,sname in enumerate(state_codes):    
    
        is_state= (spells[:,0]==ist)    
        all_spells[sname]=spells[is_state,:]      
        is_state= (all_spells[sname][:,1]!=0)    
        all_spells[sname]=all_spells[sname][is_state,:]    
            
  
        ############################################     
    #Construct sample of first relationships     
    ############################################     
    
         
    #Now define variables     
    rel_end = -1*np.ones((N,99),dtype=np.int16)     
    rel_age= -1*np.ones((N,99),dtype=np.int16)     
    rel_unid= -1*np.ones((N,99),dtype=np.int16)     
    rel_number= -1*np.ones((N,99),dtype=np.int16)    
    rel_sex=-1*np.ones((N,99),dtype=np.int16)  
    rel_edu=np.ones((N,99),dtype='<U1') 
    rel_edup=np.ones((N,99),dtype='<U1') 
    isrel = np.zeros((N,),dtype=np.int8)     
         
    for t in range(2,mdl.setup.pars['Tret']-int(6/mdl.setup.pars['py'])):     
             
        irchange = ((state[:,t-1] != state[:,t]) & ((state[:,t-1]==0) | (state[:,t-1]==1)))     
             
        if not np.any(irchange): continue     
         
        rel_end[irchange,isrel[irchange]]=state[irchange,t]     
        rel_age[irchange,isrel[irchange]]=t     
        rel_edu[irchange,isrel[irchange]]=educ[irchange,t] 
        rel_edup[irchange,isrel[irchange]]=edup[irchange,t] 
        rel_unid[irchange,isrel[irchange]]=changep[irchange,t]     
        rel_number[irchange,isrel[irchange]]=isrel[irchange]+1     
        rel_sex[irchange,isrel[irchange]]=female[irchange,0]   
             
        isrel[irchange] = isrel[irchange]+1     
         
    #Get the final Variables     
    allrel_end=rel_end[(rel_end!=-1)]     
    allrel_age=rel_age[(rel_age!=-1)]     
    allrel_uni=rel_unid[(rel_unid!=-1)]   
    allrel_sex=rel_sex[(rel_unid!=-1)]
    allrel_edu=rel_edu[(rel_unid!=-1)]
    allrel_edup=rel_edup[(rel_unid!=-1)]
    allrel_number=rel_number[(rel_number!=-1)]     
         
    #Get whetehr marraige     
    allrel_mar=np.zeros((allrel_end.shape))     
    allrel_mar[(allrel_end==2)]=1     
    
    #Modify education variables
    wedu=(allrel_edu=='e')
    educ_rel=np.zeros(allrel_edu.shape)
    educ_rel[wedu]=1.0
    
    wedu=(allrel_edup=='e')
    educ_relp=np.zeros(allrel_edup.shape)
    educ_relp[wedu]=1.0
         
    #Create a Pandas Dataframe     
    data_rel=np.array(np.stack((allrel_mar,allrel_age,allrel_uni,allrel_number,allrel_sex,educ_rel,educ_relp),axis=0).T)     
    data_rel_panda=pd.DataFrame(data=data_rel,columns=['mar','age','uni','rnumber','sex','edu','edup'])     
                        
     
         
    #Regression      
    try:     
        FE_ols = smf.ols(formula='mar ~ edu+C(age)+C(sex)', data = data_rel_panda.dropna()).fit()     
        beta_edu_s=FE_ols.params['edu']     
        beta_unid_s=0.0
    except:     
        print('No data for unilateral divorce regression...')     
        beta_unid_s=0.0  
        beta_edu_s=0.0
 
      

        
    moments['beta unid']=0.0  
       
    ###################################################   
    # Second regression for the length of cohabitation   
    ###################################################   

    try:
        
        #Build the dataset for analysis on divorce risk
        spells_empiricalm=spells_empiricalt[(spells_empiricalt[:,0]==2),1:9]   
        data_m_panda=pd.DataFrame(data=spells_empiricalm,columns=['age','duration','end','rel','uni','coh','edu','edup']) 
        
        from lifelines import CoxPHFitter   
        cph = CoxPHFitter()   
        #data_m_panda.drop(data_m_panda[data_m_panda['age']>=25].index, inplace=True)
        data_m_panda['ecoh']=0.0
        data_m_panda.loc[data_m_panda['coh']>0.0,'ecoh']=1.0   
        data_m_panda['lcoh']=np.log(data_m_panda['coh']+0.001)
        #dummy=pd.get_dummies(data_m_panda['age'])
        #data_m_panda=pd.concat([data_m_panda,dummy],axis=1)
        #data_m_panda['age2']=data_m_panda['age']**2   
        #data_m_panda['age3']=data_m_panda['age']**3   
        #data_m_panda['rel2']=data_m_panda['rel']**2   
        #data_m_panda['rel3']=data_m_panda['rel']**3   
    
           
        #Standard Cox   
        data_m_panda['endd']=1.0   
        data_m_panda.loc[data_m_panda['end']==2.0,'endd']=0.0   
        data_m_panda1=data_m_panda.drop(columns=['rel', 'uni','coh','end','rel','edup','age']) 
        cox_join=cph.fit(data_m_panda1, duration_col='duration', event_col='endd')   
        parm=cox_join.params_
        
        #Get premartial cohabitation
        Tmax=int(15/mdl.setup.pars['py'])
        
        raw_dut=np.zeros((Tmax))
        ref_dut=np.zeros((Tmax))
          
        for i in range(Tmax):
            isp=(allspells_beg==2) & (allspells_prec==i) #& (allspells_timeb<15)
            raw_dut[i]=np.mean(allspells_end[isp]!=2)
            ref_dut[i]=np.exp(np.log(i+0.001)*parm['lcoh']+parm['ecoh'])/np.exp(np.log(0.001)*parm['lcoh'])
            
        raw_dut=raw_dut/raw_dut[0]
        ref_dut[0]=1.0
        
        #Get paramter for education
        beta_div_edu_s=cox_join.hazard_ratios_['edu']
        #beta_div_edup_s=cox_join.hazard_ratios_['edup']
        moments['beta_edu']= beta_div_edu_s
        moments['ref_coh']=ref_dut[1:5]
        
    except:
        Tmax=int(15/mdl.setup.pars['py'])
        raw_dut=np.zeros((Tmax))
        ref_dut=np.zeros((Tmax))
        beta_div_edup_s=0.0
        beta_div_edu_s=0.0
        moments['beta_edu']= 0.0
        moments['ref_coh']=np.array([1.0,1.0,1.0])
        
    ###################################################   
    # Second regression for the length of cohabitation   
    ###################################################   
    if draw:
        data_coh_panda=pd.DataFrame(data=spells_empiricalt[(spells_empiricalt[:,0]==3),1:9] ,
                                                           columns=['age','duration','end','rel','uni','coh','edu','edup'])    
   
        #Regression    
        try:    
        #FE_ols = smf.ols(formula='duration ~ uni+C(age)', data = data_coh_panda.dropna()).fit()    
        #beta_dur_s=FE_ols.params['uni']    
           
            from lifelines import CoxPHFitter   

            
            cph = CoxPHFitter()   
            data_coh_panda['age2']=data_coh_panda['age']**2   
            data_coh_panda['age3']=data_coh_panda['age']**3   
            data_coh_panda['rel2']=data_coh_panda['rel']**2   
            data_coh_panda['rel3']=data_coh_panda['rel']**3   
            
            #data_coh_panda=pd.get_dummies(data_coh_panda, columns=['age'])   
               
            #Standard Cox   
            data_coh_panda['endd']=1.0   
            data_coh_panda.loc[data_coh_panda['end']==3.0,'endd']=0.0   
            data_coh_panda1=data_coh_panda.drop(['end','coh','uni','edup'], axis=1)   
            cox_join=cph.fit(data_coh_panda1, duration_col='duration', event_col='endd')   
            haz_join=cox_join.hazard_ratios_['edu']
            haz_joinp=1.0#cox_join.hazard_ratios_['edup']
               
            #Cox where risk is marriage   
            data_coh_panda['endd']=0.0   
            data_coh_panda.loc[data_coh_panda['end']==2.0,'endd']=1.0   
            data_coh_panda2=data_coh_panda.drop(['end','coh','uni','edup'], axis=1)   
            cox_mar=cph.fit(data_coh_panda2, duration_col='duration', event_col='endd')   
            haz_mar=cox_mar.hazard_ratios_['edu'] 
            haz_marp=1.0#cox_mar.hazard_ratios_['edup'] 
               
            #Cox where risk is separatio   
            data_coh_panda['endd']=0.0   
            data_coh_panda.loc[data_coh_panda['end']==0.0,'endd']=1.0   
            data_coh_panda3=data_coh_panda.drop(['end','coh','uni','edup'], axis=1)   
            cox_sep=cph.fit(data_coh_panda3, duration_col='duration', event_col='endd')   
            haz_sep=cox_sep.hazard_ratios_['edu']   
            haz_sepp=1.0#cox_sep.hazard_ratios_['edup']   
               
        except:    
            print('No data for unilateral divorce regression...')    
            haz_sep=1.0  
            haz_join=1.0  
            haz_mar=1.0  
            haz_sepp=1.0  
            haz_joinp=1.0  
            haz_marp=1.0  
        
    ##################################    
    # Construct the Hazard functions    
    #################################    
             
    #Hazard of Divorce    
    hazd=list()    
    lgh=len(all_spells['Couple, M'][:,0])    
    for t in range(mdl.setup.pars['T']):    
             
        cond=all_spells['Couple, M'][:,1]==t+1    
        temp=all_spells['Couple, M'][cond,2]    
        cond1=temp!=2    
        temp1=temp[cond1]    
        if lgh>0:    
            haz1=len(temp1)/lgh    
            lgh=lgh-len(temp)    
        else:    
            haz1=0.0    
        hazd=[haz1]+hazd    
             
    hazd.reverse()    
    hazd=np.array(hazd).T    
    
    
    #Hazard of Divorce-Educated
    where=all_spells['Couple, M'][:,-1]==1
    all_spells_e=all_spells['Couple, M'][where,:]
    hazde=list()    
    lgh=len(all_spells_e[:,0])    
    for t in range(mdl.setup.pars['T']):    
             
        cond=all_spells_e[:,1]==t+1    
        temp=all_spells_e[cond,2]    
        cond1=temp!=2    
        temp1=temp[cond1]    
        if lgh>0:    
            haz1=len(temp1)/lgh    
            lgh=lgh-len(temp)    
        else:    
            haz1=0.0    
        hazde=[haz1]+hazde    
             
    hazde.reverse()    
    hazde=np.array(hazde).T  
         
    #Hazard of Separation    
    hazs=list()    
    lgh=len(all_spells['Couple, C'][:,0])    
    for t in range(mdl.setup.pars['T']):    
             
        cond=all_spells['Couple, C'][:,1]==t+1    
        temp=all_spells['Couple, C'][cond,2]    
        cond1=(temp>=0) & (temp<=1) 
        temp1=temp[cond1]    
        if lgh>0:    
            haz1=len(temp1)/lgh    
            lgh=lgh-len(temp)    
        else:    
            haz1=0.0    
        hazs=[haz1]+hazs    
             
    hazs.reverse()    
    hazs=np.array(hazs).T   
    
     
         
    #Hazard of Marriage (Cohabitation spells)    
    hazm=list()    
    lgh=len(all_spells['Couple, C'][:,0])    
    for t in range(mdl.setup.pars['T']):    
             
        cond=all_spells['Couple, C'][:,1]==t+1    
        temp=all_spells['Couple, C'][cond,2]    
        cond1=temp==2    
        temp1=temp[cond1]    
        if lgh>0:    
            haz1=len(temp1)/lgh    
            lgh=lgh-len(temp)    
        else:    
            haz1=0.0    
        hazm=[haz1]+hazm    
             
    hazm.reverse()    
    hazm=np.array(hazm).T    
         
         
    where=all_spells['Couple, C'][:,-1]==1
    all_spells_e=all_spells['Couple, C'][where,:]
    
    
    #Hazard of Separation    
    hazse=list()    
    lgh=len(all_spells_e[:,0])    
    for t in range(mdl.setup.pars['T']):    
             
        cond=all_spells_e[:,1]==t+1    
        temp=all_spells_e[cond,2]    
        cond1=(temp>=0) & (temp<=1) 
        temp1=temp[cond1]    
        if lgh>0:    
            haz1=len(temp1)/lgh    
            lgh=lgh-len(temp)    
        else:    
            haz1=0.0    
        hazse=[haz1]+hazse    
             
    hazse.reverse()    
    hazse=np.array(hazse).T   
    
     
         
    #Hazard of Marriage (Cohabitation spells)    
    hazme=list()    
    lgh=len(all_spells_e[:,0])    
    for t in range(mdl.setup.pars['T']):    
             
        cond=all_spells_e[:,1]==t+1    
        temp=all_spells_e[cond,2]    
        cond1=temp==2    
        temp1=temp[cond1]    
        if lgh>0:    
            haz1=len(temp1)/lgh    
            lgh=lgh-len(temp)    
        else:    
            haz1=0.0    
        hazme=[haz1]+hazme    
             
    hazme.reverse()    
    hazme=np.array(hazme).T  
    

    moments['hazard sep'] = hazs    
    moments['hazard div'] = hazd    
    moments['hazard mar'] = hazm  
    moments['hazard dive'] =hazde  
        
    
     
         
    #Singles: Marriage vs. cohabitation transition    
    #spells_s=np.append(spells_Femalesingle,spells_Malesingle,axis=0)    
    spells_s =all_spells['Female, single']    
    cond=spells_s[:,2]>1    
    spells_sc=spells_s[cond,2]    
    condm=spells_sc==2    
    sharem=len(spells_sc[condm])/max(len(spells_sc),0.0001)    
       
    if draw:
        #For some graphs
        changem=np.zeros(state.shape,dtype=bool)  
        changec=np.zeros(state.shape,dtype=bool)  
        change=np.zeros(state.shape,dtype=bool)  

    
        for t in range(1,mdl.setup.pars['Tret']-1):     
                 
            irchangem = ((state[:,t]==2) & ((state[:,t-1]==0) | (state[:,t-1]==1)))   
            irchangec = ((state[:,t]==3) & ((state[:,t-1]==0) | (state[:,t-1]==1)))   
            irchange=((state[:,t]!=state[:,t-1]) & ((state[:,t-1]==0) | (state[:,t-1]==1)))   
            changem[:,t]=irchangem  
            changec[:,t]=irchangec  
            change[:,t]=irchange 
        statempsi=(state==2)
        statecpsi=(state==3)
        
        
        #Get the share of assortative mating
        assce=np.mean(edup[(educ=='e')  & (changec)]=='e')/(mdl.setup.pars['Nme']*mdl.setup.pars['Nfe'])
        assme=np.mean(edup[(educ=='e')  & (changem)]=='e')/(mdl.setup.pars['Nme']*mdl.setup.pars['Nfe'])
        asscn=np.mean(edup[(educ=='n')  & (changec)]=='n')/(mdl.setup.pars['Nmn']*mdl.setup.pars['Nfn'])
        assmn=np.mean(edup[(educ=='n')  & (changem)]=='n')/(mdl.setup.pars['Nmn']*mdl.setup.pars['Nfn'])
        
        print('A mating (e measured) cohabitations is {}, marriage is {}'.format(assce,assme) )
        print('A mating n-n (n measured) cohabitations is {}, marriage is {}'.format(asscn,assmn)) 
        
    #Cut the first two periods give new 'length'    
    lenn=mdl.setup.pars['T']-mdl.setup.pars['Tbef']    
    assets_t=assets_t[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']]    
    iexo=iexo[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']] 
    state_old=state
    state=state[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']] 
    state_true=state_true[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']]    
    labor=labor[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']]        
    female=female[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']] 
    edupp=edup[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']] 
    #psi_check=psi_check[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']]
    ###########################################    
    #Moments: FLS   
    ###########################################    
         
         
    flsm=np.ones(mdl.setup.pars['Tret'])    
    flsc=np.ones(mdl.setup.pars['Tret'])    
         
         
    for t in range(mdl.setup.pars['Tret']):    
             
        pick = agents.state[:,t]==2           
        if pick.any(): flsm[t] = np.array(setup.ls_levels)[agents.ils_i[pick,t]].mean()    
        pick = agents.state[:,t]==3    
        if pick.any(): flsc[t] = np.array(setup.ls_levels)[agents.ils_i[pick,t]].mean()    
             
 

    ################ 
    #Ratio of fls  
    ############### 
    resha=len(state[0,:])*len(state[:,0])
    agetemp=np.linspace(1,len(state[0,:]),len(state[0,:]))  
    agegridtemp=np.reshape(np.repeat(agetemp,len(state[:,0])),(len(state[:,0]),len(agetemp)),order='F')  
    agegrid=np.reshape(agegridtemp,resha) 
    
    incouple= ((state==2) | (state==3)) & (female==1)
    incoupler=np.reshape(incouple,resha)
    ages=agegrid[incoupler]
    state_par=np.reshape(state,resha)[incoupler]
    labor_par=np.reshape(labor,resha) 
    labor_par=labor_par[incoupler]
    #female_par=np.reshape(female,resha)[incoupler]
    
     
    mean_fls_m=0.0  
    picky=(state_par[:]==2) & (ages>=5) & (ages<=6) #& (female_par[:]==1)
    pickm=(state_par[:]==2) & (ages>=9) & (ages<=11) #& (female_par[:]==1)
    picko=(state_par[:]==2) & (ages>=14) & (ages<=16) #& (female_par[:]==1)
    mean_fls_m=np.zeros((3)) 
    if picky.any():mean_fls_m[0]=np.mean(labor_par[picky]*setup.ls_levels[-1])  
    if pickm.any():mean_fls_m[1]=np.mean(labor_par[pickm]*setup.ls_levels[-1])  
    if picko.any():mean_fls_m[2]=np.mean(labor_par[picko]*setup.ls_levels[-1])  
        
    mean_fls_c=0.0  
    picky=(state_par[:]==3) & (ages>=5) & (ages<=6) #& (female_par[:]==1)
    pickm=(state_par[:]==3) & (ages>=9) & (ages<=11) #& (female_par[:]==1)
    picko=(state_par[:]==3) & (ages>=14) & (ages<=16) #& (female_par[:]==1)
    mean_fls_c=np.zeros((3)) 
    if picky.any():mean_fls_c[0]=np.mean(labor_par[picky]*setup.ls_levels[-1])  
    if pickm.any():mean_fls_c[1]=np.mean(labor_par[pickm]*setup.ls_levels[-1])  
    if picko.any():mean_fls_c[2]=np.mean(labor_par[picko]*setup.ls_levels[-1])   
      
    small=mean_fls_c<0.0001*np.ones((3)) 
    mean_fls_c[small]=0.0001*np.ones((3))[small] 

    moments['flsm']=mean_fls_m
    moments['flsc']=mean_fls_c
     
     
    if draw:
        
        
        ################ 
        #Ratio of fls  
        ############### 
      
        full_flsm={'e':{'e':np.array([0.0,0.0]),'n':np.array([0.0,0.0])},'n':{'e':np.array([0.0,0.0]),'n':np.array([0.0,0.0])}}
        full_flsc={'e':{'e':np.array([0.0,0.0]),'n':np.array([0.0,0.0])},'n':{'e':np.array([0.0,0.0]),'n':np.array([0.0,0.0])}}
         
        
        if setup.pars['Tren']<31:
            i=3
            m=9
            f=12
            
        else:
            i=5
            m=15
            f=30
            
        for e in ['e','n']:
            for eo in ['e','n']:
            
               
                picky=(state[:,i:m]==2) & (educ[:,i:m]==e) & (edup[:,i:m]==eo) & (female[:,i:m]==1)
                picko=(state[:,m:f]==2) & (educ[:,m:f]==e) & (edup[:,m:f]==eo) & (female[:,m:f]==1)
               
                if picky.any():full_flsm[e][eo][0]=np.mean(np.array(labor[:,i:m])[picky])  
                if picko.any():full_flsm[e][eo][1]=np.mean(np.array(labor[:,m:f])[picko])  
                    
                picky=(state[:,i:m]==3) & (educ[:,i:m]==e) & (edup[:,i:m]==eo) & (female[:,i:m]==1)
                picko=(state[:,m:f]==3) & (educ[:,m:f]==e) & (edup[:,m:f]==eo) & (female[:,m:f]==1)
               
                if picky.any():full_flsc[e][eo][0]=np.mean(np.array(labor[:,i:m])[picky])  
                if picko.any():full_flsc[e][eo][1]=np.mean(np.array(labor[:,m:f])[picko])  
     
     
    ########################################################### 
    #Ever MArried and Cohabited 
    ######################################################### 
         
    relt=np.zeros((len(state_codes),lenn))    
    relt1=np.zeros((len(state_codes),lenn))    
        
    for ist,sname in enumerate(state_codes):  
     
        for t in range(lenn):    
                   
                 
             
            #Arrays for preparation    
            is_state = (((np.any(state_old[:,:max(t+3,0)]==ist,1))  & (female[:,0]==1))  |  ((np.any(state_old[:,:max(t+1,0)]==ist,1))  & (female[:,0]==0)))   
            is_state1 = (state[:,t]==ist) 
            
            if not (np.any(is_state) or np.any(is_state1)): continue    
             
          
            #Relationship over time    
            relt[ist,t]=np.sum(is_state)    
            relt1[ist,t]=np.sum(is_state1)    
             
                  
    #Now, before saving the moments, take interval of 5 years    
    # if (mdl.setup.pars['Tret']>=mdl.setup.pars['Tret']):            
    reltt=relt[:,:mdl.setup.pars['Tret']-mdl.setup.pars['Tbef']+1]    
    years=np.linspace(20,35,4)    
    years_model=np.linspace(20,35,int(15/mdl.setup.pars['py']))   
    
    #Find the right entries for creating moments    
    if setup.pars['Tren']<31:
        pos=[0,4, 8, 12]
    else:
        pos=[0, 5, 10, 15]
    #Approximation if more than 5 years in one period    
    if len(pos)<4:    
        for i in range(4-len(pos)):    
            pos=pos+[pos[-1]]    
    pos=np.array(pos)    
        
        
        
     
    reltt=reltt[:,pos]  
    moments['everm']=reltt[2,:]/N
    moments['everc']=reltt[3,:]/N
                 
         

        
   
    
    moments['share single'] = reltt[0,:]/N    
    moments['share mar'] = reltt[2,:]/N    
    moments['share coh'] = reltt[3,:]/N    
     
    
    
     
    ########################################################### 
    #Ever in a relationship by Education
    ######################################################### 
         
    Erelt=np.zeros((2,lenn))    
    Erelt1=np.zeros((2,lenn))    
        
   
    for ed,edn in zip(['e','n'],range(len(['e','n']))):
        for t in range(lenn):    
                   
                 
             
            #Arrays for preparation    
            is_state = ((np.any(state_old[:,:max(t+3,0)]>1,1)) & (educ[:,t+1]==ed) & (female[:,0]==1)) |  ((np.any(state_old[:,:max(t+1,0)]>1,1)) & (educ[:,max(t-1,0)]==ed) & (female[:,0]==0))      
            is_state1 = (state[:,t]>1)    
            
            if not (np.any(is_state) or np.any(is_state1)): continue    
             
          
            #Relationship over time    
            Erelt[edn,t]=np.sum(is_state)    
            Erelt1[edn,t]=np.sum(is_state1)    
             
                  
    #Now, before saving the moments, take interval of 5 years    
    # if (mdl.setup.pars['Tret']>=mdl.setup.pars['Tret']):            
    Ereltt=Erelt[:,:mdl.setup.pars['Tret']-mdl.setup.pars['Tbef']+1]           
    Ereltt=Ereltt[:,pos]    
    
    
    moments['everr_e']=Ereltt[0,:]/np.sum(educ[:,0]=='e')
    moments['everr_ne']=Ereltt[1,:]/np.sum(educ[:,0]=='n')
    
    #
  
    ecoh1=np.cumsum(state==3,axis=1)>1
    emar1=np.cumsum(state==2,axis=1)>1
    ec=np.cumsum(ecoh1,axis=1)
    em=np.cumsum(emar1,axis=1)
    
    firstmar=np.any(((em>ec) & (em>0))==True,axis=1)
    firstcoh=np.any(((ec>em) & (ec>0))==True,axis=1)
    

    weightsh=np.ones(state[:,0].shape)
    weightsh[(firstmar+firstcoh)==False]=0.0
    try:
        ratio_mar=np.average(firstmar[educ[:,0]=='n'],weights=weightsh[educ[:,0]=='n'])/np.average(firstmar[educ[:,0]=='e'],weights=weightsh[educ[:,0]=='e'])
    except:
        ratio_mar=1.0
        
    moments['ratio_mar']=ratio_mar
    
    ########################################################### 
    #Ever in a relationship by Education 2
    ######################################################### 
         
    EErelt=np.zeros((2,2,lenn))    
   
   
    for ed,edn in zip(['e','n'],range(len(['e','n']))):
        for t in range(lenn):    
                   
                 
             
            #Arrays for preparation    
            is_statem = (np.any(state[:,:t+1]==2,1)) & (educ[:,t+1]==ed)          
            is_statec = (np.any(state[:,:t+1]==3,1)) & (educ[:,t+1]==ed)    
            
            if not (np.any(is_state) or np.any(is_state1)): continue    
             
          
            #Relationship over time    
            EErelt[0,edn,t]=np.sum(is_statem)   
            EErelt[1,edn,t]=np.sum(is_statec)   
           
             
                  
    #Now, before saving the moments, take interval of 5 years    
    # if (mdl.setup.pars['Tret']>=mdl.setup.pars['Tret']):            
    EEreltt=EErelt[:,:,:mdl.setup.pars['Tret']-mdl.setup.pars['Tbef']+1]           
    EEreltt=EEreltt[:,:,pos]    
    

 
 
   
     
    ################################################## 
    #EVERYTHING BELOW IS NOT NECESSARY FOR MOMENTS 
    ################################################# 
     
    if draw:  
        
            
        ########################################################### 
        #Ever MArried and Cohabited 
        ######################################################### 
             
        Frelt=np.zeros((len(state_codes_full),lenn))    
        Frelt1=np.zeros((len(state_codes_full),lenn))    
            
        for ist,sname in enumerate(state_codes_full):  
         
            for t in range(lenn):    
                       
                     
                 
                #Arrays for preparation    
                is_state = (np.any(state_true[:,:t+1]==ist,1))           
                is_state1 = (state_true[:,t]==ist)    
                
                if not (np.any(is_state) or np.any(is_state1)): continue    
                 
              
                #Relationship over time    
                Frelt[ist,t]=np.sum(is_state)    
                Frelt1[ist,t]=np.sum(is_state1)    
                 
                      
       
             
        #Update N to the new sample size    
        #N=len(state)    
     
        ###########################################    
        #Moments: Variables over Age    
        ###########################################    
         
        ifemale2=(female[:,0]==1) 
        imale2=(female[:,0]==0) 
        ass_rel=np.zeros((len(state_codes),lenn,2))    
        inc_rel=np.zeros((len(state_codes),lenn,2))  
        log_inc_rel=np.zeros((2,2,len(state)))  
         
        #Create wages 
        wage_f=np.zeros(state.shape) 
        wage_m=np.zeros(state.shape) 
         
        wage_fc=np.zeros(len(state[0,:])) 
        wage_mc=np.zeros(len(state[0,:])) 
        wage_fm=np.zeros(len(state[0,:])) 
        wage_mm=np.zeros(len(state[0,:])) 
        wage_fr=np.zeros(len(state[0,:])) 
        wage_mr=np.zeros(len(state[0,:])) 
         
        wage_f2=np.zeros(state.shape) 
        wage_m2=np.zeros(state.shape) 
        wage_fp=np.zeros(state.shape) 
        wage_mp=np.zeros(state.shape) 
         
        wage_fpc=np.zeros(len(state[0,:])) 
        wage_mpc=np.zeros(len(state[0,:])) 
        wage_fpm=np.zeros(len(state[0,:])) 
        wage_mpm=np.zeros(len(state[0,:])) 
        wage_fpr=np.zeros(len(state[0,:])) 
        wage_mpr=np.zeros(len(state[0,:])) 
         
        wage_fs=np.zeros(len(state[0,:])) 
        wage_ms=np.zeros(len(state[0,:])) 
         
        var_wage_fc=np.zeros(len(state[0,:])) 
        var_wage_mc=np.zeros(len(state[0,:])) 
        var_wage_fm=np.zeros(len(state[0,:])) 
        var_wage_mm=np.zeros(len(state[0,:])) 
        var_wage_fr=np.zeros(len(state[0,:])) 
        var_wage_mr=np.zeros(len(state[0,:])) 
         
        var_wage_fpc=np.zeros(len(state[0,:])) 
        var_wage_mpc=np.zeros(len(state[0,:])) 
        var_wage_fpm=np.zeros(len(state[0,:])) 
        var_wage_mpm=np.zeros(len(state[0,:])) 
        var_wage_fpr=np.zeros(len(state[0,:])) 
        var_wage_mpr=np.zeros(len(state[0,:])) 
         
        #For assets 
        assets_fc=np.zeros(len(state[0,:])) 
        assets_fm=np.zeros(len(state[0,:])) 
        assets_mc=np.zeros(len(state[0,:])) 
        assets_mm=np.zeros(len(state[0,:])) 
         
        assets_fpc=np.zeros(len(state[0,:])) 
        assets_fpm=np.zeros(len(state[0,:])) 
        assets_mpc=np.zeros(len(state[0,:])) 
        assets_mpm=np.zeros(len(state[0,:])) 
         
        var_assets_fc=np.zeros(len(state[0,:])) 
        var_assets_fm=np.zeros(len(state[0,:])) 
        var_assets_mc=np.zeros(len(state[0,:])) 
        var_assets_mm=np.zeros(len(state[0,:])) 
         
        var_assets_fpc=np.zeros(len(state[0,:])) 
        var_assets_fpm=np.zeros(len(state[0,:])) 
        var_assets_mpc=np.zeros(len(state[0,:])) 
        var_assets_mpm=np.zeros(len(state[0,:])) 
         
        corr_ass_sepm=np.zeros(len(state[0,:])) 
        corr_ass_sepf=np.zeros(len(state[0,:])) 
        share_ass_sepm=np.zeros(len(state[0,:])) 
        share_ass_sepf=np.zeros(len(state[0,:])) 
        mcorr_ass_sepm=np.zeros(len(state[0,:])) 
        mcorr_ass_sepf=np.zeros(len(state[0,:])) 
        mshare_ass_sepm=np.zeros(len(state[0,:])) 
        mshare_ass_sepf=np.zeros(len(state[0,:])) 
         
         
      
        ifemale=(female[:,0]==1) 
        imale=(female[:,0]==0) 
        
            #Routine for computing the wage trend
        def wtrend(sex,age,education,compute='mean'):
            
            if compute=='mean':
                meane=np.mean(education==['e'])
                meann=np.mean(education==['n'])
                return meane*np.array(setup.pars['wtrend'][sex]['e'])[age[(education=='e')]]+meann*np.array(setup.pars['wtrend'][sex]['n'])[age[(education=='n')]]
            
            else:
                
                return np.array(setup.pars['wtrend'][sex][compute])[age[(education==compute)]]
                 
 
        for i in range(len(state[0,:])): 
             
            #For Income 
            singlef=(ifemale) & (state[:,i]==0) 
            singlem=(imale) & (state[:,i]==1)        
            nsinglef=(ifemale) & (state[:,i]>=2) 
            nsinglem=(imale) & (state[:,i]>=2) 
            nsinglefc=(ifemale) & (state[:,i]==3) 
            nsinglemc=(imale) & (state[:,i]==3) 
            nsinglefm=(ifemale) & (state[:,i]==2) 
            nsinglemm=(imale) & (state[:,i]==2) 
            nsinglefr=(ifemale) & (state[:,i]>=2) 
            nsinglemr=(imale) & (state[:,i]>=2) 
             
            singlef2=(ifemale2) & (state[:,i]==0) 
            singlem2=(imale2) & (state[:,i]==1)        
            nsinglef2=(ifemale2) & (state[:,i]>=2) 
            nsinglem2=(imale2) & (state[:,i]>=2) 
             
            #For assets 
            cohf=(ifemale) & (state[:,i]==3) & (state[:,max(i-1,0)]<=1) 
            marf=(ifemale) & (state[:,i]==2) & (state[:,max(i-1,0)]<=1) 
            cohm=(imale) & (state[:,i]==3) & (state[:,max(i-1,0)]<=1) 
            marm=(imale) & (state[:,i]==2) & (state[:,max(i-1,0)]<=1) 
             
            acm=(imale) & (state[:,i]==1) & (state[:,max(i-1,0)]==3) 
            acf=(ifemale) & (state[:,i]==0) & (state[:,max(i-1,0)]==3) 
            macm=(imale) & (state[:,i]==1) & (state[:,max(i-1,0)]==2) #& (state[:,max(i-2,0)]==2) 
            macf=(ifemale) & (state[:,i]==0) & (state[:,max(i-1,0)]==2)# & (state[:,max(i-2,0)]==2) 
            acmp=(acm) & (assetss_w[:,i]>0)  
            acfp=(acf) & (assetss_w[:,i]>0) 
            macmp=(macm) & (assetss_w[:,i]>0) 
            macfp=(macf) & (assetss_w[:,i]>0) 
             
            #Corr assets at separation+share 
            if np.any(acf):corr_ass_sepf[i]=np.corrcoef(assets_w[acf,i],(assetss_w[acf,i]-assets_w[acf,i]))[0,1] 
            if np.any(acm):corr_ass_sepm[i]=np.corrcoef(assets_w[acm,i],(assetss_w[acm,i]-assets_w[acm,i]))[0,1] 
            if np.any(acf):share_ass_sepf[i]=0.5 
            if np.any(acm):share_ass_sepm[i]=0.5 
            if np.any(acfp):share_ass_sepf[i]=np.mean(assets_w[acfp,i]/assetss_w[acfp,i]) 
            if np.any(acmp):share_ass_sepm[i]=np.mean(1-assets_w[acmp,i]/assetss_w[acmp,i]) 
            
             
            if np.any(macf):mcorr_ass_sepf[i]=np.corrcoef(assets_w[macf,i],(mdl.setup.div_costs.assets_kept*assetss_w[macf,i]-assets_w[macf,i]))[0,1] 
            if np.any(macm):mcorr_ass_sepm[i]=np.corrcoef(assets_w[macm,i],(mdl.setup.div_costs.assets_kept*assetss_w[macm,i]-assets_w[macm,i]))[0,1] 
            if np.any(macf):mshare_ass_sepf[i]=0.5 
            if np.any(macm):mshare_ass_sepm[i]=0.5 
            if np.any(macfp):mshare_ass_sepf[i]=np.mean(assets_w[macfp,i]/(mdl.setup.div_costs.assets_kept*assetss_w[macfp,i])) 
            if np.any(macmp):mshare_ass_sepm[i]=np.mean(1-assets_w[macmp,i]/(mdl.setup.div_costs.assets_kept*assetss_w[macmp,i])) 
            #print(np.mean(assets_w[marm,i])) 
            #Assets Marriage 
            if np.any(marf):assets_fm[i]=np.mean(assetss_w[marf,i]) 
            if np.any(marm):assets_mm[i]=np.mean(assetss_w[marm,i]) 
            if np.any(marm):assets_fpm[i]=np.mean(assets_w[marm,i]-assetss_w[marm,i]) 
            if np.any(marf):assets_mpm[i]=np.mean(assets_w[marf,i]-assetss_w[marf,i]) 
             
            if np.any(marf):var_assets_fm[i]=np.var(assetss_w[marf,i]) 
            if np.any(marm):var_assets_mm[i]=np.var(assetss_w[marm,i]) 
            if np.any(marm):var_assets_fpm[i]=np.var(assets_w[marm,i]-assetss_w[marm,i]) 
            if np.any(marf):var_assets_mpm[i]=np.var(assets_w[marf,i]-assetss_w[marf,i]) 
             
            #Assets Cohabitaiton 
            if np.any(cohf):assets_fc[i]=np.mean(assetss_w[cohf,i]) 
            if np.any(cohm):assets_mc[i]=np.mean(assetss_w[cohm,i]) 
            if np.any(cohm):assets_fpc[i]=np.mean(assets_w[cohm,i]-assetss_w[cohm,i]) 
            if np.any(cohf):assets_mpc[i]=np.mean(assets_w[cohf,i]-assetss_w[cohf,i]) 
         
            if np.any(cohf):var_assets_fc[i]=np.var(assetss_w[cohf,i]) 
            if np.any(cohm):var_assets_mc[i]=np.var(assetss_w[cohm,i]) 
            if np.any(cohm):var_assets_fpc[i]=np.var(assets_w[cohm,i]-assetss_w[cohm,i]) 
            if np.any(cohf):var_assets_mpc[i]=np.var(assets_w[cohf,i]-assetss_w[cohf,i]) 
         
            
       
            #Here consider heterogeneity by Education
 
            #Aggregate Income 
            if np.any(nsinglef):wage_f[nsinglef,i]=np.concatenate((
                            np.exp(setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglef) & (educ[:,i]=='e')],
                            np.exp(setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglef) & (educ[:,i]=='n')]))
            
            if np.any(nsinglem):wage_m[nsinglem,i]=np.concatenate((
            np.exp(setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglem) & (educ[:,i]=='e')],            
            np.exp(setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglem) & (educ[:,i]=='n')]))
            
            
            if np.any(singlef):wage_f[singlef,i]=np.concatenate((
            np.exp(setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][iexo[singlef,i]])[educ[singlef,i]=='e'],            
            np.exp(setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][iexo[singlef,i]])[educ[singlef,i]=='n']))
            
            
            if np.any(singlem):wage_m[singlem,i]=np.concatenate((
            np.exp(setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][iexo[singlem,i]])[educ[singlem,i]=='e'],            
            np.exp(setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][iexo[singlem,i]])[educ[singlem,i]=='n']))
            
            
            # if np.any(nsinglef):wage_mp[nsinglef,i]=np.concatenate((
            # np.exp(setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglef) & (edupp[:,i]=='e')],            
            # np.exp(setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglef) & (edupp[:,i]=='n')]))
            
            if np.any(nsinglef):wage_mp[(nsinglef) &  (edupp[:,i]=='e') ,i]=np.exp(setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglef) & (edupp[:,i]=='e')]    
            if np.any(nsinglef):wage_mp[(nsinglef) &  (edupp[:,i]=='n') ,i]=np.exp(setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglef) & (edupp[:,i]=='n')]    
       
            
            # if np.any(nsinglem):wage_fp[nsinglem,i]=np.concatenate((
            # np.exp(setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglem) & (edupp[:,i]=='e')],            
            # np.exp(setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglem) & (edupp[:,i]=='n')]))
            
            if np.any(nsinglem):wage_fp[(nsinglem) &  (edupp[:,i]=='e') ,i]=np.exp(setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglem) & (edupp[:,i]=='e')]    
            if np.any(nsinglem):wage_fp[(nsinglem) &  (edupp[:,i]=='n') ,i]=np.exp(setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglem) & (edupp[:,i]=='n')]    
       
            #((setup.exogrid.psi_t[i][(setup.all_indices(i,iexo[:,i]))[3]])) 
             
            #For income process validation 
            wage_f2[nsinglef2,i]=np.concatenate((
            np.exp(setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglef2) & (educ[:,i]=='e')],           
            np.exp(setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglef2) & (educ[:,i]=='n')] ))
            
            
            
            wage_m2[nsinglem2,i]=np.concatenate((
            np.exp(setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglem2) & (educ[:,i]=='e')],            
            np.exp(setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglem2) & (educ[:,i]=='n')]))
            
            
            
            wage_m2[nsinglef2,i]=np.concatenate((
            np.exp(setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglef2) & (edupp[:,i]=='e')] ,            
            np.exp(setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])])[(nsinglef2) & (edupp[:,i]=='n')] ))
            
            
            
            wage_f2[nsinglem2,i]=np.concatenate((
            np.exp(setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglem2) & (edupp[:,i]=='e')],            
            np.exp(setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])])[(nsinglem2) & (edupp[:,i]=='n')]))
            
            
            
            wage_f2[singlef2,i]=np.concatenate((
            np.exp(setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][iexo[singlef2,i]])[educ[singlef2,i]=='e'],            
            np.exp(setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][iexo[singlef2,i]])[educ[singlef2,i]=='n']))
            
            
            
            wage_m2[singlem2,i]=np.concatenate((
            np.exp(setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][iexo[singlem2,i]])[educ[singlem2,i]=='e'],            
            np.exp(setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][iexo[singlem2,i]])[educ[singlem2,i]=='n']))
            
            
            #Single only Income 
            wage_fs[i]=np.mean(
            np.concatenate((np.array(setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][iexo[singlef,i]])[educ[singlef,i]=='e'] ,           
                            np.array(setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][iexo[singlef,i]])[educ[singlef,i]=='n'])))
            
            
            wage_ms[i]=np.mean(
            np.concatenate((np.array(setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][iexo[singlem,i]])[educ[singlem,i]=='e'] ,            
                            np.array(setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][iexo[singlem,i]])[educ[singlem,i]=='n'])))
             
            #Cohabitation Income 
            if np.any(nsinglefc):wage_fc[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefc) & (educ[:,i]=='e')],            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefc) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglemc):wage_mc[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemc) & (educ[:,i]=='e')],            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemc) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglefc):wage_mpc[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefc) & (edupp[:,i]=='e')],            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefc) & (edupp[:,i]=='n')])))
            
            
            
            if np.any(nsinglemc):wage_fpc[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemc) & (edupp[:,i]=='e')] ,            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemc) & (edupp[:,i]=='n')]) ))
     
     
            if np.any(nsinglefc):var_wage_fc[i]=np.var(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefc) & (educ[:,i]=='e')],            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefc) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglemc):var_wage_mc[i]=np.var(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemc) & (educ[:,i]=='e')],            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemc) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglefc):var_wage_mpc[i]=np.var(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefc) & (edupp[:,i]=='e')] ,            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefc) & (edupp[:,i]=='n')])))
            
            
            
            if np.any(nsinglemc):var_wage_fpc[i]=np.var(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemc) & (edupp[:,i]=='e')],            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemc) & (edupp[:,i]=='n')])))
     
            #Marriage Income 
            if np.any(nsinglefm):wage_fm[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefm) & (educ[:,i]=='e')],            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefm) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglemm):wage_mm[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemm) & (educ[:,i]=='e')],            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemm) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglefm):wage_mpm[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefm) & (edupp[:,i]=='e')],            
                           setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefm) & (edupp[:,i]=='n')])))
            
                       
            
            if np.any(nsinglemm):wage_fpm[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemm) & (edupp[:,i]=='e')],            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemm) & (edupp[:,i]=='n')])))
     
        
            #Marriage Income 
            if np.any(nsinglefr):wage_fr[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefr) & (educ[:,i]=='e')],            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefr) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglemr):wage_mr[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemr) & (educ[:,i]=='e')],            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemr) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglefr):wage_mpr[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefr) & (edupp[:,i]=='e')],            
                           setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefr) & (edupp[:,i]=='n')])))
            
                       
            
            if np.any(nsinglemr):wage_fpr[i]=np.mean(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemr) & (edupp[:,i]=='e')],            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemr) & (edupp[:,i]=='n')])))
     
            if np.any(nsinglefm):var_wage_fm[i]=np.var(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefm) & (educ[:,i]=='e')],            
                           setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefm) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglemm):var_wage_mm[i]=np.var(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemm) & (educ[:,i]=='e')] ,            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemm) & (educ[:,i]=='n')]) ))
            
            
            
            if np.any(nsinglefm):var_wage_mpm[i]=np.var(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefm) & (edupp[:,i]=='e')] ,            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefm) & (edupp[:,i]=='n')]) ))
            
            
            
            if np.any(nsinglemm):var_wage_fpm[i]=np.var(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemm) & (edupp[:,i]=='e')],            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemm) & (edupp[:,i]=='n')])))
                       
            if np.any(nsinglefr):var_wage_fr[i]=np.var(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefr) & (educ[:,i]=='e')],            
                           setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglefr) & (educ[:,i]=='n')])))
            
            
            
            if np.any(nsinglemr):var_wage_mr[i]=np.var(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemr) & (educ[:,i]=='e')] ,            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglemr) & (educ[:,i]=='n')]) ))
            
            
            
            if np.any(nsinglefr):var_wage_mpr[i]=np.var(
            np.concatenate((setup.pars['wtrend']['m']['e'][i]+setup.exogrid.zm_t['e'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefr) & (edupp[:,i]=='e')] ,            
                            setup.pars['wtrend']['m']['n'][i]+setup.exogrid.zm_t['n'][i][((setup.all_indices(i,iexo[:,i]))[2])][(nsinglefr) & (edupp[:,i]=='n')]) ))
            
            
            
            if np.any(nsinglemr):var_wage_fpr[i]=np.var(
            np.concatenate((setup.pars['wtrend']['f']['e'][i]+setup.exogrid.zf_t['e'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemr) & (edupp[:,i]=='e')],            
                            setup.pars['wtrend']['f']['n'][i]+setup.exogrid.zf_t['n'][i][((setup.all_indices(i,iexo[:,i]))[1])][(nsinglemr) & (edupp[:,i]=='n')])))
                       
            
             
        #Log Income over time 
        for t in range(lenn): 
            for i in range(2):
                iparte=(labor_w[:,t]>0.0) & (ifemale2) & (educ[:,t]=='e') #& (state[:,t]>1) 
                ipartne=(labor_w[:,t]>0.0) & (ifemale2) & (educ[:,t]=='n')
                
                log_inc_rel[0,0,t]=np.mean(np.log(wage_f2[iparte,t])) 
                log_inc_rel[0,1,t]=np.mean(np.log(wage_m2[(imale2) & (educ[:,t]=='e'),t])) 
                
                log_inc_rel[1,0,t]=np.mean(np.log(wage_f2[ipartne,t])) 
                log_inc_rel[1,1,t]=np.mean(np.log(wage_m2[(imale2) & (educ[:,t]=='n'),t])) 
                     
            
            
             
        for ist,sname in enumerate(state_codes):  
                 
         
            for t in range(lenn):    
            
                #Arrays for preparation    
                is_state = (np.any(state[:,0:t]==ist,1))           
                is_state1 = (state[:,t]==ist)    
                is_state2 = (state[:,t]==ist)    
                if t<1:    
                    is_state=is_state1    
                
                ind1 = np.where(is_state1)[0]  
                ind1f = np.where((is_state1) & (agents.is_female[:,0]))[0] 
                ind1m = np.where((is_state1) & ~(agents.is_female[:,0]))[0] 
                         
                if not (np.any(is_state) or np.any(is_state1)): continue    
                     
                zf,zm,psi=mdl.setup.all_indices(t,iexo[ind1,t])[1:5]    
                         
 
                     
                #Assets over time      
                if sname=="Female, single" or  sname=="Male, single":  
                     
                    if np.any(is_state2):ass_rel[ist,t,0]=np.mean(assets_w[is_state2,t])   
                 
                else: 
                 
                     
                    if np.any((is_state2) & (ifemale2)):ass_rel[ist,t,0]=np.mean(assets_w[(is_state2) & (ifemale2),t])  
                    if np.any((is_state2) & (imale2)):ass_rel[ist,t,1]=np.mean(assets_w[(is_state2) & (imale2),t])  
                     
     
                 
                if sname=="Female, single":   
                     
                    inc_rel[ist,t,0]=np.mean(wage_f[ind1f,t])#max(np.mean(np.log(wage_f[ind1,t])),-0.2)#np.mean(np.exp(mdl.setup.exogrid.zf_t[s][zf]  + ftrend ))  #  
                     
                    
                elif sname=="Male, single":   
                     
                     inc_rel[ist,t,0]=np.mean(wage_m[ind1m,t])#np.mean(np.exp(mdl.setup.exogrid.zf_t[s][zm] + mtrend))  #np.mean(wage_m[(state[:,t]==ist)])# 
                      
                      
                elif sname=="Couple, C" or sname=="Couple, M":  
                     
                     if np.any(ind1f):inc_rel[ist,t,0]=np.mean(wage_f[:,t][ind1f])+np.mean(wage_mp[:,t][ind1f]) 
                     if np.any(ind1m):inc_rel[ist,t,1]=np.mean(wage_m[:,t][ind1m])+np.mean(wage_fp[:,t][ind1m]) 
                      
                else:   
                    
                   print('Error: No relationship chosen')   
                    

                 
        #Check correlations 
       
        ifemale1=(female==1) 
        imale1=(female==0) 
        nsinglefc1=(ifemale1[:,:60]) & (state[:,:60]==3) & (labor_w[:,:60]>0.1) 
        nsinglefm1=(ifemale1[:,:60]) & (state[:,:60]==2) & (labor_w[:,:60]>0.1) 
        nsinglefc2=(imale1[:,:60]) & (state[:,:60]==3) & (labor_w[:,:60]>0.1) 
        nsinglefm2=(imale1[:,:60]) & (state[:,:60]==2) & (labor_w[:,:60]>0.1) 
        wage_ft=wage_f[:,:60] 
        wage_mpt=wage_mp[:,:60] 
        wage_mt=wage_m[:,:60] 
        wage_fpt=wage_fp[:,:60] 
        
        #For constructing relative net worth measure
        agei=int(30/setup.pars['py'])
        agef=int(40/setup.pars['py'])
        if (agei==agef):agef=agef+1
        assets_w_mod=assets_w[:,agei:agef]
        nsinglefc1_mod=(ifemale1[:,agei:agef]) & (state[:,agei:agef]==3) & (labor_w[:,agei:agef]>0) 
        nsinglefm1_mod=(ifemale1[:,agei:agef]) & (state[:,agei:agef]==2) & (labor_w[:,agei:agef]>0) 
        nsinglefc2_mod=(imale1[:,agei:agef]) & (state[:,agei:agef]==3) & (labor_w[:,agei:agef]>0) 
        nsinglefm2_mod=(imale1[:,agei:agef]) & (state[:,agei:agef]==2) & (labor_w[:,agei:agef]>0) 
        nsinglefc1_mod1=(ifemale1[:,agei:agef]) & (state[:,agei:agef]==3)  
        nsinglefm1_mod1=(ifemale1[:,agei:agef]) & (state[:,agei:agef]==2)
        nsinglefc2_mod1=(imale1[:,agei:agef]) & (state[:,agei:agef]==3)
        nsinglefm2_mod1=(imale1[:,agei:agef]) & (state[:,agei:agef]==2)
        wage_ft_mod=wage_f[:,agei:agef] 
        wage_mpt_mod=wage_mp[:,agei:agef] 
        wage_mt_mod=wage_m[:,agei:agef] 
        wage_fpt_mod=wage_fp[:,agei:agef] 
        labor_w_mod=labor_w[:,agei:agef]
        

        #Wages correlations
        corr=np.corrcoef(wage_ft[nsinglefc1]*setup.ls_levels[-1],wage_mpt[nsinglefc1]) 
        corr1=np.corrcoef(wage_ft[nsinglefm1]*setup.ls_levels[-1],wage_mpt[nsinglefm1]) 
        corrm=np.corrcoef(wage_mt[nsinglefc2],wage_fpt[nsinglefc2]*setup.ls_levels[-1])
        corrm1=np.corrcoef(wage_mt[nsinglefm2],wage_fpt[nsinglefm2]*setup.ls_levels[-1]) 
        share_fcm=np.mean(wage_ft[nsinglefc1]*setup.ls_levels[-1]/(wage_mpt[nsinglefc1]+wage_ft[nsinglefc1]*setup.ls_levels[-1])) 
        share_fmm=np.mean(wage_ft[nsinglefm1]*setup.ls_levels[-1]/(wage_mpt[nsinglefm1]+wage_ft[nsinglefm1]*setup.ls_levels[-1])) 
        share_mcm=np.mean(wage_fpt[nsinglefc2]*setup.ls_levels[-1]/(wage_mt[nsinglefc2]+wage_fpt[nsinglefc2]*setup.ls_levels[-1])) 
        share_mmm=np.mean(wage_fpt[nsinglefm2]*setup.ls_levels[-1]/(wage_mt[nsinglefm2]+wage_fpt[nsinglefm2]*setup.ls_levels[-1])) 
        

        print('FM Correlation in potential wages for cohabitaiton is {}, for marriage only is {}'.format(corr[0,1],corr1[0,1]) )   
        print('MM Correlation in potential wages for cohabitaiton is {}, for marriage only is {}'.format(corrm[0,1],corrm1[0,1]) )   
        print('FM Share wages earned by female in cohabitaiton is {}, for marriage only is {}'.format(share_fcm,share_fmm) )   
        print('MM Share wages earned by female in cohabitaiton is {}, for marriage only is {}'.format(share_mcm,share_mmm) )  

        #For assortative matin
        lime=min(20,len(edup)-1)
       
        educ1=educ[:,0:lime]
        
        edup1=edup[:,0:lime]
        female1=female[:,0:lime]
        state1=state[:,0:lime]
        ee=(educ1=='e') & (edup1=='e')
        ne=(educ1=='n') & (edup1=='e')
        en=(educ1=='e') & (edup1=='n')
        nn=(educ1=='n') & (edup1=='n')
        print('Share m ee: f {}, m {}'.format(np.mean(ee[(edup1!='single') & (female1==1) & (state1==2)]),np.mean(ee[(edup1!='single') & (female1==0) & (state1==2)])))
        print('Share m ne: f {}, m {}'.format(np.mean(ne[(edup1!='single') & (female1==1) & (state1==2)]),np.mean(en[(edup1!='single') & (female1==0) & (state1==2)])))
        print('Share m en: f {}, m {}'.format(np.mean(en[(edup1!='single') & (female1==1) & (state1==2)]),np.mean(ne[(edup1!='single') & (female1==0) & (state1==2)])))
        print('Share m nn: f {}, m {}'.format(np.mean(nn[(edup1!='single') & (female1==1) & (state1==2)]),np.mean(nn[(edup1!='single') & (female1==0) & (state1==2)])))
        print('Share c ee: f {}, m {}'.format(np.mean(ee[(edup1!='single') & (female1==1) & (state1==3)]),np.mean(ee[(edup1!='single') & (female1==0) & (state1==3)])))
        print('Share c ne: f {}, m {}'.format(np.mean(ne[(edup1!='single') & (female1==1) & (state1==3)]),np.mean(en[(edup1!='single') & (female1==0) & (state1==3)])))
        print('Share c en: f {}, m {}'.format(np.mean(en[(edup1!='single') & (female1==1) & (state1==3)]),np.mean(ne[(edup1!='single') & (female1==0) & (state1==3)])))
        print('Share c nn: f {}, m {}'.format(np.mean(nn[(edup1!='single') & (female1==1) & (state1==3)]),np.mean(nn[(edup1!='single') & (female1==0) & (state1==3)])))
                


         
        #Construct fls by first and last point in trans matrix shocks
        lmzz=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==0) & ((setup.all_indices(1,iexo[:,0:25]))[2]==0)  & (state[:,0:25]==2)])
        lmzm=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==0) & ((setup.all_indices(1,iexo[:,0:25]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,0:25]==2)])         
        lmmz=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,0:25]))[2]==0)  & (state[:,0:25]==2)])
        lmmm=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,0:25]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,0:25]==2)])        
        lczz=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==0) & ((setup.all_indices(1,iexo[:,0:25]))[2]==0)  & (state[:,0:25]==3)])
        lczm=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==0) & ((setup.all_indices(1,iexo[:,0:25]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,0:25]==3)])         
        lcmz=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,0:25]))[2]==0)  & (state[:,0:25]==3)])
        lcmm=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,0:25]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,0:25]==3)])
                                                            
        lmzi=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==0) & ((setup.all_indices(1,iexo[:,0:25]))[2]==1)  & (state[:,0:25]==2)])
        lmiz=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==1) & ((setup.all_indices(1,iexo[:,0:25]))[2]==0)  & (state[:,0:25]==2)])
        lmmi=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==2) & ((setup.all_indices(1,iexo[:,0:25]))[2]==1)  & (state[:,0:25]==2)])
        lmim=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==1) & ((setup.all_indices(1,iexo[:,0:25]))[2]==2)  & (state[:,0:25]==2)])
        lmii=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==1) & ((setup.all_indices(1,iexo[:,0:25]))[2]==1)  & (state[:,0:25]==2)])
        lczi=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==0) & ((setup.all_indices(1,iexo[:,0:25]))[2]==1)  & (state[:,0:25]==3)])
        lciz=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==1) & ((setup.all_indices(1,iexo[:,0:25]))[2]==0)  & (state[:,0:25]==3)])
        lcmi=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==2) & ((setup.all_indices(1,iexo[:,0:25]))[2]==1)  & (state[:,0:25]==3)])
        lcim=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==1) & ((setup.all_indices(1,iexo[:,0:25]))[2]==2)  & (state[:,0:25]==3)])
        lcii=np.mean(labor[:,0:25][((setup.all_indices(1,iexo[:,0:25]))[1]==1) & ((setup.all_indices(1,iexo[:,0:25]))[2]==1)  & (state[:,0:25]==3)])
        
        
        
        print('zz grid, m {} c{}'.format(lmzz,lczz))
        print('zm grid, m {} c{}'.format(lmzm,lczm))
        print('mz grid, m {} c{}'.format(lmmz,lcmz))
        print('mm grid, m {} c{}'.format(lmmm,lcmm))
        print('zi grid, m {} c{}'.format(lmzi,lczi))
        print('iz grid, m {} c{}'.format(lmiz,lciz))
        print('mi grid, m {} c{}'.format(lmmi,lcmi))
        print('im grid, m {} c{}'.format(lmim,lcim))
        print('ii grid, m {} c{}'.format(lmii,lcii))
        
        #Get useful package for denisty plots 
        import seaborn as sns 
         
        #Print something useful for debug and rest    
        print('The share of singles choosing marriage is {0:.2f}'.format(sharem))    
 
        #Setup a file for the graphs    
        import time
        timestr = time.strftime("%H%M%S")
        
        pdf = matplotlib.backends.backend_pdf.PdfPages("moments"+timestr+"_graphs.pdf")    
             
        #################    
        #Get data moments    
        #################    
             
        #Get Data Moments    
        with open('moments.pkl', 'rb') as file:    
            packed_data=pickle.load(file)    
             
            #Unpack Moments (see data_moments.py to check if changes)    
     
            
            hazs_d=packed_data['hazs']   
            hazm_d=packed_data['hazm']   
            hazd_d=packed_data['hazd']  
            hazde_d=packed_data['hazde']  
            everc_d=packed_data['everc']   
            everm_d=packed_data['everm']
            everr_e_d=packed_data['everr_e']
            everr_ne_d=packed_data['everr_ne']
            flsc_d=packed_data['flsc']
            flsm_d=packed_data['flsm']
            beta_edu_d=packed_data['beta_edu']
            ref_coh_d=packed_data['ref_coh']
            ratio_mar_d=packed_data['ratio_mar']
            
            hazs_i=packed_data['hazsi']   
            hazm_i=packed_data['hazmi']   
            hazd_i=packed_data['hazdi']
            hazde_i=packed_data['hazdei']
            everc_i=packed_data['everci']   
            everm_i=packed_data['evermi']
            everr_e_i=packed_data['everr_ei']
            everr_ne_i=packed_data['everr_nei']
            flsc_i=packed_data['flsci']
            flsm_i=packed_data['flsmi']
            beta_edu_i=packed_data['beta_edui']
            ref_coh_i=packed_data['ref_cohi']  
            ratio_mari=packed_data['ratio_mari']
     
             
             
        #############################################    
        # Hazard of Divorce    
        #############################################    
        fig = plt.figure()    
        f1=fig.add_subplot(1.5,1,1)    
        lg=min(len(hazd_d),len(hazd))  
        if lg<2:    
            one='o'    
            two='o'    
        else:    
            one='r'    
            two='b'    
        plt.plot(np.array(range(lg))+1, hazd[0:lg],one, linestyle='--',linewidth=1.5, label='Simulation')    
        plt.plot(np.array(range(lg))+1, hazd_d[0:lg],two,linewidth=1.5, label='Data')    
        plt.fill_between(np.array(range(lg))+1, hazd_i[0,0:lg], hazd_i[1,0:lg],alpha=0.2,facecolor='b') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                  fancybox=True, shadow=True, ncol=2, fontsize=14)   
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Years', fontsize=16)    
        plt.ylabel('Hazard', fontsize=16)    
        plt.savefig('hazd.pgf', bbox_inches = 'tight',pad_inches = 0)  
        
        
        #############################################    
        # Hazard of Divorce    
        #############################################    
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1)    
        lg=min(len(hazd_d),len(hazd))  
        if lg<2:    
            one='o'    
            two='o'    
        else:    
            one='r'    
            two='b'      
        plt.plot(np.array(range(lg))+1, hazd_d[0:lg],two,linewidth=1.5)    
        plt.fill_between(np.array(range(lg))+1, hazd_i[0,0:lg], hazd_i[1,0:lg],alpha=0.2,facecolor='b') 
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
         #         fancybox=True, shadow=True, ncol=3, fontsize='x-small')    
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Years')    
        plt.ylabel('Hazard')    
        plt.savefig('hazd_data.pgf', bbox_inches = 'tight',pad_inches = 0)  
        
        
        
        #############################################    
        # Hazard of Divorce- Educated  
        #############################################    
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1)    
        lg=min(len(hazd_d),len(hazd))  
        if lg<2:    
            one='o'    
            two='o'    
        else:    
            one='r'    
            two='b'    
        plt.plot(np.array(range(lg))+1, hazde[0:lg],one, linestyle='--',linewidth=1.5, label='Hazard of Divorce edu - S')    
        plt.plot(np.array(range(lg))+1, hazde_d[0:lg],two,linewidth=1.5, label='Hazard of Divorce edy - D')    
        plt.fill_between(np.array(range(lg))+1, hazde_i[0,0:lg], hazde_i[1,0:lg],alpha=0.2,facecolor='b') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                  fancybox=True, shadow=True, ncol=2, fontsize=14)     
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Years', fontsize=16)    
        plt.ylabel('Hazard', fontsize=16)    
        plt.savefig('hazde.pgf', bbox_inches = 'tight',pad_inches = 0)  
             
        #############################################    
        # Hazard of Separation    
        #############################################    
        fig = plt.figure()    
        f1=fig.add_subplot(1.5,1,1)    
        lg=min(len(hazs_d),len(hazs))  
        plt.plot(np.array(range(lg))+1, hazs[0:lg],one, linestyle='--',linewidth=1.5, label='Simulation')    
        plt.plot(np.array(range(lg))+1, hazs_d[0:lg],two,linewidth=1.5, label='Data')    
        plt.fill_between(np.array(range(lg))+1, hazs_i[0,0:lg], hazs_i[1,0:lg],alpha=0.2,facecolor='b')  
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                  fancybox=True, shadow=True, ncol=2, fontsize=14)      
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Years', fontsize=16)    
        plt.ylabel('Hazard', fontsize=16)    
        plt.savefig('hazs.pgf', bbox_inches = 'tight',pad_inches = 0)  
        
        #############################################    
        # Hazard of Separation    Educated
        #############################################    
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1)    
        lg=min(len(hazs_d),len(hazse))  
        plt.plot(np.array(range(lg))+1, hazse[0:lg],one, linestyle='--',linewidth=1.5, label='Hazard of Separation Edu - S')    
        plt.legend(loc='best', ncol=1, fontsize='x-small',frameon=False)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
         #         fancybox=True, shadow=True, ncol=3, fontsize='x-small')    
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Years', fontsize=16)    
        plt.ylabel('Hazard', fontsize=16)    
        
            
             
        #############################################    
        # Hazard of Marriage    
        #############################################    
        fig = plt.figure()    
        f1=fig.add_subplot(1.5,1,1)    
        lg=min(len(hazm_d),len(hazm))  
    
        plt.plot(np.array(range(lg))+1, hazm[0:lg],one, linestyle='--',linewidth=1.5, label='Simulation')    
        plt.plot(np.array(range(lg))+1, hazm_d[0:lg],two,linewidth=1.5, label='Data')    
        plt.fill_between(np.array(range(lg))+1, hazm_i[0,0:lg], hazm_i[1,0:lg],alpha=0.2,facecolor='b') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                  fancybox=True, shadow=True, ncol=2, fontsize=14)     
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Years', fontsize=16)    
        plt.ylabel('Hazard', fontsize=16)    
        plt.savefig('hazm.pgf', bbox_inches = 'tight',pad_inches = 0)  
        
        
        #############################################    
        # Hazard of Marriage   Educated  
        #############################################    
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1)    
        lg=min(len(hazm_d),len(hazme))  
    
        plt.plot(np.array(range(lg))+1, hazme[0:lg],one, linestyle='--',linewidth=1.5, label='Hazard of Marriage Edu- S')    
        plt.legend(loc='best', ncol=1, fontsize='x-small',frameon=False)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
         #         fancybox=True, shadow=True, ncol=3, fontsize='x-small')    
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Edu Years')    
        plt.ylabel('Hazard')    
        
        
     
         
        ##########################################    
        # Assets Over the Live Cycle    
        ##########################################    
        fig = plt.figure()    
        f2=fig.add_subplot(2,1,1)    
             
        for ist,sname in enumerate(state_codes):    
            plt.plot(np.array(range(lenn)), ass_rel[ist,:,0],markersize=6, label=sname)    
        plt.plot(np.array(range(lenn)), ass_rel[2,:,1], linestyle='--',color='r',markersize=6, label='Marriage male') 
        plt.plot(np.array(range(lenn)), ass_rel[3,:,1], linestyle='--',color='b',markersize=6, label='Cohabitation other')
        plt.legend(loc='best', ncol=1, fontsize='x-small')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Time')    
        plt.ylabel('Assets')    
             
        ##########################################    
        # Income Over the Live Cycle    
        ##########################################    
        fig = plt.figure()    
        f3=fig.add_subplot(2,1,1)    
             
        for ist,sname in enumerate(state_codes):    
               
            plt.plot(np.array(range(lenn)), inc_rel[ist,:,0],markersize=6, label=sname)  
             
        plt.plot(np.array(range(lenn)), inc_rel[2,:,1], linestyle='--',color='r',markersize=6, label='Marriage male') 
        plt.plot(np.array(range(lenn)), inc_rel[3,:,1], linestyle='--',color='b',markersize=6, label='Cohabitation Male') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Time')    
        plt.ylabel('Income')    
         
        ##########################################    
        # Log Income and Data 
        ##########################################  
         
        #Import Data 
        inc_men = pd.read_csv("income_men.csv")   
        inc_women = pd.read_csv("income_women.csv")   
         
         
        fig = plt.figure()    
        f3=fig.add_subplot(2,1,1)    
         
        lend=len(inc_men['earn_age']) 
        agea=np.array(range(lend))+20 
        #plt.plot(agea, inc_men['earn_age'], marker='o',color='r',markersize=6, label='Men Data') 
        #plt.plot(agea, inc_women['earn_age'], marker='o',color='b',markersize=6, label='Women Data') 
        plt.plot(agea, log_inc_rel[0,0,mdl.setup.pars['Tbef']:lend+mdl.setup.pars['Tbef']],'y',markersize=6, label='Women Simulation e') 
        plt.plot(agea, log_inc_rel[0,1,mdl.setup.pars['Tbef']:lend+mdl.setup.pars['Tbef']],'k',markersize=6, label='Men Simulation e') 
        plt.plot(agea, log_inc_rel[1,0,mdl.setup.pars['Tbef']:lend+mdl.setup.pars['Tbef']],'r',markersize=6, label='Women Simulation ne') 
        plt.plot(agea, log_inc_rel[1,1,mdl.setup.pars['Tbef']:lend+mdl.setup.pars['Tbef']],'g',markersize=6, label='Men Simulation ne') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Income') 
        
        ########################
        #Income and Data
        ############################
        fig = plt.figure()    
        f3=fig.add_subplot(1.5,1,1)    
         

               
        inc_men = pd.read_csv("divome.csv")  
        lend=len(inc_men['wtmedian']) 
        agea=np.array(range(lend))+20 
        plt.scatter(agea, inc_men['wtmedian'], marker='o',color='b',s=60, facecolors='none',  label='Data') 
        plt.plot(agea, log_inc_rel[0,1,:lend],'r',markersize=6, label='Simulations')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                  fancybox=True, shadow=True, ncol=2, fontsize=14)    
        plt.xlabel('Age', fontsize=16)    
        plt.ylabel('Average Log Wage', fontsize=16)
        plt.ylim(ymax=4.5,ymin=2.0) 
        plt.savefig('em.pgf', bbox_inches = 'tight',pad_inches = 0)  
        
        
        fig = plt.figure()    
        f3=fig.add_subplot(1.5,1,1)    

        inc_men = pd.read_csv("divomn.csv")  
        plt.scatter(agea, inc_men['wtmedian'], marker='o',color='b',s=60, facecolors='none', label='Data') 
        plt.plot(agea, log_inc_rel[1,1,:lend],'r',markersize=6, label='Simulations')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                  fancybox=True, shadow=True, ncol=2, fontsize=14)    
        plt.xlabel('Age', fontsize=16)    
        plt.ylabel('Average Log Wage', fontsize=16)
        plt.ylim(ymax=4.5,ymin=2.0) 
        plt.savefig('nm.pgf', bbox_inches = 'tight',pad_inches = 0)  
        
        fig = plt.figure()    
        f3=fig.add_subplot(1.5,1,1)  
        inc_women = pd.read_csv("divofn.csv")  
        plt.scatter(agea, inc_women['wtmedian'], marker='o',color='b',s=60, facecolors='none', label='Data') 
        plt.plot(agea, log_inc_rel[1,0,mdl.setup.pars['Tbef']:lend+mdl.setup.pars['Tbef']],'r',markersize=6, label='Simulations')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                  fancybox=True, shadow=True, ncol=2, fontsize=14)    
        plt.xlabel('Age', fontsize=16)    
        plt.ylabel('Average Log Wage', fontsize=16)
        plt.ylim(ymax=4.5,ymin=2.0) 
        plt.savefig('nf.pgf', bbox_inches = 'tight',pad_inches = 0)  
        
        
        fig = plt.figure()    
        f3=fig.add_subplot(1.5,1,1)  
        inc_women = pd.read_csv("divofe.csv")  
        plt.scatter(agea, inc_women['wtmedian'], marker='o',color='b',s=60, facecolors='none', label='Data') 
        plt.plot(agea, log_inc_rel[0,0,mdl.setup.pars['Tbef']:lend+mdl.setup.pars['Tbef']],'r',markersize=6, label='Simulations')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                  fancybox=True, shadow=True, ncol=2, fontsize=14)    
        plt.xlabel('Age', fontsize=16)    
        plt.ylabel('Average Log Wage', fontsize=16)
        plt.ylim(ymax=4.5,ymin=2.0) 
        plt.savefig('ef.pgf', bbox_inches = 'tight',pad_inches = 0)  
                     
         
        ##########################################    
        # More on Income 
        ##########################################  
        if mdl.setup.pars['T']>60:

            fig = plt.figure()    
            f3=fig.add_subplot(1.5,1,1)    
             
            lend=len(wage_fs) 
            agea=np.array(range(lend))+18
                   
            plt.plot(agea[5:60], wage_fr[5-mdl.setup.pars['Tbef']:60-mdl.setup.pars['Tbef']],color='b',markersize=3, label='Women (main person)') 
            plt.plot(agea[5:60], wage_mr[5-mdl.setup.pars['Tbef']:60-mdl.setup.pars['Tbef']],color='k',markersize=3, label='Men (main person)') 
            plt.plot(agea[5:60], wage_fpr[5-mdl.setup.pars['Tbef']:60-mdl.setup.pars['Tbef']],linestyle='--',color='r',markersize=3, label='Women (met person)') 
            plt.plot(agea[5:60], wage_mpr[5-mdl.setup.pars['Tbef']:60-mdl.setup.pars['Tbef']],linestyle='--',color='m',markersize=3, label='Men (met person)') 
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                      fancybox=True, shadow=True, ncol=2, fontsize=14)    
            plt.xlabel('Age', fontsize=16)    
            plt.ylabel('Average Log Wage', fontsize=16)
            plt.savefig('sy_minc.pgf', bbox_inches = 'tight',pad_inches = 0)  
             
            ##########################################    
            # Variance Income 
            ##########################################         
            fig = plt.figure()    
            f3=fig.add_subplot(1.5,1,1)    
             
            lend=len(wage_fs) 
            agea=np.array(range(lend))+18
            
            plt.plot(agea[5:60], var_wage_fr[5-mdl.setup.pars['Tbef']:60-mdl.setup.pars['Tbef']],color='b',markersize=3, label='Women (main person)') 
            plt.plot(agea[5:60], var_wage_mr[5-mdl.setup.pars['Tbef']:60-mdl.setup.pars['Tbef']],color='k',markersize=3, label='Men (main person)') 
            plt.plot(agea[5:60], var_wage_fpr[5-mdl.setup.pars['Tbef']:60-mdl.setup.pars['Tbef']],linestyle='--',color='r',markersize=3, label='Women (met person)') 
            plt.plot(agea[5:60], var_wage_mpr[5-mdl.setup.pars['Tbef']:60-mdl.setup.pars['Tbef']],linestyle='--',color='m',markersize=3, label='Men (met person)') 
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),    
                      fancybox=True, shadow=True, ncol=2, fontsize=14)    
            plt.xlabel('Age', fontsize=16)    
            plt.ylabel('Log Wage Variance', fontsize=16)  
            plt.savefig('sy_vinc.pgf', bbox_inches = 'tight',pad_inches = 0)
             
        ##########################################    
        # Level Assets Assets 
        ##########################################         
        fig = plt.figure()    
        f3=fig.add_subplot(2,1,1)    
         
        lend=len(wage_fs) 
        agea=np.array(range(lend))+20 
        
        plt.plot(agea, assets_fc,color='r',markersize=3, label='Women coh') 
        plt.plot(agea, assets_mc,color='b',markersize=3, label='Men coh') 
        plt.plot(agea, assets_fm,color='m',markersize=3, label='Women mar') 
        plt.plot(agea, assets_mm,color='k',markersize=3, label='Men mar') 
        plt.plot(agea, assets_fpc,linestyle='--',color='r',markersize=3, label='Women coh-o') 
        plt.plot(agea, assets_mpc,linestyle='--',color='b',markersize=3, label='Men coh-o') 
        plt.plot(agea, assets_fpm,linestyle='--',color='m',markersize=3, label='Women mar-o') 
        plt.plot(agea, assets_mpm,linestyle='--',color='k',markersize=3, label='Men mar-o') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Asset level first meeting')  
         
        ##########################################    
        # Variance Assets first meeting 
        ##########################################         
        fig = plt.figure()    
        f3=fig.add_subplot(2,1,1)    
         
        lend=len(wage_fs) 
        agea=np.array(range(lend))+20 
        
        plt.plot(agea, var_assets_fc,color='r',markersize=3, label='Women coh') 
        plt.plot(agea, var_assets_mc,color='b',markersize=3, label='Men coh') 
        plt.plot(agea, var_assets_fm,color='m',markersize=3, label='Women mar') 
        plt.plot(agea, var_assets_mm,color='k',markersize=3, label='Men mar') 
        plt.plot(agea, var_assets_fpc,linestyle='--',color='r',markersize=3, label='Women coh-o') 
        plt.plot(agea, var_assets_mpc,linestyle='--',color='b',markersize=3, label='Men coh-o') 
        plt.plot(agea, var_assets_fpm,linestyle='--',color='m',markersize=3, label='Women mar-o') 
        plt.plot(agea, var_assets_mpm,linestyle='--',color='k',markersize=3, label='Men mar-o') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Asset Variance first meeting')  
         
        ##########################################    
        # Correlation Assets at Breack up 
        ##########################################         
        fig = plt.figure()    
        f3=fig.add_subplot(2,1,1)    
         
        lend=len(wage_fs) 
        agea=np.array(range(lend))+20 
        
        plt.plot(agea, corr_ass_sepf,color='r',markersize=3, label='Separation-fm') 
        plt.plot(agea, corr_ass_sepm,color='b',markersize=3, label='Separation-mm') 
        plt.plot(agea, mcorr_ass_sepf,color='m',markersize=3, label='Divorce-fm') 
        plt.plot(agea, mcorr_ass_sepm,color='k',markersize=3, label='Divorce-mm') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Correlation assets at breack up')  
         
        
        ##########################################
        #Premarital Cohabitation and Divorce
        #######################################
        fig = plt.figure()    
        f3=fig.add_subplot(2,1,1)    
         
        gridcd=np.array(np.linspace(1,Tmax,Tmax)-1,dtype=np.int16)
        
        plt.plot(gridcd[0:5], ref_coh_d,color='b',markersize=3, label='Data') 
        plt.fill_between(gridcd[0:5], ref_coh_i[0,:], ref_coh_i[1,:],alpha=0.2,facecolor='g')
        plt.plot(gridcd[0:5], ref_dut[0:5],linestyle='--',color='r',markersize=3, label='Simulation') 
        #plt.plot(gridcd[0:5], raw_dut[0:5],linestyle='--',color='y',markersize=3, label='Simulation')  
        plt.yticks(np.arange(0, 2, 0.2))
        plt.legend(loc='best', fontsize='x-small',frameon=False,ncol=2)     
        plt.xlabel('Premarital Cohabitation Duration (yrs)')    
        plt.ylabel('Rel. Haz. of Divorce')    
        plt.savefig('prec.pgf', bbox_inches = 'tight',pad_inches = 0)
         
         
        ##########################################    
        # Share Assets at Breack up 
        ##########################################         
        fig = plt.figure()    
        f3=fig.add_subplot(2,1,1)    
         
        lend=len(wage_fs) 
        agea=np.array(range(lend))+20 
        
        plt.plot(agea, share_ass_sepf,color='r',markersize=3, label='Separation-fm') 
        plt.plot(agea, share_ass_sepm,color='b',markersize=3, label='Separation-mm') 
        plt.plot(agea, mshare_ass_sepf,color='m',markersize=3, label='Divorce-fm') 
        plt.plot(agea, mshare_ass_sepm,color='k',markersize=3, label='Divorce-mm') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Female Share assets at breack up')  
         
         
                     
        
         
        ##########################################    
        # Relationship Over the Live Cycle    
        ##########################################          
        fig = plt.figure()    
        f4=fig.add_subplot(2,1,1)    
        xa=(mdl.setup.pars['py']*np.array(range(len(relt1[0,])))+20)   
        for ist,sname in enumerate(state_codes):   plt.plot([],[], label=sname)                
        plt.stackplot(xa,relt1[0,]/N,relt1[1,]/N,relt1[2,]/N,relt1[3,]/N,    
                      colors = ['b','y','g','r'])               
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Share')    
        
        ##########################################    
        # Relationship Over the Live Cycle - Full    
        ##########################################          
        fig = plt.figure()    
        f4=fig.add_subplot(2,1,1)    
        xa=(mdl.setup.pars['py']*np.array(range(len(Frelt1[0,])))+20)   
        for ist,sname in enumerate(state_codes_full): plt.plot([],[], label=sname)               
        plt.stackplot(xa,Frelt1[0,]/N,Frelt1[1,]/N,Frelt1[2,]/N,Frelt1[3,]/N,
                         Frelt1[4,]/N,Frelt1[5,]/N,Frelt1[6,]/N,Frelt1[7,]/N,
                         Frelt1[8,]/N,Frelt1[9,]/N,Frelt1[10,]/N,Frelt1[11,]/N,
                         colors=['b','y','g','r','c','m','pink','k','linen','gold','crimson','coral'])               
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True,  ncol=int(len(state_codes_full)/3), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Share')  
             
        ##########################################    
        # Relationship and Data    
        ##########################################  

      
        fig = plt.figure()    
        f4=fig.add_subplot(1.5,1,1)    
        lg=min(len(everm_d),len(relt[1,:]))    
        xa=(5*np.array(range(lg))+20)   
        plt.plot(xa, everm_d[0:lg],'b',linewidth=1.5, label='Married - D')    
        plt.fill_between(xa, everm_i[0,0:lg], everm_i[1,0:lg],alpha=0.2,facecolor='b')    
        plt.plot(xa, reltt[2,0:lg]/N,'r',linestyle='--',linewidth=1.5, label='Married - S')    
        plt.plot(xa, everc_d[0:lg],'k',linewidth=1.5, label='Cohabiting - D')    
        plt.fill_between(xa, everc_i[0,0:lg], everc_i[1,0:lg],alpha=0.2,facecolor='k')    
        plt.plot(xa, reltt[3,0:lg]/N,'m',linestyle='--',linewidth=1.5, label='Cohabiting - S')    
        plt.legend(loc='best', fontsize=14,frameon=False,ncol=2)    
        plt.ylim(ymax=1.0)    
        plt.xlabel('Age', fontsize=16)    
        plt.ylabel('Share',fontsize=16)    
        plt.margins(0,0)  
        plt.savefig('erel.pgf', bbox_inches = 'tight',pad_inches = 0)  
        
        ##########################################    
        # Relationship by Education
        ##########################################   
        fig = plt.figure()    
        f4=fig.add_subplot(1.5,1,1)    
        lg=min(len(everm_d),len(relt[1,:]))    
        xa=(5*np.array(range(lg))+20)   
        plt.plot(xa, everr_e_d,'b',linewidth=1.5, label='College - D')    
        plt.fill_between(xa, everr_e_i[0,:], everr_e_i[1,:],alpha=0.2,facecolor='b')    
        plt.plot(xa, Ereltt[0,0:lg]/np.sum(educ[:,0]=='e'),'r',linestyle='--',linewidth=1.5, label='College - S')    
        plt.plot(xa, everr_ne_d,'k',linewidth=1.5, label='NoCollege - D')    
        plt.fill_between(xa, everr_ne_i[0,:], everr_ne_i[1,:],alpha=0.2,facecolor='k')    
        plt.plot(xa, Ereltt[1,0:lg]/np.sum(educ[:,0]=='n'),'m',linestyle='--',linewidth=1.5, label='NoCollege - S')   
        plt.legend(loc='best', fontsize=14,frameon=False)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
         #         fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')  

        plt.ylim(ymax=1.0)    
        plt.xlabel('Age', fontsize=16)    
        plt.ylabel('Share', fontsize=16)    
        plt.margins(0,0)  
        plt.savefig('erel_edu.pgf', bbox_inches = 'tight',pad_inches = 0) 
       
        
        ##########################################    
        # Relationship by Education
        ##########################################          
        fig = plt.figure()    
        f4=fig.add_subplot(2,1,1)    
        lg=min(len(everm_d),len(relt[1,:]))    
        xa=(5*np.array(range(lg))+20)      
        plt.plot(xa, EEreltt[0,0,0:lg]/np.sum(educ[:,0]=='e'),'g',linestyle='--',linewidth=1.5, label='College M')      
        plt.plot(xa, EEreltt[0,1,0:lg]/np.sum(educ[:,0]=='n'),'r',linestyle='--',linewidth=1.5, label='No College M')  
        plt.plot(xa, EEreltt[1,0,0:lg]/np.sum(educ[:,0]=='e'),'g',linewidth=1.5, label='College C')      
        plt.plot(xa, EEreltt[1,1,0:lg]/np.sum(educ[:,0]=='n'),'r',linewidth=1.5, label='No College C')  
        plt.legend(loc='best', fontsize='x-small',frameon=False)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
         #         fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.ylim(ymax=1.0)    
        plt.xlabel('Age')    
        plt.ylabel('Share')    
        plt.margins(0,0)  
     
             
        ##########################################    
        # FLS Over the Live Cycle    
        ##########################################          
        fig = plt.figure()    
        f5=fig.add_subplot(2,1,1)    
        xa=(mdl.setup.pars['py']*np.array(range(mdl.setup.pars['Tret']))+20)   
        plt.plot(xa, flsm,color='r', label='Marriage')    
        plt.plot(xa, flsc,color='k', label='Cohabitation')        
        plt.legend(loc='best', fontsize='x-small',frameon=False)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
         #         fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('FLS')    
         
        ##########################################    
        # Distribution of Love  
        ##########################################   
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1) 
         
         
        sns.kdeplot(psi_check[statecpsi], shade=True,shade_lowest=False,linewidth=0.01, color="r", bw=.05,label = 'Cohabitaition') 
        sns.kdeplot(psi_check[statempsi], shade=True,shade_lowest=False,linewidth=0.01, color="b", bw=.05,label = 'Marriage') 
        sns.kdeplot(psi_check[changec], color="r", bw=.05,label = 'Cohabitaition Beg') 
        sns.kdeplot(psi_check[changem], color="b", bw=.05,label = 'Marriage Beg')   
        plt.legend(loc='best', fontsize='x-small',frameon=False)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
         #         fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Love Shock')    
        plt.ylabel('Denisty') 
        

         
        ##########################################    
        # Distribution of Love - Cumulative 
        ##########################################   
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1) 
         
        # evaluate the histogram 
        valuesc, basec = np.histogram(psi_check[changec], bins=1000) 
        valuesm, basem = np.histogram(psi_check[changem], bins=1000) 
        #valuesct, basect = np.histogram(psi_check[state==3], bins=1000) 
        #valuesmt, basemt = np.histogram(psi_check[state==2], bins=1000) 
        #evaluate the cumulative 
        cumulativec = np.cumsum(valuesc) 
        cumulativem = np.cumsum(valuesm) 
        #cumulativect = np.cumsum(valuesct) 
        #cumulativemt = np.cumsum(valuesmt) 
        # plot the cumulative function 
        plt.plot(basec[:-1], cumulativec/max(cumulativec), c='red',label = 'Cohabitaition') 
        plt.plot(basem[:-1], cumulativem/max(cumulativem), c='blue',label = 'Marriage') 
        #plt.plot(basect[:-1], cumulativect/max(cumulativect),linestyle='--', c='red',label = 'Cohabitaition-All') 
        #plt.plot(basemt[:-1], cumulativemt/max(cumulativemt),linestyle='--', c='blue',label = 'Marriage-All') 
        plt.legend(loc='best', fontsize='x-small',frameon=False,ncol=2)      
        plt.xlabel('Love Shock $\psi$')    
        plt.ylabel('Probability')  
        plt.savefig('psidist.pgf', bbox_inches = 'tight',pad_inches = 0)  
         
 
        ##########################################    
        # Distribution of Love by Education
        #################################
        #########   
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1) 
         
        # evaluate the histogram 
        valuescte, basecte = np.histogram(psi_check[(changec) & (educ=='e')], bins=1000) 
        valuesmte, basemte = np.histogram(psi_check[(changem) & (educ=='e')], bins=1000) 
        valuesctn, basectn = np.histogram(psi_check[(changec) & (educ=='n')], bins=1000) 
        valuesmtn, basemtn = np.histogram(psi_check[(changem) & (educ=='n')], bins=1000) 
        #evaluate the cumulative 

        cumulativecte = np.cumsum(valuescte) 
        cumulativemte = np.cumsum(valuesmte) 
        cumulativectn = np.cumsum(valuesctn) 
        cumulativemtn = np.cumsum(valuesmtn) 
        # plot the cumulative function 
        plt.plot(basecte[:-1], cumulativecte/max(cumulativecte), c='black',label = 'Cohabitaition - Co') 
        plt.plot(basemte[:-1], cumulativemte/max(cumulativemte), c='b',label = 'Marriage - Co') 
        plt.plot(basectn[:-1], cumulativectn/max(cumulativectn), c='y',linestyle='--',label = 'Cohabitaition - NoCo') 
        plt.plot(basemtn[:-1], cumulativemtn/max(cumulativemtn), c='r',linestyle='--',label = 'Marriage - NoCo') 
        plt.legend(loc='best', fontsize='x-small',frameon=False)  
        plt.xlabel('Love Shock $\psi$')    
        plt.ylabel('Probability')  
        plt.savefig('psidist_edu.pgf', bbox_inches = 'tight',pad_inches = 0) 
         
        ##########################################    
        # Distribution of Pareto Weight  
        ##########################################   
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1) 
         
         
        sns.kdeplot(theta_t[statecpsi], shade=True,shade_lowest=False,linewidth=0.01, color="r", bw=.05,label = 'Cohabitaition') 
        sns.kdeplot(theta_t[statempsi], shade=True,shade_lowest=False,linewidth=0.01, color="b", bw=.05,label = 'Marriage') 
        sns.kdeplot(theta_t[changec], color="r", bw=.05,label = 'Cohabitaition Beg') 
        sns.kdeplot(theta_t[changem], color="b", bw=.05,label = 'Marriage Beg')  
        plt.legend(loc='best', fontsize='x-small',frameon=False)   
        plt.xlabel('Female Pareto Weight')    
        plt.ylabel('Denisty')  
        plt.savefig('thtdist.pgf', bbox_inches = 'tight',pad_inches = 0) 
         
         
         

          
        ##########################################    
        # FLS: Marriage vs. cohabitation  
        ##########################################     
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1)    
             
 
         
        lg=3
        # create plot    
         
        plt.plot(np.array([0,1,2]), flsc_d,linewidth=1.5,color="b", label='Cohabitation---D')  
        plt.plot(np.array([0,1,2]), flsm_d,linewidth=1.5,color="k", label='Marriage---D')  
        plt.fill_between(np.array([0,1,2]), flsc_i[0,0:lg], flsc_i[1,0:lg],alpha=0.2,facecolor='b')   
        plt.fill_between(np.array([0,1,2]), flsm_i[0,0:lg], flsm_i[1,0:lg],alpha=0.2,facecolor='k')  
        plt.plot(np.array([0,1,2]), moments['flsc'], linestyle='--',linewidth=1.5,color="r", label='Cohabitation---S') 
        plt.plot(np.array([0,1,2]), moments['flsm'], linestyle='--',linewidth=1.5,color="m", label='Marriage---S')
        plt.ylabel('FLS')
        plt.xlabel('Age')
        plt.legend(loc='best', fontsize='x-small',frameon=False,ncol=2)     
        plt.xticks(np.arange(3), ('24-26','29-31','34-36'))
        plt.ylim(ymax=1.0,ymin=0.0) 
        plt.savefig('labor.pgf', bbox_inches = 'tight',pad_inches = 0)
        
        #plt.ylim(ymax=0.1)    
        #plt.xlim(xmax=1.0,xmin=0.0)    
         
        ##########################################    
        # FLS: Marriage vs. cohabitation  
        ##########################################     
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1)    
             
 
         
        lg=2 
        # create plot    
        plt.plot(np.array([25,35]), full_flsc['e']['e'], linestyle='--',linewidth=1.5,color="b", label='fls-ee-C')  
        plt.plot(np.array([25,35]), full_flsm['e']['e'], linewidth=1.5,color="b", label='fls-ee-M')  
        plt.plot(np.array([25,35]), full_flsc['e']['n'], linestyle='--',linewidth=1.5,color="r", label='fls-en-C')  
        plt.plot(np.array([25,35]), full_flsm['e']['n'], linewidth=1.5,color="r", label='fls-en-M')  
        plt.plot(np.array([25,35]), full_flsc['n']['e'], linestyle='--',linewidth=1.5,color="y", label='fls-ne-C')  
        plt.plot(np.array([25,35]), full_flsm['n']['e'], linewidth=1.5,color="y", label='fls-ne-M')  
        plt.plot(np.array([25,35]), full_flsc['n']['n'], linestyle='--',linewidth=1.5,color="g", label='fls-nn-C')  
        plt.plot(np.array([25,35]), full_flsm['n']['n'], linewidth=1.5,color="g", label='fls-nn-M')  
        plt.ylim(ymax=1.0,ymin=0.0) 
        plt.ylabel('FLS mar and coh')
        plt.xlabel('Age')
        plt.legend(loc='best', fontsize='x-small',frameon=False)  
        
        


    
       
        ##########################################    
        # FLS: Marriage vs. cohabitation  
        ##########################################  
        if len(iexo[0,:])>35:
            fig = plt.figure()    
            f6=fig.add_subplot(2,1,1)    
                 
     
            #Construct fls by first and last point in trans matrix shocks
            lmzz=[np.mean([((setup.all_indices(1,iexo[:,25]))[1]==0) & ((setup.all_indices(1,iexo[:,25]))[2]==0)  & (state[:,25]==2)])/np.mean(state[:,25]==2),
                  np.mean([((setup.all_indices(1,iexo[:,35]))[1]==0) & ((setup.all_indices(1,iexo[:,35]))[2]==0)  & (state[:,35]==2)])/np.mean(state[:,35]==2)]
    
            lmzm=[np.mean([((setup.all_indices(1,iexo[:,25]))[1]==0) & ((setup.all_indices(1,iexo[:,25]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,25]==2)])/np.mean(state[:,25]==2),
                  np.mean([((setup.all_indices(1,iexo[:,35]))[1]==0) & ((setup.all_indices(1,iexo[:,35]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,35]==2)])/np.mean(state[:,35]==2)]
             
            lmmz=[np.mean([((setup.all_indices(1,iexo[:,25]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,25]))[2]==0)  & (state[:,25]==2)])/np.mean(state[:,25]==2),
                  np.mean([((setup.all_indices(1,iexo[:,35]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,35]))[2]==0)  & (state[:,35]==2)])/np.mean(state[:,35]==2)]
    
            lmmm=[np.mean([((setup.all_indices(1,iexo[:,25]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,25]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,25]==2)])/np.mean(state[:,25]==2),
                  np.mean([((setup.all_indices(1,iexo[:,35]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,35]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,35]==2)])/np.mean(state[:,35]==2)]
            
            lczz=[np.mean([((setup.all_indices(1,iexo[:,25]))[1]==0) & ((setup.all_indices(1,iexo[:,25]))[2]==0)  & (state[:,25]==3)])/np.mean(state[:,35]==3),
                  np.mean([((setup.all_indices(1,iexo[:,35]))[1]==0) & ((setup.all_indices(1,iexo[:,35]))[2]==0)  & (state[:,35]==3)])/np.mean(state[:,35]==3)]
    
            lczm=[np.mean([((setup.all_indices(1,iexo[:,25]))[1]==0) & ((setup.all_indices(1,iexo[:,25]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,25]==3)])/np.mean(state[:,25]==3),
                  np.mean([((setup.all_indices(1,iexo[:,35]))[1]==0) & ((setup.all_indices(1,iexo[:,35]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,35]==3)])/np.mean(state[:,35]==3)]
             
            lcmz=[np.mean([((setup.all_indices(1,iexo[:,25]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,25]))[2]==0)  & (state[:,25]==3)])/np.mean(state[:,25]==3),
                  np.mean([((setup.all_indices(1,iexo[:,35]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,35]))[2]==0)  & (state[:,35]==3)])/np.mean(state[:,35]==3)]
    
            lcmm=[np.mean([((setup.all_indices(1,iexo[:,25]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,25]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,25]==3)])/np.mean(state[:,25]==3),
                  np.mean([((setup.all_indices(1,iexo[:,35]))[1]==(setup.pars['n_zf_t'][0]-1)) & ((setup.all_indices(1,iexo[:,35]))[2]==(setup.pars['n_zm_t'][0]-1))  & (state[:,35]==3)])/np.mean(state[:,35]==3)]
                                                                
            lg=2 
            # create plot    
            plt.plot(np.array([25,35]), lczz, linestyle='--',linewidth=1.5,color="b", label='00-C')  
            plt.plot(np.array([25,35]), lmzz, linewidth=1.5,color="b", label='00-M')  
            plt.plot(np.array([25,35]), lczm, linestyle='--',linewidth=1.5,color="r", label='01-C')  
            plt.plot(np.array([25,35]), lmzm, linewidth=1.5,color="r", label='01-M')  
            plt.plot(np.array([25,35]), lcmz, linestyle='--',linewidth=1.5,color="y", label='10-C')  
            plt.plot(np.array([25,35]), lmmz, linewidth=1.5,color="y", label='10-M')  
            plt.plot(np.array([25,35]), lcmm, linestyle='--',linewidth=1.5,color="g", label='11-C')  
            plt.plot(np.array([25,35]), lmmm, linewidth=1.5,color="g", label='11-M')  
            plt.ylim(ymin=0.0) 
            plt.ylabel('Share married and cohabiting by grid by grid of income')
            plt.xlabel('Age')
            plt.legend(loc='best', fontsize='x-small',frameon=False)  
            
              
            
        ##########################################################  
        # Histogram of Unilateral Divorce on Cohabitation Length  
        #############################################################  
        fig = plt.figure()   
        f6=fig.add_subplot(2,1,1)   
            
           
        # create plot   
        x=np.array([0.2,0.5,0.8])  
        y=np.array([haz_join,haz_mar,haz_sep])   
        yerr=y*0.0   
        plt.axhline(y=1.0,linewidth=0.1, color='r')   
        plt.errorbar(x, y, yerr=yerr, fmt='o', elinewidth=0.03)   
        plt.ylabel('Relative Hazard - Education')   
        plt.xticks(x, ["Overall Risk","Risk of Marriage","Risk of Separation"] )  
        #plt.ylim(ymax=1.2,ymin=0.7)   
        plt.xlim(xmax=1.0,xmin=0.0)   
        
        # ##########################################################  
        # # Histogram of Unilateral Divorce on Cohabitation Length  
        # #############################################################  
        # fig = plt.figure()   
        # f6=fig.add_subplot(2,1,1)   
            
           
        # # create plot   
        # x=np.array([0.2,0.5,0.8])  
        # y=np.array([haz_joinp,haz_marp,haz_sepp])   
        # yerr=y*0.0   
        # plt.axhline(y=1.0,linewidth=0.1, color='r')   
        # plt.errorbar(x, y, yerr=yerr, fmt='o', elinewidth=0.03)   
        # plt.ylabel('Relative Hazard - Education Partner')   
        # plt.xticks(x, ["Overall Risk","Risk of Marriage","Risk of Separation"] )  
        # #plt.ylim(ymax=1.2,ymin=0.7)   
        # plt.xlim(xmax=1.0,xmin=0.0)   
        
      
        
        
        ##########################################   
        # Divorce by Edu
        ##########################################   
        fig = plt.figure()   
        f6=fig.add_subplot(2,1,1)   
            
        

        # create plot   
        x=np.array([0.25,0.75])  
        y=np.array([beta_edu_d,beta_div_edu_s])   
        yerr=np.array([(beta_edu_i[1]-beta_edu_i[0])/2.0,0.0])   
        plt.axhline(y=1.0,linewidth=0.1, color='r')   
        plt.errorbar(x, y, yerr=yerr, fmt='o', elinewidth=0.03)   
        plt.ylabel('Divorce-Education')   
        plt.xticks(x, ["Data","Simulation"] )  
        plt.ylim(ymax=1.4)   
        plt.xlim(xmax=1.0,xmin=0.0)   
        
        
#        ##########################################   
#        # Divorce by Edu P  
#        ##########################################   
#        fig = plt.figure()   
#        f6=fig.add_subplot(2,1,1)   
#            
#        
#        beta_div_edup_d=1.0
#        beta_div_edup_i=np.array([1.0,1.0])
#        # create plot   
#        x=np.array([0.25,0.75])  
#        y=np.array([beta_div_edup_d,beta_div_edup_s])   
#        yerr=np.array([(beta_div_edup_i[1]-beta_div_edup_i[0])/2.0,0.0])   
#        plt.axhline(y=1.0,linewidth=0.1, color='r')   
#        plt.errorbar(x, y, yerr=yerr, fmt='o', elinewidth=0.03)   
#        plt.ylabel('Divorce-Education P')   
#        plt.xticks(x, ["Data","Simulation"] )  
#        plt.ylim(ymax=1.4)   
#        plt.xlim(xmax=1.0,xmin=0.0)   
     
          
        ##########################################   
        # Divorce by Edu
        ##########################################   
        fig = plt.figure()   
        f6=fig.add_subplot(2,1,1)   
            
        

        # create plot   
        x=np.array([0.25,0.75])  
        y=np.array([ratio_mar_d,ratio_mar])   
        yerr=np.array([(ratio_mari[1]-ratio_mari[0])/2.0,0.0])   
        plt.axhline(y=1.0,linewidth=0.1, color='r')   
        plt.errorbar(x, y, yerr=yerr, fmt='o', elinewidth=0.03)   
        plt.ylabel('Ratio No college marria/College marriage')   
        plt.xticks(x, ["Data","Simulation"] )  
        #plt.ylim(ymax=1.4)   
        #plt.xlim(xmax=1.0,xmin=0.0)   
        
        ##########################################   
        # Divorce Cost
        ##########################################   
        fig = plt.figure()   
        f6=fig.add_subplot(2,1,1)   
            
        

        # create plot   
        wage=np.linspace(0.001,2,1000)
        atax=np.exp(np.log(1.0-mdl.setup.div_costs.money_lost_m_ez)+(1.0-mdl.setup.div_costs.prog)*np.log(10.65*wage))/10.65
        atax=np.exp(np.log(1.0-mdl.setup.div_costs.money_lost_m_ez)+(1.0-mdl.setup.div_costs.prog)*np.log(wage))+0.75
        plt.plot(wage,wage,color='b')
        plt.plot(wage,atax,color='r')
        plt.ylim(ymax=2,ymin=0.001)   
        #plt.yscale('log')
        #plt.xscale('log')
        
        
        ############################################################ 
        # Graph at the beginning-
        ################################################################### 
         
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1) 
         
        base=np.linspace(0,5,6,dtype=np.int16)
        raw=np.array([1.0,1.21,0.82,0.79,0.58,0.46])
        predicted=np.array([1.0,1.11,0.84,0.78,0.76,0.72])
        # plot the cumulative function 
        plt.plot(base,raw, c='red',label = 'Raw Data') 
        plt.plot(base,predicted, c='blue',label = 'Model Prediction') 
        plt.legend(loc='best', fontsize='x-small',frameon=False,ncol=2)  
        plt.xlabel('Premarital Cohabitation Duration (Years)')    
        plt.ylabel('Rel. Haz. of Divorce')  
        plt.savefig('cohrel.pgf', bbox_inches = 'tight',pad_inches = 0)  
        
        
        ##########################################    
        # Put graphs together    
        ##########################################    
        #show()    
        for fig in range(1, plt.gcf().number + 1): ## will open an empty extra figure :(    
            pdf.savefig( fig )    
            
        pdf.close()    
        matplotlib.pyplot.close("all")    
         
           
    return moments   
   
class KeySet(object):
    def __init__(self, i, arr):
        self.i = i
        self.arr = arr
    def __hash__(self):
        return hash((self.i, hash(self.arr.tostring())))          
            
