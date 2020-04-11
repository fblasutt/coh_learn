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
import statsmodels.api as sm    
import statsmodels.formula.api as smf    
   
     
def moment(mdl_list,agents,agents_male,draw=True,validation=False):    
#This function compute moments coming from the simulation    
#Optionally it can also plot graphs about them. It is feeded with    
#matrixes coming from simulations    
     
 
 
    mdl=mdl_list[0] 
 
         
    #Import simulated values   
    state=agents.state  
    assets_t=mdl.setup.agrid_c[agents.iassets] 
    assets_t[agents.state<=1]=mdl.setup.agrid_s[agents.iassets[agents.state<=1]] 
    iexo=agents.iexo      
    theta_t=mdl.setup.thetagrid_fine[agents.itheta]    
    setup = mdl.setup   
    female=agents.is_female 
    cons=agents.c 
    consx=agents.x 
    labor=agents.ils_i 
    shks = agents.shocks_single_iexo  
    psi_check=np.zeros(state.shape) 
    shift_check=np.array((state==2),dtype=np.float32) 
    single=np.array((state==0),dtype=bool) 
    betag=mdl.setup.pars['beta_t'][0]**(np.linspace(1,len(state[0,:]),len(state[0,:]))-1) 
    betam=np.reshape(np.repeat(betag,len(state[:,0])),(len(state[:,0]),len(betag)),order='F') 
    #agegrid=np.reshape(agegridtemp,resha) 
     
    #Fill psi and ushift here 
    for i in range(len(state[0,:])): 
        psi_check[:,i]=((setup.exogrid.psi_t[i][(setup.all_indices(i,iexo[:,i]))[3]]))  
     
     
    psi_check[single]=0.0 
    state_psid=agents_male.state 
    labor_psid=agents_male.ils_i 
    iexo_psid=agents_male.iexo 
    change_psid=agents_male.policy_ind 
     
    if draw: 
        #Import values for female labor supply (simulated men only) 

        iexo_w=agents.iexo  
        labor_w=agents.ils_i 
        female_w=agents.is_female 
        divorces_w=agents.divorces 
        state_w=agents.state  
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
    if draw:
        if len(mdl_list) > 1: 
            welf_dec(mdl_list,agents) 
         
    ##########################################    
    #START COMPUTATION OF SIMULATED MOMENTS    
    #########################################    
       
         
    #Create a file with the age of the change foreach person    
    changep=agents.policy_ind   
        
         
    #Get states codes    
    state_codes = {name: i for i, name in enumerate(mdl.setup.state_names)}    
       
     ###########################################    
    #Sample selection    
    ###########################################    
        
    #Sample Selection to replicate the fact that    
    #in NSFH wave two cohabitning couples were    
    #excluded.    
    #Birth cohorts: 45-55    
    #Second wave of NLSFH:1992-1994.    
    #    
    #Assume that people are interviewd in 1993 and that age is uniformly    
    #distributed. Clearly we can adjust this later on.    
        
        
        
    #First cut the first two periods give new 'length'    
    assets_t=assets_t[:,:mdl.setup.pars['T']]    
    iexo=iexo[:,:mdl.setup.pars['T']]    
    state=state[:,:mdl.setup.pars['T']]    
    theta_t=theta_t[:,:mdl.setup.pars['T']]    
    female=female[:,:mdl.setup.pars['T']]    
    labor_psid=labor_psid[:,:mdl.setup.pars['T']] 
    iexo_psid=iexo_psid[:,:mdl.setup.pars['T']] 
     
    if draw: 
        iexo_w=iexo_w[:,:mdl.setup.pars['T']] 
        labor_w=labor_w[:,:mdl.setup.pars['T']] 
        change_psid=change_psid[:,:mdl.setup.pars['T']] 
        state_psid=state_psid[:,:mdl.setup.pars['T']] 
        female_w=female_w[:,:mdl.setup.pars['T']] 
        state_w=state_w[:,:mdl.setup.pars['T']] 
        assets_w=assets_w[:,:mdl.setup.pars['T']] 
        assetss_w=assetss_w[:,:mdl.setup.pars['T']] 
        theta_w=theta_w[:,:mdl.setup.pars['T']] 
        changep_w=changep_w[:,:mdl.setup.pars['T']] 
        divorces_w=divorces_w[:,:mdl.setup.pars['T']] 
       
        
    ####################################################################    
    #Now drop observation to mimic the actual data gathering process    
    ####################################################################    
        
    #Get distribution of age conditional on cohabiting on the second wave    
    with open('age_sw.pkl', 'rb') as file:    
        age_sw=pickle.load(file)    
            
    keep=(assets_t[:,0]>-1)    
       
    
    summa=0.0    
    summa1=0.0    
    for i in age_sw:    
        summa+=age_sw[i]    
        keep[int(summa1*len(state[:,0])/sum(age_sw.values())):int(summa*len(state[:,0])/sum(age_sw.values()))]=(state[int(summa1*len(state[:,0])/sum(age_sw.values())):int(summa*len(state[:,0])/sum(age_sw.values())),int((i-20)/mdl.setup.pars['py'])]!=3)    
          
        summa1+=age_sw[i]    
    
     
    state=state[keep,]     
    changep=changep[keep,]  
    female=female[keep,]  
    iexo=iexo[keep,] 
    assets_t=assets_t[keep,] 
    labor=labor[keep,] 
     
       
     
    ################################################################### 
    # Draw from simulated agents to match NSFH distribution 
    # according to the following stratas: 
    # 1) Age at unilateral divorce 
    # 2) Gender 
    #  
    ################################################################### 
     
    #Import the distribution from the data 
    with open('freq_nsfh.pkl', 'rb') as file:    
        freq_nsfh_data=pickle.load(file)   
     
    #value=mdl.V[0]['Female, single']['V'][0,iexo_w[:,0]] 
    #Make data compatible with current age 
    freq_nsfh_data['age_unid']=freq_nsfh_data['age_unid']-18.0 
    freq_nsfh_data.loc[freq_nsfh_data['age_unid']<=0.0,'age_unid']=0.0 
    freq_nsfh_data.loc[freq_nsfh_data['age_unid']>=900.0,'age_unid']=1000 
     
    #Drop if no change in law! 
    if np.all(changep==0): 
        freq_nsfh_data.loc[freq_nsfh_data['age_unid']<1910.0,'age_unid']=1000 
    
  
         
    freq_nsfh=freq_nsfh_data.groupby(['M2DP01','age_unid'])['SAMWT'].count() 
    #Create a Dataframe with simulated data to perform the draw 
    age_unid=np.argmax(changep,axis=1) 
    never=(changep[:,0]==0) & (age_unid[:]==0) 
    age_unid[never]=1000 
    age_unid[changep[:,-1]==0]=1000 
     
    fem=np.array(['FEMALE']*len(female)) 
    fem[female[:,0]==0]='MALE' 
     
    inde=np.linspace(1,len(fem),len(fem),dtype=np.int32) 
     
    ddd=np.stack((inde,age_unid,fem),axis=0).T 
    df=pd.DataFrame(data=ddd,columns=["Index","age","sex"],index=ddd[:,0]) 
    df['age']=df['age'].astype(np.float) 
    try:#if (len(df)>0) &  (setup.pars['py']==1):   
        sampletemp=strata_sample(["'sex'", "'age'"],freq_nsfh,frac=0.2,tsample=df,distr=True) 
        final2=df.merge(sampletemp,how='left',on='Index',indicator=True) 
         
        keep2=[False]*len(df) 
        keep2=(np.array(final2['_merge'])=='both') 
    except:#else: 
        keep2=[True]*len(df) 
     
    #Keep again for all relevant variables    
    state=state[keep2,]      
    changep=changep[keep2,]  
    female=female[keep2,] 
    iexo=iexo[keep2,] 
    assets_t=assets_t[keep2,] 
    labor=labor[keep2,] 
   
     
    #Initial distribution 
    prima=freq_nsfh/np.sum(freq_nsfh) 
     
    #Final distribution 
    final3=df[keep2] 
    final4=final3.groupby(['sex','age'])['sex'].count() 
    dopo=final4/np.sum(final4) 
     
    try: 
        print('The average deviation from actual to final ditribution is {:0.2f}%'.format(np.mean(abs(prima-dopo))*100)) 
    except: 
        print('No stratified sampling') 
    ###################################################################   
    #Get age we stop observing spells: this matters for hazards 
    ###################################################################   
    with open('age_sint.pkl', 'rb') as file:    
        age_sint=pickle.load(file)    
            
    aged=np.ones((state.shape))   
       
    
    summa=0.0   
    summa1=0.0   
    for i in age_sint:   
        summa+=age_sint[int(i)]   
        aged[int(summa1*len(aged[:])/sum(age_sint.values())):int(summa*len(aged[:])/sum(age_sint.values()))]=round((i-20)/mdl.setup.pars['py'],0)   
        summa1+=age_sint[int(i)]   
       
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
    time_end = -1*np.ones((N,nspells),dtype=np.bool)    
    sp_length = -1*np.ones((N,nspells),dtype=np.int16)    
    sp_person = -1*np.ones((N,nspells),dtype=np.int16)    
    is_unid = -1*np.ones((N,nspells),dtype=np.int16)    
    is_unid_end = -1*np.ones((N,nspells),dtype=np.int16)    
    is_unid_lim = -1*np.ones((N,nspells),dtype=np.int16)    
    n_spell = -1*np.ones((N,nspells),dtype=np.int16)    
    is_spell = np.zeros((N,nspells),dtype=np.bool)    
       
         
    state_beg[:,0] = 0 # THIS ASSUMES EVERYONE STARTS AS SINGLE   #TODO consistent with men stuff? 
    time_beg[:,0] = 0    
    sp_length[:,0] = 1    
    is_spell[:,0] = True    
    ispell = np.zeros((N,),dtype=np.int8)    
         
    for t in range(1,mdl.setup.pars['T']):    
        ichange = ((state[:,t-1] != state[:,t]))    
        sp_length[((~ichange)),ispell[((~ichange))]] += 1    
        #ichange = ((state[:,t-1] != state[:,t]) & (t<=aged[:,t]))    
        #sp_length[((~ichange) & (t<=aged[:,t])),ispell[((~ichange) & (t<=aged[:,t]))]] += 1    
             
        if not np.any(ichange): continue    
             
        did_end[ichange,ispell[ichange]] = True    
             
        is_spell[ichange,ispell[ichange]+1] = True    
        sp_length[ichange,ispell[ichange]+1] = 1 # if change then 1 year right    
        state_end[ichange,ispell[ichange]] = state[ichange,t]    
        sp_person[ichange,ispell[ichange]] = index[ichange]   
        time_end[ichange,ispell[ichange]] = t-1    
        state_beg[ichange,ispell[ichange]+1] = state[ichange,t]     
        time_beg[ichange,ispell[ichange]+1] = t    
        n_spell[ichange,ispell[ichange]+1]=ispell[ichange]+1   
        is_unid[ichange,ispell[ichange]+1]=changep[ichange,t]   
        is_unid_lim[ichange,ispell[ichange]+1]=changep[ichange,aged[ichange,0]]   
        is_unid_end[ichange,ispell[ichange]]=changep[ichange,t-1]   
           
             
        ispell[ichange] = ispell[ichange]+1    
             
             
    allspells_beg = state_beg[is_spell]    
    allspells_len = sp_length[is_spell]    
    allspells_end = state_end[is_spell] # may be -1 if not ended    
    allspells_timeb = time_beg[is_spell]   
    allspells_isunid=is_unid[is_spell]   
    allspells_isunidend=is_unid_end[is_spell]   
    allspells_isunidlim=is_unid_lim[is_spell]   
    allspells_person=sp_person[is_spell]   
    allspells_nspells=n_spell[is_spell]   
       
       
         
    # If the spell did not end mark it as ended with the state at its start    
    allspells_end[allspells_end==-1] = allspells_beg[allspells_end==-1]    
    allspells_isunidend[allspells_isunidend==-1] = allspells_isunidlim[allspells_isunidend==-1]   
    allspells_nspells[allspells_nspells==-1]=0   
    allspells_nspells=allspells_nspells+1   
        
    #Use this to construct hazards   
    spells = np.stack((allspells_beg,allspells_len,allspells_end),axis=1)    
       
    #Use this for empirical analysis   
    spells_empirical=np.stack((allspells_beg,allspells_timeb,allspells_len,allspells_end,allspells_nspells,allspells_isunid,allspells_isunidend),axis=1)   
    is_coh=((spells_empirical[:,0]==3) & (spells_empirical[:,5]==spells_empirical[:,6]))   
    spells_empirical=spells_empirical[is_coh,1:6]   
       
      
         
         
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
    isrel = np.zeros((N,),dtype=np.int8)    
        
    for t in range(2,mdl.setup.pars['Tret']-int(6/mdl.setup.pars['py'])):    
            
        irchange = ((state[:,t-1] != state[:,t]) & ((state[:,t-1]==0) | (state[:,t-1]==1)))    
            
        if not np.any(irchange): continue    
        
        rel_end[irchange,isrel[irchange]]=state[irchange,t]    
        rel_age[irchange,isrel[irchange]]=t    
        rel_unid[irchange,isrel[irchange]]=changep[irchange,t]    
        rel_number[irchange,isrel[irchange]]=isrel[irchange]+1    
        rel_sex[irchange,isrel[irchange]]=female[irchange,0]  
            
        isrel[irchange] = isrel[irchange]+1    
        
    #Get the final Variables    
    allrel_end=rel_end[(rel_end!=-1)]    
    allrel_age=rel_age[(rel_age!=-1)]    
    allrel_uni=rel_unid[(rel_unid!=-1)]  
    allrel_sex=rel_sex[(rel_unid!=-1)]  
    allrel_number=rel_number[(rel_number!=-1)]    
        
    #Get whetehr marraige    
    allrel_mar=np.zeros((allrel_end.shape))    
    allrel_mar[(allrel_end==2)]=1    
        
    #Create a Pandas Dataframe    
    data_rel=np.array(np.stack((allrel_mar,allrel_age,allrel_uni,allrel_number,allrel_sex),axis=0).T,dtype=np.float64)    
    data_rel_panda=pd.DataFrame(data=data_rel,columns=['mar','age','uni','rnumber','sex'])    
                       
    
        
    #Regression    
    if np.var(data_rel_panda['uni'])>0.0001: 
        try:    
            FE_ols = smf.ols(formula='mar ~ uni+C(age)+C(sex)', data = data_rel_panda.dropna()).fit()    
            beta_unid_s=FE_ols.params['uni']    
        except:    
            print('No data for unilateral divorce regression...')    
            beta_unid_s=0.0  
    else: 
        beta_unid_s=0.0  
         
        
        
    moments['beta unid']=beta_unid_s     
       
    ###################################################   
    # Second regression for the length of cohabitation   
    ###################################################   
    if draw:
        data_coh_panda=pd.DataFrame(data=spells_empirical,columns=['age','duration','end','rel','uni'])    
           
        if np.var(data_rel_panda['uni'])>0.0001: 
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
                data_coh_panda1=data_coh_panda.drop(['end'], axis=1)   
                cox_join=cph.fit(data_coh_panda1, duration_col='duration', event_col='endd')   
                haz_join=cox_join.hazard_ratios_['uni']   
                   
                #Cox where risk is marriage   
                data_coh_panda['endd']=0.0   
                data_coh_panda.loc[data_coh_panda['end']==2.0,'endd']=1.0   
                data_coh_panda2=data_coh_panda.drop(['end'], axis=1)   
                cox_mar=cph.fit(data_coh_panda2, duration_col='duration', event_col='endd')   
                haz_mar=cox_mar.hazard_ratios_['uni']   
                   
                #Cox where risk is separatio   
                data_coh_panda['endd']=0.0   
                data_coh_panda.loc[data_coh_panda['end']==0.0,'endd']=1.0   
                data_coh_panda3=data_coh_panda.drop(['end'], axis=1)   
                cox_sep=cph.fit(data_coh_panda3, duration_col='duration', event_col='endd')   
                haz_sep=cox_sep.hazard_ratios_['uni']   
                   
            except:    
                print('No data for unilateral divorce regression...')    
                haz_sep=1.0  
                haz_join=1.0  
                haz_mar=1.0  
        else: 
            print('No data for unilateral divorce regression...')    
            haz_sep=1.0  
            haz_join=1.0  
            haz_mar=1.0  
             
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
         
    #Transform hazards pooling moments 
    mdl.setup.pars['ty']=2 
    if mdl.setup.pars['ty']>1: 
        #Divorce 
        hazdp=list() 
        pop=1 
        for i in range(int(mdl.setup.pars['T']/(mdl.setup.pars['ty']))): 
            haz1=hazd[mdl.setup.pars['ty']*i]*pop 
            haz2=hazd[mdl.setup.pars['ty']*i+1]*(pop-haz1) 
            hazdp=[(haz1+haz2)/pop]+hazdp  
            pop=pop-(haz1+haz2) 
        hazdp.reverse()    
        hazdp=np.array(hazdp).T  
        hazd=hazdp 
             
        #Separation and Marriage 
        hazsp=list() 
        hazmp=list() 
        pop=1 
        for i in range(int(mdl.setup.pars['T']/(mdl.setup.pars['ty']))): 
            hazs1=hazs[mdl.setup.pars['ty']*i]*pop 
            hazm1=hazm[mdl.setup.pars['ty']*i]*pop 
             
            hazs2=hazs[mdl.setup.pars['ty']*i+1]*(pop-hazs1-hazm1) 
            hazm2=hazm[mdl.setup.pars['ty']*i+1]*(pop-hazs1-hazm1) 
            hazsp=[(hazs1+hazs2)/pop]+hazsp 
            hazmp=[(hazm1+hazm2)/pop]+hazmp 
            pop=max(pop-(hazs1+hazs2+hazm1+hazm2),0.000001) 
             
        hazsp.reverse()    
        hazsp=np.array(hazsp).T  
        hazs=hazsp 
         
        hazmp.reverse()    
        hazmp=np.array(hazmp).T  
        hazm=hazmp 
         
    moments['hazard sep'] = hazs    
    moments['hazard div'] = hazd    
    moments['hazard mar'] = hazm    
        
    
     
         
    #Singles: Marriage vs. cohabitation transition    
    #spells_s=np.append(spells_Femalesingle,spells_Malesingle,axis=0)    
    spells_s =all_spells['Female, single']    
    cond=spells_s[:,2]>1    
    spells_sc=spells_s[cond,2]    
    condm=spells_sc==2    
    sharem=len(spells_sc[condm])/max(len(spells_sc),0.0001)    
       
       
    #Cut the first two periods give new 'length'    
    lenn=mdl.setup.pars['T']-mdl.setup.pars['Tbef']    
    assets_t=assets_t[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']]    
    iexo=iexo[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']]    
    state=state[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']]    
    theta_t=theta_t[:,mdl.setup.pars['Tbef']:mdl.setup.pars['T']]    
         
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
             
         
             
    moments['flsm'] = flsm    
    moments['flsc'] = flsc    
     
    ################## 
    #Sample Selection# 
    ################# 
     
    #Import the distribution from the data 
    with open('freq_psid_tot.pkl', 'rb') as file:    
        freq_psid_tot_data=pickle.load(file)   
         
    #Import Get when in a couple and reshape accordingly 
    resha=len(change_psid[0,:])*len(change_psid[:,0]) 
    state_totl=np.reshape(state_psid,resha) 
    incouple= (state_totl==2) | (state_totl==3) 
    incoupler=np.reshape(incouple,resha) 
     
    #Define main variables 
    ctemp=change_psid.copy() 
    change_psid2=np.reshape(ctemp,resha) 
    agetemp=np.linspace(1,len(change_psid[0,:]),len(change_psid[0,:])) 
    agegridtemp=np.reshape(np.repeat(agetemp,len(change_psid[:,0])),(len(change_psid[:,0]),len(agetemp)),order='F') 
    agegrid=np.reshape(agegridtemp,resha) 
     
    #Keep all those guys only if the are men and in a relatioinship 
    #TODO 
     
    #Make data compatible with current age. 
    freq_psid_tot_data['age']=freq_psid_tot_data['age']-18.0 
    freq_psid_tot_data.loc[freq_psid_tot_data['age']<0.0,'age']=0.0 
     
    #Drop if no change in law! 
    if np.all(changep==0): 
        #freq_psid_tot_data.loc[freq_psid_tot_data['age']<1910.0,'age']=1000 
        freq_psid_tot_data['unid']=0 
    
         
    freq_psid_tot_data2=freq_psid_tot_data.groupby(['age','unid'])['age'].count() 
     
    #Create a Dataframe with simulated data to perform the draw 
    inde=np.linspace(1,resha,resha,dtype=np.int32) 
     
    ddd2=np.stack((inde[incoupler],agegrid[incoupler],change_psid2[incoupler]),axis=0).T 
    df_psidt=pd.DataFrame(data=ddd2,columns=["Index","age","unid"],index=ddd2[:,0]) 
    df_psidt['age']=df_psidt['age'].astype(np.float) 
     
    try:#if (len(df_psidt)>0) & (setup.pars['py']==1) & (max(df_psidt['unid'])>0.9) & (min(df_psidt['unid'])<0.9): 
        sampletemp=strata_sample(["'age'", "'unid'"],freq_psid_tot_data2,frac=0.1,tsample=df_psidt,distr=True) 
        final2t=df_psidt.merge(sampletemp,how='left',on='Index',indicator=True) 
         
        keep3=[False]*len(df_psidt) 
        keep3=(np.array(final2t['_merge'])=='both') 
         
        #TODO assign labor according to stuff above 
        #Keep again for all relevant variables 
     
         
        #Initial distribution 
        prima_psid_tot=freq_psid_tot_data2/np.sum(freq_psid_tot_data2) 
         
        #Final distribution 
        final3=df_psidt[keep3] 
        final4=final3.groupby(['age','unid'])['age'].count() 
        dopo_psid_tot=final4/np.sum(final4) 
         
         
        print('The average deviation from actual to final psid_tot ditribution is {:0.2f}%'.format(np.mean(abs(prima_psid_tot-dopo_psid_tot))*100)) 
          
    except:#else: 
        keep3=[True]*len(df_psidt) 
    ############ 
    #Average FLS 
    ############ 
     
    state_totl=state_totl[incoupler][keep3] 
    labor_totl=np.reshape(labor_psid,resha) 
    labor_totl=labor_totl[incoupler][keep3] 
    mean_fls=0.0  
    pick=((state_totl[:]==2)  | (state_totl[:]==3))  
    if pick.any():mean_fls=np.array(setup.ls_levels)[labor_totl[pick]].mean()  
      
    moments['mean_fls'] = mean_fls  
     
    ###########################################    
    #Moments: wage   Ratio 
    ###########################################    
    iexo_totl=np.reshape(iexo_psid,resha) 
    iexo_totl=iexo_totl[incoupler][keep3] 
    age_w=np.array(agegrid[incoupler][keep3],dtype=np.int16)-1 
    wagem1=np.array(setup.pars['m_wage_trend'])[age_w][...,None]+np.array(setup.exogrid.zm_t)[age_w] 
    index=np.reshape(np.repeat(setup.all_indices(0,iexo_totl)[2],setup.pars['n_zm_t'][0]),(len(iexo_totl),setup.pars['n_zm_t'][0]),order='C') 
    maskp=np.reshape(np.repeat(np.linspace(0,setup.pars['n_zm_t'][0]-1,setup.pars['n_zm_t'][0]),len(index)), 
                      (index.shape),order='F') 
    wagem=wagem1[maskp==index] 
     
    moments['wage_ratio']=np.mean(wagem[state_totl==2])-np.mean(wagem[state_totl==3]) 
     
     
    ################### 
    #Sample Selection 
    ################### 
     
    #Import the distribution from the data 
    with open('freq_psid_par.pkl', 'rb') as file:    
        freq_psid_par_data=pickle.load(file)   
         
    #Import Get when in a couple and reshape accordingly 
    resha=len(change_psid[0,:])*len(change_psid[:,0]) 
    state_par=np.reshape(state_psid,resha) 
    incouplep= (state_par==2) | (state_par==3) 
    incouplepm= np.zeros(incouplep.shape) 
    incouplepm[np.where(state_par[incouplep]==2)[0]]=1 
    incouplerp=np.reshape(incouplep,resha) 
    incouplepm2=np.reshape(incouplepm,resha) 
     
 
     
    #Define main variables 
    ctemp=change_psid.copy() 
    change_psid3=np.reshape(ctemp,resha) 
     
    #Keep all those guys only if the are men and in a relatioinship 
    #TODO 
     
    #Make data compatible with current age. 
    freq_psid_par_data['age']=freq_psid_par_data['age']-18.0 
    freq_psid_par_data.loc[freq_psid_par_data['age']<0.0,'age']=0.0 
     
    #Drop if no change in law! 
    if np.all(changep==0): 
        #freq_psid_par_data.loc[freq_psid_par_data['age']<1910.0,'age']=1000 
        freq_psid_par_data['unid']=0 
         
  
         
    freq_psid_par_data2=freq_psid_par_data.groupby(['age','unid','mar'])['age'].count() 
 
     
    ddd3=np.stack((inde[incouplerp],agegrid[incouplerp],change_psid3[incouplerp],incouplepm2[incouplerp]),axis=0).T 
    
    df_psidp=pd.DataFrame(data=ddd3,columns=["Index","age","unid","mar"],index=ddd3[:,0]) 
    df_psidp['age']=df_psidp['age'].astype(np.float) 
     
    try:#if (len(df_psidp)>0) &  (setup.pars['py']==1) & (max(df_psidp['unid'])>0.9) & (min(df_psidp['unid'])<0.9):   
        sampletempp=strata_sample(["'age'", "'unid'", "'mar'"],freq_psid_par_data2,frac=0.02,tsample=df_psidp,distr=True) 
        final2p=df_psidp.merge(sampletempp,how='left',on='Index',indicator=True) 
         
        keep4=[False]*len(df_psidp) 
        keep4=(np.array(final2p['_merge'])=='both') 
         
        #TODO assign labor according to stuff above 
        #Keep again for all relevant variables 
     
         
        #Initial distribution 
        prima_psid_par=freq_psid_par_data2/np.sum(freq_psid_par_data2) 
         
        #Final distribution 
        final3p=df_psidp[keep4] 
        final4p=final3p.groupby(['age','unid','mar'])['age'].count() 
        dopo_psid_par=final4p/np.sum(final4p) 
         
         
        print('The average deviation from actual to final psid_tot ditribution is {:0.2f}%'.format(np.mean(abs(prima_psid_par-dopo_psid_par))*100)) 
          
    except:#else: 
        keep4=[True]*len(df_psidp) 
         
    ################ 
    #Ratio of fls  
    ############### 
    ages=agegrid[incouplerp][keep4] 
    state_par=state_par[incouplerp][keep4] 
    labor_par=np.reshape(labor_psid,resha) 
    labor_par=labor_par[incouplerp][keep4] 
     
    mean_fls_m=0.0  
    picky=(state_par[:]==2) & (ages<=15) 
    picko=(state_par[:]==2) & (ages>15) 
    mean_fls_m=np.zeros((2)) 
    if picky.any():mean_fls_m[0]=np.array(setup.ls_levels)[labor_par[picky]].mean()  
    if picko.any():mean_fls_m[1]=np.array(setup.ls_levels)[labor_par[picko]].mean()  
        
    mean_fls_c=0.0  
    picky=(state_par[:]==3) & (ages<=15) 
    picko=(state_par[:]==3) & (ages>15) 
    mean_fls_c=np.zeros((2)) 
    if picky.any():mean_fls_c[0]=np.array(setup.ls_levels)[labor_par[picky]].mean()  
    if picko.any():mean_fls_c[1]=np.array(setup.ls_levels)[labor_par[picko]].mean()  
      
    small=mean_fls_c<0.0001*np.ones((2)) 
    mean_fls_c[small]=0.0001*np.ones((2))[small] 

    moments['fls_ratio']=[min(mean_fls_m[0]/mean_fls_c[0],2.0),min(mean_fls_m[1]/mean_fls_c[1],2.0)]
     
    grid=np.linspace(5,35,31,dtype=np.int16) 
    storem=np.zeros(grid.shape) 
    storec=np.zeros(grid.shape) 
    for i in range(len(grid)): 
        storem[i]=np.mean(labor_par[(state_par==2) & (ages==grid[i])]) 
        storec[i]=np.mean(labor_par[(state_par==3) & (ages==grid[i])]) 
         
         
     
     
     
     
    ########################################################### 
    #Ever MArried and Cohabited 
    ######################################################### 
         
    relt=np.zeros((len(state_codes),lenn))    
    relt1=np.zeros((len(state_codes),lenn))    
        
    for ist,sname in enumerate(state_codes):  
     
        for t in range(lenn):    
                   
                 
             
            #Arrays for preparation    
            is_state = (np.any(state[:,:t+1]==ist,1))           
            is_state1 = (state[:,t]==ist)    
            
            if not (np.any(is_state) or np.any(is_state1)): continue    
             
          
            #Relationship over time    
            relt[ist,t]=np.sum(is_state)    
            relt1[ist,t]=np.sum(is_state1)    
             
                  
    #Now, before saving the moments, take interval of 5 years    
    # if (mdl.setup.pars['Tret']>=mdl.setup.pars['Tret']):            
    reltt=relt[:,:mdl.setup.pars['Tret']-mdl.setup.pars['Tbef']+1]    
    years=np.linspace(20,50,7)    
    years_model=np.linspace(20,50,int(30/mdl.setup.pars['py']))    
        
    #Find the right entries for creating moments    
    pos=list()    
    for j in range(len(years)):    
        pos=pos+[np.argmin(np.abs(years_model-years[j]))]    
        
    #Approximation if more than 5 years in one period    
    if len(pos)<7:    
        for i in range(7-len(pos)):    
            pos=pos+[pos[-1]]    
    pos=np.array(pos)    
        
        
        
    reltt=reltt[:,pos]    
            
    moments['share single'] = reltt[0,:]/N    
    moments['share mar'] = reltt[2,:]/N    
    moments['share coh'] = reltt[3,:]/N    
     
     
    ############################################## 
    #Sample selection for divorce by income 
    ############################################ 
     
     #Import the distribution from the data 
    with open('freq_psid_div.pkl', 'rb') as file:    
        freq_psid_div_data=pickle.load(file)   
         
         
     
    #Define main variables 
    ctemp=change_psid.copy() 
    change_psid4=np.reshape(ctemp,resha) 
     
   
    #Keep all those guys only if the are men and in a relatioinship 
    #TODO 
     
    
    #Drop if no change in law! 
    if np.all(changep==0): 
        #freq_psid_tot_data.loc[freq_psid_tot_data['age']<1910.0,'age']=1000 
        freq_psid_div_data['unid']=0 
    
         
    freq_psid_div_data2=freq_psid_div_data.groupby(['age','unid'])['age'].count() 
     
    #Create a Dataframe with simulated data to perform the draw 
    inde=np.linspace(1,resha,resha,dtype=np.int32) 
     
    ddd4=np.stack((inde,agegrid,change_psid4),axis=0).T 
    df_psidd=pd.DataFrame(data=ddd4,columns=["Index","age","unid"],index=ddd4[:,0]) 
    df_psidd['age']=df_psidd['age'].astype(np.float) 
     
    try:#if (len(df_psidd)>0) & (setup.pars['py']==1)   & (max(df_psidd['unid'])>0.9) & (min(df_psidd['unid'])<0.9):   
        sampletemp=strata_sample(["'age'", "'unid'"],freq_psid_div_data2,frac=0.1,tsample=df_psidd,distr=True) 
        final2d=df_psidd.merge(sampletemp,how='left',on='Index',indicator=True) 
         
        keep5=[False]*len(df_psidd) 
        keep5=(np.array(final2d['_merge'])=='both') 
         
        #TODO assign labor according to stuff above 
        #Keep again for all relevant variables 
     
         
        #Initial distribution 
        prima_psid_div=freq_psid_div_data2/np.sum(freq_psid_div_data2) 
         
        #Final distribution 
        final3=df_psidd[keep5] 
        final4=final3.groupby(['age','unid'])['age'].count() 
        dopo_psid_div=final4/np.sum(final4) 
         
         
        print('The average deviation from actual to final psid_div ditribution is {:0.2f}%'.format(np.mean(abs(prima_psid_div-dopo_psid_div))*100)) 
          
    except:#else: 
        keep5=[True]*len(df_psidd) 
         
 
 
    ################################################## 
    #DIVORCE BY INCOME 
    #################################################  
     
    #Wage 
    iexo_div=np.reshape(iexo_psid,resha) 
    iexo_div=iexo_div 
    stated=np.reshape(state_psid,resha) 
    age_w=np.array(agegrid,dtype=np.int16)-1 
    wagem1=np.array(setup.pars['m_wage_trend'])[age_w][...,None]+np.array(setup.exogrid.zm_t)[age_w] 
    indexm=np.reshape(np.repeat(setup.all_indices(0,iexo_div)[2],setup.pars['n_zm_t'][0]),(len(iexo_div),setup.pars['n_zm_t'][0]),order='C') 
    indexs=np.reshape(np.repeat(setup.all_indices(0,iexo_div)[0],setup.pars['n_zm_t'][0]),(len(iexo_div),setup.pars['n_zm_t'][0]),order='C') 
    maskp=np.reshape(np.repeat(np.linspace(0,setup.pars['n_zm_t'][0]-1,setup.pars['n_zm_t'][0]),len(indexs)), (indexs.shape),order='F') 
                      
    indexs[indexs>setup.pars['n_zm_t'][0]-1]=0 
    wagem=wagem1[maskp==indexs] 
    wagem[stated>1]=wagem1[maskp==indexm][stated>1] 
    
     
    #Marital Status 
    married=(state_psid==2) 
    agegridtemp=agegridtemp.copy()-1 
    agem=np.argmax(married,axis=1) 
    neverm=(agem==0) 
    agem[neverm]=9999 
    divo1=(state_psid==1) & (agegridtemp>agem[...,None]) 
    divo=np.reshape(divo1,resha)[keep5] 
    married=np.reshape(married,resha)[keep5] 
     
    #Poor rich indicator 
    keep5m=np.reshape(keep5,(int(len(keep5)/len(iexo_psid[0,:])),len(iexo_psid[0,:]))) 
    wageinter=np.reshape(wagem,(int(len(wagem)/len(iexo_psid[0,:])),len(iexo_psid[0,:]))) 
    wageinter[(~keep5m)]=None 
    mincome1=np.nanmedian(wageinter,axis=0) 
    mincome2=np.repeat(mincome1,len(iexo_psid[:,0]),axis=0) 
    mincome3=np.reshape(mincome2,(int(len(wagem)/len(iexo_psid[0,:])),len(iexo_psid[0,:])),order='F') 
    mincome=np.reshape(mincome3,wagem.shape)[keep5] 
    sq=(wagem[keep5]==mincome) 
    even=(np.random.random_sample(mincome.shape)>0.5)#np.repeat((np.linspace(1,len(mincome3[:,0]),len(mincome3[:,0]),dtype=np.int32)%2==0),len(mincome3[0,:])) 
    lq=(wagem[keep5]<mincome) | ((even) & (sq)) 
    uq=(wagem[keep5]>mincome) | ((~even) & (sq)) 
     
    moments['div_ratio']=(np.sum(divo[uq])/np.sum(married[uq]))/(np.sum(divo[lq])/np.sum(married[lq])) 
     
     
    ################################################## 
    #EVERYTHING BELOW IS NOT NECESSARY FOR MOMENTS 
    ################################################# 
     
    if draw:  
         
        #Update N to the new sample size    
        #N=len(state_w)    
     
        ###########################################    
        #Moments: Variables over Age    
        ###########################################    
         
        ifemale2=(female_w[:,0]==1) 
        imale2=(female_w[:,0]==0) 
        ass_rel=np.zeros((len(state_codes),lenn,2))    
        inc_rel=np.zeros((len(state_codes),lenn,2))  
        log_inc_rel=np.zeros((2,len(state_w)))  
         
        #Create wages 
        wage_f=np.zeros(state_w.shape) 
        wage_m=np.zeros(state_w.shape) 
         
        wage_fc=np.zeros(len(state_w[0,:])) 
        wage_mc=np.zeros(len(state_w[0,:])) 
        wage_fm=np.zeros(len(state_w[0,:])) 
        wage_mm=np.zeros(len(state_w[0,:])) 
         
        wage_f2=np.zeros(state_w.shape) 
        wage_m2=np.zeros(state_w.shape) 
        wage_fp=np.zeros(state_w.shape) 
        wage_mp=np.zeros(state_w.shape) 
         
        wage_fpc=np.zeros(len(state_w[0,:])) 
        wage_mpc=np.zeros(len(state_w[0,:])) 
        wage_fpm=np.zeros(len(state_w[0,:])) 
        wage_mpm=np.zeros(len(state_w[0,:])) 
         
        wage_fs=np.zeros(len(state_w[0,:])) 
        wage_ms=np.zeros(len(state_w[0,:])) 
         
        var_wage_fc=np.zeros(len(state_w[0,:])) 
        var_wage_mc=np.zeros(len(state_w[0,:])) 
        var_wage_fm=np.zeros(len(state_w[0,:])) 
        var_wage_mm=np.zeros(len(state_w[0,:])) 
         
        var_wage_fpc=np.zeros(len(state_w[0,:])) 
        var_wage_mpc=np.zeros(len(state_w[0,:])) 
        var_wage_fpm=np.zeros(len(state_w[0,:])) 
        var_wage_mpm=np.zeros(len(state_w[0,:])) 
         
        #For assets 
        assets_fc=np.zeros(len(state_w[0,:])) 
        assets_fm=np.zeros(len(state_w[0,:])) 
        assets_mc=np.zeros(len(state_w[0,:])) 
        assets_mm=np.zeros(len(state_w[0,:])) 
         
        assets_fpc=np.zeros(len(state_w[0,:])) 
        assets_fpm=np.zeros(len(state_w[0,:])) 
        assets_mpc=np.zeros(len(state_w[0,:])) 
        assets_mpm=np.zeros(len(state_w[0,:])) 
         
        var_assets_fc=np.zeros(len(state_w[0,:])) 
        var_assets_fm=np.zeros(len(state_w[0,:])) 
        var_assets_mc=np.zeros(len(state_w[0,:])) 
        var_assets_mm=np.zeros(len(state_w[0,:])) 
         
        var_assets_fpc=np.zeros(len(state_w[0,:])) 
        var_assets_fpm=np.zeros(len(state_w[0,:])) 
        var_assets_mpc=np.zeros(len(state_w[0,:])) 
        var_assets_mpm=np.zeros(len(state_w[0,:])) 
         
        corr_ass_sepm=np.zeros(len(state_w[0,:])) 
        corr_ass_sepf=np.zeros(len(state_w[0,:])) 
        share_ass_sepm=np.zeros(len(state_w[0,:])) 
        share_ass_sepf=np.zeros(len(state_w[0,:])) 
        mcorr_ass_sepm=np.zeros(len(state_w[0,:])) 
        mcorr_ass_sepf=np.zeros(len(state_w[0,:])) 
        mshare_ass_sepm=np.zeros(len(state_w[0,:])) 
        mshare_ass_sepf=np.zeros(len(state_w[0,:])) 
         
         
        psis=np.zeros(state_w.shape) 
        ifemale=(female_w[:,0]==1) 
        imale=(female_w[:,0]==0) 
 
        for i in range(len(state_w[0,:])): 
             
            #For Income 
            singlef=(ifemale) & (state_w[:,i]==0) 
            singlem=(imale) & (state_w[:,i]==1)        
            nsinglef=(ifemale) & (state_w[:,i]>=2) 
            nsinglem=(imale) & (state_w[:,i]>=2) 
            nsinglefc=(ifemale) & (state_w[:,i]==3) 
            nsinglemc=(imale) & (state_w[:,i]==3) 
            nsinglefm=(ifemale) & (state_w[:,i]==2) 
            nsinglemm=(imale) & (state_w[:,i]==2) 
             
            singlef2=(ifemale2) & (state_w[:,i]==0) 
            singlem2=(imale2) & (state_w[:,i]==1)        
            nsinglef2=(ifemale2) & (state_w[:,i]>=2) 
            nsinglem2=(imale2) & (state_w[:,i]>=2) 
             
            #For assets 
            cohf=(ifemale) & (state_w[:,i]==3) & (state_w[:,max(i-1,0)]<=1) 
            marf=(ifemale) & (state_w[:,i]==2) & (state_w[:,max(i-1,0)]<=1) 
            cohm=(imale) & (state_w[:,i]==3) & (state_w[:,max(i-1,0)]<=1) 
            marm=(imale) & (state_w[:,i]==2) & (state_w[:,max(i-1,0)]<=1) 
             
            acm=(imale) & (state_w[:,i]==1) & (state_w[:,max(i-1,0)]==3) 
            acf=(ifemale) & (state_w[:,i]==0) & (state_w[:,max(i-1,0)]==3) 
            macm=(imale) & (state_w[:,i]==1) & (state_w[:,max(i-1,0)]==2) #& (state_w[:,max(i-2,0)]==2) 
            macf=(ifemale) & (state_w[:,i]==0) & (state_w[:,max(i-1,0)]==2)# & (state_w[:,max(i-2,0)]==2) 
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
         
            #Aggregate Income 
            if np.any(nsinglef):wage_f[nsinglef,i]=np.exp(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])])[nsinglef] 
            if np.any(nsinglem):wage_m[nsinglem,i]=np.exp(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])])[nsinglem] 
            if np.any(singlef):wage_f[singlef,i]=np.exp(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][iexo_w[singlef,i]])  
            if np.any(singlem):wage_m[singlem,i]=np.exp(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][iexo_w[singlem,i]])  
            if np.any(nsinglef):wage_mp[nsinglef,i]=np.exp(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])])[nsinglef] 
            if np.any(nsinglem):wage_fp[nsinglem,i]=np.exp(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])])[nsinglem] 
            if np.any(marf):psis[:,i]=((setup.exogrid.psi_t[i][(setup.all_indices(i,iexo_w[:,i]))[3]])) 
             
            #Single only Income 
            wage_fs[i]=np.mean(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][iexo_w[singlef,i]] ) 
            wage_ms[i]=np.mean(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][iexo_w[singlem,i]] ) 
             
            #Cohabitation Income 
            if np.any(nsinglefc):wage_fc[i]=np.mean(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])][nsinglefc]) 
            if np.any(nsinglemc):wage_mc[i]=np.mean(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])][nsinglemc]) 
            if np.any(nsinglefc):wage_mpc[i]=np.mean(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])][nsinglefc]) 
            if np.any(nsinglemc):wage_fpc[i]=np.mean(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])][nsinglemc]) 
     
     
            if np.any(nsinglefc):var_wage_fc[i]=np.var(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])][nsinglefc]) 
            if np.any(nsinglemc):var_wage_mc[i]=np.var(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])][nsinglemc]) 
            if np.any(nsinglefc):var_wage_mpc[i]=np.var(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])][nsinglefc]) 
            if np.any(nsinglemc):var_wage_fpc[i]=np.var(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])][nsinglemc]) 
     
            #Marriage Income 
            if np.any(nsinglefm):wage_fm[i]=np.mean(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])][nsinglefm]) 
            if np.any(nsinglemm):wage_mm[i]=np.mean(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])][nsinglemm]) 
            if np.any(nsinglefm):wage_mpm[i]=np.mean(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])][nsinglefm]) 
            if np.any(nsinglemm):wage_fpm[i]=np.mean(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])][nsinglemm]) 
     
            if np.any(nsinglefm):var_wage_fm[i]=np.var(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])][nsinglefm]) 
            if np.any(nsinglemm):var_wage_mm[i]=np.var(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])][nsinglemm]) 
            if np.any(nsinglefm):var_wage_mpm[i]=np.var(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])][nsinglefm]) 
            if np.any(nsinglemm):var_wage_fpm[i]=np.var(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])][nsinglemm]) 
                       
              
            #For income process validation 
            wage_f2[nsinglef2,i]=np.exp(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])])[nsinglef2] 
            wage_m2[nsinglem2,i]=np.exp(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])])[nsinglem2] 
            wage_m2[nsinglef2,i]=np.exp(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][((setup.all_indices(i,iexo_w[:,i]))[2])])[nsinglef2] 
            wage_f2[nsinglem2,i]=np.exp(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][((setup.all_indices(i,iexo_w[:,i]))[1])])[nsinglem2] 
            wage_f2[singlef2,i]=np.exp(setup.pars['f_wage_trend'][i]+setup.exogrid.zf_t[i][iexo_w[singlef2,i]])  
            wage_m2[singlem2,i]=np.exp(setup.pars['m_wage_trend'][i]+setup.exogrid.zm_t[i][iexo_w[singlem2,i]])  
            
             
        #Log Income over time 
        for t in range(lenn): 
            ipart=(labor_w[:,t]>0.1) & (ifemale2) #& (state_w[:,t]>1) 
            ipartm=(state_w[:,t]==1) & (imale2) 
            log_inc_rel[0,t]=np.mean(np.log(wage_f2[ipart,t])) 
            log_inc_rel[1,t]=np.mean(np.log(wage_m2[imale2,t])) 
                 
             
        for ist,sname in enumerate(state_codes):  
                 
         
            for t in range(lenn):    
            
                #Arrays for preparation    
                is_state = (np.any(state[:,0:t]==ist,1))           
                is_state1 = (state[:,t]==ist)    
                is_state2 = (state_w[:,t]==ist)    
                if t<1:    
                    is_state=is_state1    
                
                ind1 = np.where(is_state1)[0]  
                ind1f = np.where((is_state1) & (agents.is_female[:,0][keep][keep2]))[0] 
                ind1m = np.where((is_state1) & ~(agents.is_female[:,0][keep][keep2]))[0] 
                         
                if not (np.any(is_state) or np.any(is_state1)): continue    
                     
                zf,zm,psi=mdl.setup.all_indices(t,iexo[ind1,t])[1:4]    
                         
 
                     
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
                     
                     if np.any(ind1f):inc_rel[ist,t,0]=np.mean(wage_f[:,t][keep][keep2][ind1f])+np.mean(wage_mp[:,t][keep][keep2][ind1f]) 
                     if np.any(ind1m):inc_rel[ist,t,1]=np.mean(wage_m[:,t][keep][keep2][ind1m])+np.mean(wage_fp[:,t][keep][keep2][ind1m]) 
                      
                else:   
                    
                   print('Error: No relationship chosen')   
                    
                    
 
             
         
        ################################### 
        #"Event Study" with simulated data 
        ################################# 
        
        #Build the event matrix 
        age_unid_e=np.argmax(changep_w,axis=1) 
        never_e=(changep_w[:,0]==0) & (age_unid_e[:]==0) 
        age_unid_e[never_e]=1000 
        age_unid_e[changep_w[:,-1]==0]=1000 
        age_unid_e1=np.repeat(np.expand_dims(age_unid_e,axis=1),len(changep_w[0,:]),axis=1) 
         
        agetemp_e=np.linspace(1,len(changep_w[0,:]),len(changep_w[0,:])) 
        agegridtemp_e=np.reshape(np.repeat(agetemp_e,len(changep_w[:,0])),(len(changep_w[:,0]),len(agetemp_e)),order='F') 
        #age_unid_e1[agegridtemp_e<=2]=1000 
        age_unid_e1[agegridtemp_e>=mdl.setup.pars['Tret']-int(mdl.setup.pars['Tret']/6)]=1000 
        event=agegridtemp_e-age_unid_e1-1 
         
        #Get beginning of the spell 
        changem=np.zeros(state_w.shape,dtype=bool) 
        changec=np.zeros(state_w.shape,dtype=bool) 
        change=np.zeros(state_w.shape,dtype=bool) 
        for t in range(1,mdl.setup.pars['Tret']-1):    
                
            irchangem = ((state_w[:,t]==2) & ((state_w[:,t-1]==0) | (state_w[:,t-1]==1)))  
            irchangec = ((state_w[:,t]==3) & ((state_w[:,t-1]==0) | (state_w[:,t-1]==1)))  
            irchange=((state_w[:,t]!=state_w[:,t-1]) & ((state_w[:,t-1]==0) | (state_w[:,t-1]==1)))  
            changem[:,t]=irchangem 
            changec[:,t]=irchangec 
            change[:,t]=irchange 
             
        #get variables for ols event study later 
        vage=agegridtemp_e[change] 
        vgender=female_w[change] 
        vtheta=theta_w[change] 
        vpsi=psis[change] 
        vmar=state_w[change] 
        vass=assets_w[change] 
        vwagef=wage_f2[change] 
        vwagem=wage_m2[change] 
             
        #Grid of event Studies 
        eventgrid=np.array(np.linspace(-10,10,21),dtype=np.int16) 
        event_thetam=np.ones(len(eventgrid))*-1000 
        event_thetac=np.ones(len(eventgrid))*-1000 
        event_psim=np.ones(len(eventgrid))*-1000 
        event_psic=np.ones(len(eventgrid))*-1000 
        match=np.ones(state_w.shape,dtype=bool)*-1000 
        
     
        i=0 
        for e in eventgrid: 
             
            matchm=(event==e) & (changem) 
            matchc=(event==e) & (changec) 
            imatch=(event==e) & (change) 
            match[imatch]=e 
            if np.any(matchm):event_thetam[i]=np.mean(theta_w[matchm]) 
            if np.any(matchc):event_thetac[i]=np.mean(theta_w[matchc]) 
            if np.any(matchm):event_psim[i]=np.mean(psis[matchm]) 
            if np.any(matchc):event_psic[i]=np.mean(psis[matchc]) 
             
            i+=1 
             
          
        data_ev=np.array(np.stack((vage,vgender,match[change],vtheta,vpsi,vmar,vass,vwagef,vwagem),axis=0).T,dtype=np.float64)    
        data_ev_panda=pd.DataFrame(data=data_ev,columns=['age','sex','event','theta','vpsi','vmar','vass','wagef','wagem'])   
        #Eliminate if missing    
        data_ev_panda.loc[data_ev_panda['event']>10,'event']=np.nan  
        data_ev_panda.loc[data_ev_panda['event']<-10,'event']=np.nan  
        data_ev_panda.loc[data_ev_panda['vmar']==3,'vmar']=0 
        data_ev_panda.loc[data_ev_panda['vmar']==2,'vmar']=1 
        data_ev_panda.dropna(subset=['event'])  
         
        #Regressions  
        try:    
            ols_mar = smf.ols(formula='vmar ~ C(age)+C(sex)+C(event, Treatment(reference=-1))', data = data_ev_panda ).fit() 
            ols_mar_theta = smf.ols(formula='theta ~ C(age)+C(sex)+C(event, Treatment(reference=-1))', data = data_ev_panda[data_ev_panda['vmar']==1] ).fit() 
            ols_coh_theta = smf.ols(formula='theta ~ C(age)+C(sex)+C(event, Treatment(reference=-1))', data = data_ev_panda[data_ev_panda['vmar']==0] ).fit() 
            ols_mar_psi = smf.ols(formula='vpsi ~ C(age)+C(sex)+C(event, Treatment(reference=-1))', data = data_ev_panda[data_ev_panda['vmar']==1] ).fit() 
            ols_coh_psi = smf.ols(formula='vpsi ~ C(age)+C(sex)+C(event, Treatment(reference=-1))', data = data_ev_panda[data_ev_panda['vmar']==0] ).fit() 
           
        except:    
            print('No data for unilateral divorce regression...')    
            beta_unid_s=0.0   
        
        #Create an Array for the results 
        pevent_theta_mar=np.ones(len(eventgrid))*np.nan 
        pevent_theta_coh=np.ones(len(eventgrid))*np.nan 
        pevent_psi_mar=np.ones(len(eventgrid))*np.nan 
        pevent_psi_coh=np.ones(len(eventgrid))*np.nan 
        pevent_mar=np.ones(len(eventgrid))*np.nan 
         
        i=0 
        for e in eventgrid: 
             
            try: 
                pevent_mar[i]=ols_mar.params['C(event, Treatment(reference=-1))[T.'+str(e)+'.0]'] 
                pevent_theta_mar[i]=ols_mar_theta.params['C(event, Treatment(reference=-1))[T.'+str(e)+'.0]'] 
                pevent_theta_coh[i]=ols_coh_theta.params['C(event, Treatment(reference=-1))[T.'+str(e)+'.0]'] 
                pevent_psi_mar[i]=ols_mar_psi.params['C(event, Treatment(reference=-1))[T.'+str(e)+'.0]'] 
                pevent_psi_coh[i]=ols_coh_psi.params['C(event, Treatment(reference=-1))[T.'+str(e)+'.0]'] 
                 
            except: 
                print('Skip event {}'.format(e))  
            i+=1   
                 
                 
        #Adjust for the reference point        
        pevent_mar[9]=0 
        pevent_theta_mar[9]=0 
        pevent_theta_coh[9]=0 
        pevent_psi_mar[9]=0 
        pevent_psi_coh[9]=0 
                 
        #Check correlations 
        assets_ww=assets_w[:,:60]
        ifemale1=(female_w==1) 
        imale1=(female_w==0) 
        nsinglefc1=(ifemale1[:,:60]) & (state_w[:,:60]==3) & (labor_w[:,:60]>0.1) 
        nsinglefm1=(ifemale1[:,:60]) & (state_w[:,:60]==2) & (labor_w[:,:60]>0.1) 
        nsinglefc2=(imale1[:,:60]) & (state_w[:,:60]==3) & (labor_w[:,:60]>0.1) 
        nsinglefm2=(imale1[:,:60]) & (state_w[:,:60]==2) & (labor_w[:,:60]>0.1) 
        wage_ft=wage_f[:,:60] 
        wage_mpt=wage_mp[:,:60] 
        wage_mt=wage_m[:,:60] 
        wage_fpt=wage_fp[:,:60] 
        
        #For constructing relative net worth measure
        agei=int(30/setup.pars['py'])
        agef=int(40/setup.pars['py'])
        if (agei==agef):agef=agef+1
        assets_w_mod=assets_w[:,agei:agef]
        nsinglefc1_mod=(ifemale1[:,agei:agef]) & (state_w[:,agei:agef]==3) & (labor_w[:,agei:agef]>0.1) 
        nsinglefm1_mod=(ifemale1[:,agei:agef]) & (state_w[:,agei:agef]==2) & (labor_w[:,agei:agef]>0.1) 
        nsinglefc2_mod=(imale1[:,agei:agef]) & (state_w[:,agei:agef]==3) & (labor_w[:,agei:agef]>0.1) 
        nsinglefm2_mod=(imale1[:,agei:agef]) & (state_w[:,agei:agef]==2) & (labor_w[:,agei:agef]>0.1) 
        nsinglefc1_mod1=(ifemale1[:,agei:agef]) & (state_w[:,agei:agef]==3)  
        nsinglefm1_mod1=(ifemale1[:,agei:agef]) & (state_w[:,agei:agef]==2)
        nsinglefc2_mod1=(imale1[:,agei:agef]) & (state_w[:,agei:agef]==3)
        nsinglefm2_mod1=(imale1[:,agei:agef]) & (state_w[:,agei:agef]==2)
        wage_ft_mod=wage_f[:,agei:agef] 
        wage_mpt_mod=wage_mp[:,agei:agef] 
        wage_mt_mod=wage_m[:,agei:agef] 
        wage_fpt_mod=wage_fp[:,agei:agef] 
        labor_w_mod=labor_w[:,agei:agef]
        
        #Net worh
        net_f_c=np.mean(assets_w_mod[nsinglefc1_mod]/(wage_ft_mod[nsinglefc1_mod]*setup.ls_levels[-1]+wage_mpt_mod[nsinglefc1_mod]))
        net_f_m=np.mean(assets_w_mod[nsinglefm1_mod]/(wage_ft_mod[nsinglefm1_mod]*setup.ls_levels[-1]+wage_mpt_mod[nsinglefm1_mod]))
        net_m_c=np.mean(assets_w_mod[nsinglefc2_mod]/(wage_fpt_mod[nsinglefc2_mod]*setup.ls_levels[-1]+wage_mt_mod[nsinglefc2_mod]))
        net_m_m=np.mean(assets_w_mod[nsinglefm2_mod]/(wage_fpt_mod[nsinglefm2_mod]*setup.ls_levels[-1]+wage_mt_mod[nsinglefm2_mod]))
        
        
        #Wages correlations
        corr=np.corrcoef(np.log(wage_ft[nsinglefc1]*setup.ls_levels[-1]),np.log(wage_mpt[nsinglefc1])) 
        corr1=np.corrcoef(np.log(wage_ft[nsinglefm1]*setup.ls_levels[-1]),np.log(wage_mpt[nsinglefm1])) 
        corrm=np.corrcoef(np.log(wage_mt[nsinglefc2]),np.log(wage_fpt[nsinglefc2]*setup.ls_levels[-1])) 
        corrm1=np.corrcoef(np.log(wage_mt[nsinglefm2]),np.log(wage_fpt[nsinglefm2]*setup.ls_levels[-1])) 
        share_fcm=np.mean(wage_ft[nsinglefc1]*setup.ls_levels[-1]/(wage_mpt[nsinglefc1]+wage_ft[nsinglefc1]*setup.ls_levels[-1])) 
        share_fmm=np.mean(wage_ft[nsinglefm1]*setup.ls_levels[-1]/(wage_mpt[nsinglefm1]+wage_ft[nsinglefm1]*setup.ls_levels[-1])) 
        share_mcm=np.mean(wage_fpt[nsinglefc2]*setup.ls_levels[-1]/(wage_mt[nsinglefc2]+wage_fpt[nsinglefc2]*setup.ls_levels[-1])) 
        share_mmm=np.mean(wage_fpt[nsinglefm2]*setup.ls_levels[-1]/(wage_mt[nsinglefm2]+wage_fpt[nsinglefm2]*setup.ls_levels[-1])) 
        
        #Correlation hh earnings and assets
        medassetsc1=np.median(assets_w_mod[nsinglefc1_mod1])
        medassetsm1=np.median(assets_w_mod[nsinglefm1_mod1])
        medassetsc2=np.median(assets_w_mod[nsinglefc2_mod1])
        medassetsm2=np.median(assets_w_mod[nsinglefm2_mod1])
        medincomec1=np.median(wage_ft_mod[nsinglefc1_mod1]*setup.ls_levels[labor_w_mod[nsinglefc1_mod1]]+wage_mpt_mod[nsinglefc1_mod1])
        medincomem1=np.median(wage_ft_mod[nsinglefm1_mod1]*setup.ls_levels[labor_w_mod[nsinglefm1_mod1]]+wage_mpt_mod[nsinglefm1_mod1])
        medincomec2=np.median(wage_fpt_mod[nsinglefc2_mod1]*setup.ls_levels[labor_w_mod[nsinglefc2_mod1]]+wage_mt_mod[nsinglefc2_mod1])
        medincomem2=np.median(wage_fpt_mod[nsinglefm2_mod1]*setup.ls_levels[labor_w_mod[nsinglefm2_mod1]]+wage_mt_mod[nsinglefm2_mod1])
        
        sh_f_c=np.mean((assets_w_mod[nsinglefc1_mod1]>medassetsc1)[(wage_ft_mod[nsinglefc1_mod1]*setup.ls_levels[labor_w_mod[nsinglefc1_mod1]]+wage_mpt_mod[nsinglefc1_mod1]>medincomec1)])
        sh_f_m=np.mean((assets_w_mod[nsinglefm1_mod1]>medassetsm1)[(wage_ft_mod[nsinglefm1_mod1]*setup.ls_levels[labor_w_mod[nsinglefm1_mod1]]+wage_mpt_mod[nsinglefm1_mod1]>medincomem1)])
        sh_m_c=np.mean((assets_w_mod[nsinglefc2_mod1]>medassetsc2)[(wage_fpt_mod[nsinglefc2_mod1]*setup.ls_levels[labor_w_mod[nsinglefc2_mod1]]+wage_mt_mod[nsinglefc2_mod1]>medincomec2)])
        sh_m_m=np.mean((assets_w_mod[nsinglefm2_mod1]>medassetsm2)[(wage_fpt_mod[nsinglefm2_mod1]*setup.ls_levels[labor_w_mod[nsinglefm2_mod1]]+wage_mt_mod[nsinglefm2_mod1]>medincomem2)])
        
        #Results
        print('FM Correlation in potential wages for cohabitaiton is {}, for marriage only is {}'.format(corr[0,1],corr1[0,1]) )   
        print('MM Correlation in potential wages for cohabitaiton is {}, for marriage only is {}'.format(corrm[0,1],corrm1[0,1]) )   
        print('FM Share wages earned by female in cohabitaiton is {}, for marriage only is {}'.format(share_fcm,share_fmm) )   
        print('MM Share wages earned by female in cohabitaiton is {}, for marriage only is {}'.format(share_mcm,share_mmm) )  
        print('FM share wealthy if incom>median for coh is {}, for marriage only is {}'.format(sh_f_c,sh_f_m) )   
        print('MM share wealthy if incom>median for coh is {}, for marriage only is {}'.format(sh_m_c,sh_m_m) )  
        print('FM median networth age 50-60 for cohabitaiton is {}, for marriage only is {}'.format(net_f_c,net_f_m) )   
        print('MM median networth age 50-60 for cohabitaiton is {}, for marriage only is {}'.format(net_m_c,net_m_m) )  
             
        
        #Get useful package for denisty plots 
        import seaborn as sns 
         
        #Print something useful for debug and rest    
        print('The share of singles choosing marriage is {0:.2f}'.format(sharem))    
        cond=(state<2)    
        if assets_t[cond].size:    
            print('The max level of assets for singles is {:.2f}, the grid upper bound is {:.2f}'.format(np.amax(assets_t[cond]),max(mdl.setup.agrid_s)))    
        cond=(state>1)    
        if assets_t[cond].size:    
            print('The max level of assets for couples is {:.2f}, the grid upper bound is {:.2f}'.format(np.amax(assets_t[cond]),max(mdl.setup.agrid_c)))    
             
        #Setup a file for the graphs    
        pdf = matplotlib.backends.backend_pdf.PdfPages("moments_graphs.pdf")    
             
        #################    
        #Get data moments    
        #################    
             
        #Get Data Moments    
        with open('moments.pkl', 'rb') as file:    
            packed_data=pickle.load(file)    
             
            #Unpack Moments (see data_moments.py to check if changes)    
            #(hazm,hazs,hazd,mar,coh,fls_ratio,W)    
            hazm_d=packed_data['hazm']    
            hazs_d=packed_data['hazs']    
            hazd_d=packed_data['hazd']    
            mar_d=packed_data['emar']    
            coh_d=packed_data['ecoh']    
            fls_d=packed_data['fls_ratio']   
            wage_d=np.ones(1)*packed_data['wage_ratio'] 
            div_d=np.ones(1)*packed_data['div_ratio'] 
            mean_fls_d=np.ones(1)*packed_data['mean_fls']   
            beta_unid_d=np.ones(1)*packed_data['beta_unid']    
            hazm_i=packed_data['hazmi']    
            hazs_i=packed_data['hazsi']    
            hazd_i=packed_data['hazdi']    
            mar_i=packed_data['emari']    
            coh_i=packed_data['ecohi']    
            fls_i=packed_data['fls_ratioi']   
            wage_i=np.ones(1)*packed_data['wage_ratioi'] 
            mean_fls_i=np.ones(1)*packed_data['mean_flsi']  
            beta_unid_i=np.ones(1)*packed_data['beta_unidi']    
     
             
             
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
        plt.plot(np.array(range(lg))+1, hazd[0:lg],one, linestyle='--',linewidth=1.5, label='Hazard of Divorce - S')    
        plt.plot(np.array(range(lg))+1, hazd_d[0:lg],two,linewidth=1.5, label='Hazard of Divorce - D')    
        plt.fill_between(np.array(range(lg))+1, hazd_i[0,0:lg], hazd_i[1,0:lg],alpha=0.2,facecolor='b')    
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small')    
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Years')    
        plt.ylabel('Hazard')    
        plt.savefig('hazd.pgf', bbox_inches = 'tight',pad_inches = 0)  
             
        #############################################    
        # Hazard of Separation    
        #############################################    
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1)    
        lg=min(len(hazs_d),len(hazs))  
        plt.plot(np.array(range(lg))+1, hazs[0:lg],one, linestyle='--',linewidth=1.5, label='Hazard of Separation - S')    
        plt.plot(np.array(range(lg))+1, hazs_d[0:lg],two,linewidth=1.5, label='Hazard of Separation - D')    
        plt.fill_between(np.array(range(lg))+1, hazs_i[0,0:lg], hazs_i[1,0:lg],alpha=0.2,facecolor='b')    
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small')    
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Years')    
        plt.ylabel('Hazard')    
        plt.savefig('hazs.pgf', bbox_inches = 'tight',pad_inches = 0)  
            
             
        #############################################    
        # Hazard of Marriage    
        #############################################    
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1)    
        lg=min(len(hazm_d),len(hazm))  
    
        plt.plot(np.array(range(lg))+1, hazm[0:lg],one, linestyle='--',linewidth=1.5, label='Hazard of Marriage - S')    
        plt.plot(np.array(range(lg))+1, hazm_d[0:lg],two,linewidth=1.5, label='Hazard of Marriage - D')    
        plt.fill_between(np.array(range(lg))+1, hazm_i[0,0:lg], hazm_i[1,0:lg],alpha=0.2,facecolor='b')    
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small')    
        plt.ylim(ymin=0)    
        #plt.legend(loc='upper left', shadow=True, fontsize='x-small')    
        plt.xlabel('Duration - Years')    
        plt.ylabel('Hazard')    
        plt.savefig('hazm.pgf', bbox_inches = 'tight',pad_inches = 0)  
            
        ##########################################    
        # Assets Over the Live Cycle    
        ##########################################    
        fig = plt.figure()    
        f2=fig.add_subplot(2,1,1)    
             
        for ist,sname in enumerate(state_codes):    
            plt.plot(np.array(range(lenn)), ass_rel[ist,:,0],color=print(ist/len(state_codes)),markersize=6, label=sname)    
        plt.plot(np.array(range(lenn)), ass_rel[2,:,1], linestyle='--',color=print(2/len(state_codes)),markersize=6, label='Marriage male') 
        plt.plot(np.array(range(lenn)), ass_rel[3,:,1], linestyle='--',color=print(3/len(state_codes)),markersize=6, label='Cohabitation other') 
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
               
            plt.plot(np.array(range(lenn)), inc_rel[ist,:,0],color=print(ist/len(state_codes)),markersize=6, label=sname)  
             
        plt.plot(np.array(range(lenn)), inc_rel[2,:,1], linestyle='--',color=print(2/len(state_codes)),markersize=6, label='Marriage male') 
        plt.plot(np.array(range(lenn)), inc_rel[3,:,1], linestyle='--',color=print(3/len(state_codes)),markersize=6, label='Cohabitation Male') 
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
        plt.plot(agea, inc_men['earn_age'], marker='o',color=print(3/len(state_codes)),markersize=6, label='Men Data') 
        plt.plot(agea, inc_women['earn_age'], marker='o',color=print(2/len(state_codes)),markersize=6, label='Women Data') 
        plt.plot(agea, log_inc_rel[0,mdl.setup.pars['Tbef']:lend+mdl.setup.pars['Tbef']],color=print(2/len(state_codes)),markersize=6, label='Women Simulation') 
        plt.plot(agea, log_inc_rel[1,mdl.setup.pars['Tbef']:lend+mdl.setup.pars['Tbef']],color=print(3/len(state_codes)),markersize=6, label='Men Simulation') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Income')  
                     
         
        ##########################################    
        # More on Income 
        ##########################################  
         
 
         
        fig = plt.figure()    
        f3=fig.add_subplot(2,1,1)    
         
        lend=len(wage_fs) 
        agea=np.array(range(lend))+20 
        
        plt.plot(agea, wage_fs, marker='o',color='g',markersize=3, label='Women single') 
        plt.plot(agea, wage_ms, marker='o',color='y',markersize=3, label='Men single') 
        plt.plot(agea, wage_fc,color='r',markersize=3, label='Women coh') 
        plt.plot(agea, wage_mc,color='b',markersize=3, label='Men coh') 
        plt.plot(agea, wage_fm,color='m',markersize=3, label='Women mar') 
        plt.plot(agea, wage_mm,color='k',markersize=3, label='Men mar') 
        plt.plot(agea, wage_fpc,linestyle='--',color='r',markersize=3, label='Women coh-o') 
        plt.plot(agea, wage_mpc,linestyle='--',color='b',markersize=3, label='Men coh-o') 
        plt.plot(agea, wage_fpm,linestyle='--',color='m',markersize=3, label='Women mar-o') 
        plt.plot(agea, wage_mpm,linestyle='--',color='k',markersize=3, label='Men mar-o') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Log Income')  
         
        ##########################################    
        # Variance Income 
        ##########################################         
        fig = plt.figure()    
        f3=fig.add_subplot(2,1,1)    
         
        lend=len(wage_fs) 
        agea=np.array(range(lend))+20 
        
        plt.plot(agea, var_wage_fc,color='r',markersize=3, label='Women coh') 
        plt.plot(agea, var_wage_mc,color='b',markersize=3, label='Men coh') 
        plt.plot(agea, var_wage_fm,color='m',markersize=3, label='Women mar') 
        plt.plot(agea, var_wage_mm,color='k',markersize=3, label='Men mar') 
        plt.plot(agea, var_wage_fpc,linestyle='--',color='r',markersize=3, label='Women coh-o') 
        plt.plot(agea, var_wage_mpc,linestyle='--',color='b',markersize=3, label='Men coh-o') 
        plt.plot(agea, var_wage_fpm,linestyle='--',color='m',markersize=3, label='Women mar-o') 
        plt.plot(agea, var_wage_mpm,linestyle='--',color='k',markersize=3, label='Men mar-o') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Log Income Variance')  
         
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
        for ist,sname in enumerate(state_codes):    
            plt.plot([],[],color=print(ist/len(state_codes)), label=sname)    
        plt.stackplot(xa,relt1[0,]/N,relt1[1,]/N,relt1[2,]/N,relt1[3,]/N,    
                      colors = ['b','y','g','r'])               
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('Share')    
             
        ##########################################    
        # Relationship and Data    
        ##########################################          
        fig = plt.figure()    
        f4=fig.add_subplot(2,1,1)    
        lg=min(len(mar_d),len(relt[1,:]))    
        xa=(5*np.array(range(lg))+20)   
        plt.plot(xa, mar_d[0:lg],'g',linewidth=1.5, label='Married - D')    
        plt.fill_between(xa, mar_i[0,0:lg], mar_i[1,0:lg],alpha=0.2,facecolor='g')    
        plt.plot(xa, reltt[2,0:lg]/N,'g',linestyle='--',linewidth=1.5, label='Married - S')    
        plt.plot(xa, coh_d[0:lg],'r',linewidth=1.5, label='Cohabiting - D')    
        plt.fill_between(xa, coh_i[0,0:lg], coh_i[1,0:lg],alpha=0.2,facecolor='r')    
        plt.plot(xa, reltt[3,0:lg]/N,'r',linestyle='--',linewidth=1.5, label='Cohabiting - S')    
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.ylim(ymax=1.0)    
        plt.xlabel('Age')    
        plt.ylabel('Share')    
        plt.margins(0,0)  
        plt.savefig('erel.pgf', bbox_inches = 'tight',pad_inches = 0)  
             
        ##########################################    
        # FLS Over the Live Cycle    
        ##########################################          
        fig = plt.figure()    
        f5=fig.add_subplot(2,1,1)    
        xa=(mdl.setup.pars['py']*np.array(range(mdl.setup.pars['Tret']))+20)   
        plt.plot(xa, flsm,color='r', label='Marriage')    
        plt.plot(xa, flsc,color='k', label='Cohabitation')             
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Age')    
        plt.ylabel('FLS')    
         
        ##########################################    
        # Distribution of Love  
        ##########################################   
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1) 
         
         
        sns.kdeplot(psis[state_w==3], shade=True,shade_lowest=False,linewidth=0.01, color="r", bw=.05,label = 'Cohabitaition') 
        sns.kdeplot(psis[state_w==2], shade=True,shade_lowest=False,linewidth=0.01, color="b", bw=.05,label = 'Marriage') 
        sns.kdeplot(psis[changec], color="r", bw=.05,label = 'Cohabitaition Beg') 
        sns.kdeplot(psis[changem], color="b", bw=.05,label = 'Marriage Beg')   
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Love Shock')    
        plt.ylabel('Denisty')  
         
        ##########################################    
        # Distribution of Love - Cumulative 
        ##########################################   
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1) 
         
        # evaluate the histogram 
        valuesc, basec = np.histogram(psis[changec], bins=1000) 
        valuesm, basem = np.histogram(psis[changem], bins=1000) 
        valuesct, basect = np.histogram(psis[state_w==3], bins=1000) 
        valuesmt, basemt = np.histogram(psis[state_w==2], bins=1000) 
        #evaluate the cumulative 
        cumulativec = np.cumsum(valuesc) 
        cumulativem = np.cumsum(valuesm) 
        cumulativect = np.cumsum(valuesct) 
        cumulativemt = np.cumsum(valuesmt) 
        # plot the cumulative function 
        plt.plot(basec[:-1], cumulativec/max(cumulativec), c='red',label = 'Cohabitaition') 
        plt.plot(basem[:-1], cumulativem/max(cumulativem), c='blue',label = 'Marriage') 
        #plt.plot(basect[:-1], cumulativect/max(cumulativect),linestyle='--', c='red',label = 'Cohabitaition-All') 
        #plt.plot(basemt[:-1], cumulativemt/max(cumulativemt),linestyle='--', c='blue',label = 'Marriage-All') 
        plt.legend(loc='best', ncol=1, fontsize='x-small')    
        plt.xlabel('Love Shock $\psi$')    
        plt.ylabel('Probability')  
        plt.savefig('psidist.pgf', bbox_inches = 'tight',pad_inches = 0)  
         
        ############################################################ 
        # Distribution of Love - Cumulative - Before and After policy 
        ################################################################### 
         
        fig = plt.figure()    
        f1=fig.add_subplot(2,1,1) 
         
        # evaluate the histogram 
        valuesc, basec = np.histogram(psis[(changec) & (event<0)], bins=1000) 
        valuesm, basem = np.histogram(psis[(changec) & (event>=0)], bins=1000) 
 
        #evaluate the cumulative 
        cumulativec = np.cumsum(valuesc) 
        cumulativem = np.cumsum(valuesm) 
        cumulativect = np.cumsum(valuesct) 
        cumulativemt = np.cumsum(valuesmt) 
        # plot the cumulative function 
        plt.plot(basec[:-1], cumulativec/max(cumulativec), c='red',label = 'Bilateral') 
        plt.plot(basem[:-1], cumulativem/max(cumulativem), c='blue',label = 'Unilateral') 
        plt.legend(loc='best', ncol=1, fontsize='x-small')    
        plt.xlabel('Love Shock $\psi$')    
        plt.ylabel('Probability')  
        plt.savefig('psipol.pgf', bbox_inches = 'tight',pad_inches = 0)  
         
        ##########################################    
        # Distribution of Pareto Weight  
        ##########################################   
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1) 
         
         
        sns.kdeplot(theta_w[state_w==3], shade=True,shade_lowest=False,linewidth=0.01, color="r", bw=.05,label = 'Cohabitaition') 
        sns.kdeplot(theta_w[state_w==2], shade=True,shade_lowest=False,linewidth=0.01, color="b", bw=.05,label = 'Marriage') 
        sns.kdeplot(theta_w[changec], color="r", bw=.05,label = 'Cohabitaition Beg') 
        sns.kdeplot(theta_w[changem], color="b", bw=.05,label = 'Marriage Beg')  
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Female Pareto Weight')    
        plt.ylabel('Denisty')  
         
         
         
        ##########################################    
        # Event Study Love Shock 
        ##########################################   
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1) 
         
        plt.plot(eventgrid, event_psic,color='r', label='Cohabitation') 
        plt.plot(eventgrid, event_psim,color='b', label='Marriage') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Time Event Unilateral Divorce')    
        plt.ylabel('Love Shock')  
        plt.savefig('psiuni.pgf', bbox_inches = 'tight',pad_inches = 0)  
         
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1) 
         
        plt.plot(eventgrid, pevent_psi_coh,color='r', label='Cohabitation') 
        plt.plot(eventgrid, pevent_psi_mar,color='b', label='Marriage') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Time Event Unilateral Divorce')    
        plt.ylabel('Love Shock-Coefficient') 
         
        ##########################################    
        # Event Study Pareto Weight 
        ##########################################   
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1) 
         
        plt.plot(eventgrid, event_thetac,color='r', label='Cohabitation') 
        plt.plot(eventgrid, event_thetam,color='b', label='Marriage') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Time Event Unilateral Divorce')    
        plt.ylabel('Female Pareto weight')  
        plt.savefig('weight.pgf', bbox_inches = 'tight',pad_inches = 0)  
         
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1) 
         
        plt.plot(eventgrid, pevent_theta_coh,color='r', label='Cohabitation') 
        plt.plot(eventgrid, pevent_theta_mar,color='b', label='Marriage') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Time Event Unilateral Divorce')    
        plt.ylabel('Female Pareto weight-Coefficient')  
   
        ##########################################    
        # DID of unilateral fivorce on part choice    
        ##########################################    
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1)    
             
            
        # create plot    
        x=np.array([0.25,0.75])   
        y=np.array([beta_unid_d,beta_unid_s])    
        yerr=np.array([(beta_unid_i[1]-beta_unid_i[0])/2.0,0.0])    
        plt.axhline(linewidth=0.1, color='r')    
        plt.errorbar(x, y, yerr=yerr, fmt='o', elinewidth=0.03)    
        plt.ylabel('OLS Coefficient - UniD')    
        plt.xticks(x, ["Data","Simulation"] )   
        plt.ylim(ymax=0.1)    
        plt.xlim(xmax=1.0,xmin=0.0)    
         
        ########################################## 
        #Event Study Marriage 
        ########################################## 
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1) 
         
 
        plt.plot(eventgrid, pevent_mar,color='b', label='Event Study Unid') 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=len(state_codes), fontsize='x-small')    
        plt.xlabel('Time Event')    
        plt.ylabel('Marriage-Coefficient')  
          
        ##########################################    
        # FLS: Marriage vs. cohabitation  
        ##########################################     
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1)    
             
 
         
        lg=2 
        # create plot    
        plt.plot(np.array(range(lg))+1, moments['fls_ratio'], linestyle='--',linewidth=1.5, label='Simulated')    
        plt.plot(np.array(range(lg))+1, fls_d,linewidth=1.5, label='Data')    
        plt.fill_between(np.array(range(lg))+1, fls_i[0,0:lg], fls_i[1,0:lg],alpha=0.2,facecolor='b')    
        plt.ylabel('Ratio of Female Hrs: Mar/Coh') 
        plt.legend(loc='best', bbox_to_anchor=(0.5, -0.3),    
                  fancybox=True, shadow=True, ncol=2, fontsize='x-small')   
        
        #plt.ylim(ymax=0.1)    
        #plt.xlim(xmax=1.0,xmin=0.0)    
         
        ##########################################    
        # Wage: Marriage vs. cohabitation  
        ##########################################     
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1)    
             
            
        # create plot    
        x=np.array([0.25,0.75])   
        y=np.array([wage_d,moments['wage_ratio']])    
        yerr=np.array([(wage_i[1]-wage_i[0])/2.0,0.0])    
        plt.axhline(y=1.0,linewidth=0.1, color='r')    
        plt.errorbar(x, y, yerr=yerr, fmt='o', elinewidth=0.03)    
        plt.ylabel('Difference of Male Log wages: Mar-Coh')    
        plt.xticks(x, ["Data","Simulation"] )   
        #plt.ylim(ymax=0.1)    
        #plt.xlim(xmax=1.0,xmin=0.0)    
         
         
        ##########################################    
        # Wage: Marriage vs. cohabitation  
        ##########################################     
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1)    
             
            
        # create plot    
        x=np.array([0.25,0.75])   
        y=np.array([div_d,moments['div_ratio']])    
        yerr=np.array([(wage_i[1]-wage_i[0])/2.0,0.0])    
        plt.axhline(y=1.0,linewidth=0.1, color='r')    
        plt.errorbar(x, y, yerr=yerr, fmt='o', elinewidth=0.03)    
        plt.ylabel('Ratio of Male Divorce rates: Rich over poor')    
        plt.xticks(x, ["Data","Simulation"] )   
        #plt.ylim(ymax=0.1)    
        plt.xlim(xmax=1.0,xmin=0.0)  
          
          
        ##########################################    
        # FLS  
        ##########################################      
        fig = plt.figure()    
        f6=fig.add_subplot(2,1,1)    
             
            
        # create plot    
        x=np.array([0.25,0.75])   
        y=np.array([mean_fls_d,mean_fls])    
        yerr=np.array([(mean_fls_i[1]-mean_fls_i[0])/2.0,0.0])    
        plt.errorbar(x, y, yerr=yerr, fmt='o', elinewidth=0.03)    
        plt.ylabel('Female Labor Hours')    
        plt.xticks(x, ["Data","Simulation"] )   
        #plt.ylim(ymax=0.1)    
        plt.xlim(xmax=1.0,xmin=0.0)    
          
            
           
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
        plt.ylabel('Relative Hazard - UniD vs. Bil')    
        plt.xticks(x, ["Overall Risk","Risk of Marriage","Risk of Separation"] )   
        #plt.ylim(ymax=1.2,ymin=0.7)    
        plt.xlim(xmax=1.0,xmin=0.0)    
        #plt.xticks(index , ('Unilateral', 'Bilateral'))    
          
     
        ##########################################    
        # Put graphs together    
        ##########################################    
        #show()    
        for fig in range(1, plt.gcf().number + 1): ## will open an empty extra figure :(    
            pdf.savefig( fig )    
            
        pdf.close()    
        matplotlib.pyplot.close("all")    
         
           
    return moments   
             
            
