# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:11:28 2020

@author: Fabio
"""


import pandas as pd   
import numpy as np   
import pickle    
import statsmodels.formula.api as smf   
from lifelines import CoxPHFitter,WeibullAFTFitter
   
################################   
#Functions   
###############################   
def hazards(dataset,event,duration,end,listh,number,wgt):   
     #Create hazard given some spells in   
     #dataframe   
       
    #Number of unit weights-needed later   
    lgh=np.sum(dataset[wgt])#len(dataset)   
       
    for t in range(number):   
       
        #Get who has the event realized in t+1   
        cond=np.array(dataset[duration])==t+1   
        temp=dataset[[end,wgt]][cond]   
        cond1=temp[end]==event   
        temp1=temp[cond1]   
           
        #Compute the hazard   
        if lgh>0:   
            haz1=np.sum(temp1[wgt])/lgh#len(temp1)/lgh   
            lgh=lgh-np.sum(temp[wgt])#lgh-len(temp)   
        else:   
            haz1=0.0   
           
           
        #If hazard is zero do substitute this   
        #with a random very small number. This   
        #will help later for computing the variance   
        #covariance matrix   
        if (haz1<0.0001):   
            haz1=np.random.uniform(0,0.0001)   
               
        #Add hazard to the list   
        listh=[haz1]+listh   
       
    listh.reverse()   
    listh=np.array(listh).T   
    return listh   
   
   
   
   
#####################################   
#Routine that computes moments   
#####################################   
def compute(dpi,dco,dma,period=3,transform=1):   
    #compute moments, period   
    #says how many years correspond to one   
    #period   
   
    ###############################
    #Assortative Mating
    #################################
     
    dco.loc[(dco['edusp']>=0) & (dco['edusp']<=30),'collegep']=0.0  
    dco.loc[(dco['edusp']>=16) & (dco['edusp']<=30),'collegep']=1.0  
    
    dma.loc[(dma['edusp']>=0) & (dma['edusp']<=30),'collegep']=0.0  
    dma.loc[(dma['edusp']>=16) & (dma['edusp']<=30),'collegep']=1.0  
    
    dco1=dco.drop(columns=['timec','countc']).dropna()
    dma1=dma.dropna()
    
    
    male=np.concatenate((dco1[dco1.sex==1].college,dma1[dma1.sex==1].college))
    malep=np.concatenate((dco1[dco1.sex==1].collegep,dma1[dma1.sex==1].collegep))
    wmale=np.concatenate((dco1[dco1.sex==1].weight,dma1[dma1.sex==1].weight))
    
    female=np.concatenate((dco1[dco1.sex==2].college,dma1[dma1.sex==2].college))
    femalep=np.concatenate((dco1[dco1.sex==2].collegep,dma1[dma1.sex==2].collegep))
    wfemale=np.concatenate((dco1[dco1.sex==2].weight,dma1[dma1.sex==2].weight))
    
    
 
    malecorr=np.corrcoef(male,malep)[0,1]
    femalecorr=np.corrcoef(female,femalep)[0,1]
    ############################
    # RELATIONSHIP BY AGE
    ##########################

    #Filter Data to make it faster
    filter_c = [col for col in dpi if col.startswith('MAr_C')]
    filter_s = [col for col in dpi if col.startswith('MAr_st')]
       
    coh=dpi[filter_c]
    sta=dpi[filter_s]
    listc= [x for x in list(coh.columns.values) if '1994' not in x and
                                                      '1995' not in x and
                                                      '1996' not in x and
                                                      '2018' not in x]
    
    lists= [x for x in list(sta.columns.values) if '1994' not in x and
                                                      '1995' not in x and
                                                      '1996' not in x and
                                                      '2018' not in x]

       
    coh=np.array(coh[listc])
    sta=np.array(sta[lists])
 
    coht=np.zeros(coh.shape)
    mart=np.zeros(coh.shape)    
    ecoh=np.ones(coh.shape)*-100
    emar=np.ones(coh.shape)*-100

    
    
    #get if ever cohabited each month
    mart[coh>=200]=1.0
    coht[(coh>=100) & (coh<200)]=1.0
    
    # #Get first relationship
    # ever_in_rel=(np.sum(np.array(mart+coht),axis=1)>0)
    # marmin=(np.cumsum(mart+coht,axis=1)==1)
    # marrf=mart[where_first]==1 & coht[where_first]==0
   
    #Create cariavle evercohabited and married
    ecoh[(sta!=-4) & (sta!=-3)]=0.0
    emar[(sta!=-4) & (sta!=-3)]=0.0
    
    tmax=12*(2018-1997)
    
    for i in range(tmax):
        
        ecoh[:,i][(ecoh[:,max(i-1,0)]>0.1) | (coht[:,i]>0.1)]=1.0
        emar[:,i][(emar[:,max(i-1,0)]>0.1) | (mart[:,i]>0.1)]=1.0
        
        
    ecoh[(sta==-4) | (sta==-3)]=np.nan
    emar[(sta==-4) | (sta==-3)]=np.nan
    #Get age forearch column
    agei=np.array(12*(1997-dpi['birthy'])-dpi['birthm'],dtype=np.int16)
    dime=np.array(len(np.array(ecoh)[0,:]),dtype=np.int16)
    ageb=np.reshape(np.repeat(agei,dime),(len(agei),dime))+np.linspace(1,dime,dime,dtype=np.int16)
    
    #Get weights
    weightsa=np.array(dpi['weight'])
    wgt=np.reshape(np.repeat(weightsa,dime),(len(weightsa),dime))
    
    #get education
    colla=np.array(dpi['college'])
    coll=np.reshape(np.repeat(colla,dime),(len(colla),dime))
   
    
    #Cohabitation
    wgt_c=wgt[np.isnan(ecoh)==False]
    agebc_c=ageb[np.isnan(ecoh)==False]
    ecoh_c=ecoh[np.isnan(ecoh)==False]
    
    everc=np.zeros(4)
    for i in range(4): everc[i]=np.average(ecoh_c[(agebc_c==240+i*60)],weights=wgt_c[(agebc_c==240+i*60)])
    
    
    #Marriage
    wgt_m=wgt[np.isnan(emar)==False]
    agebc_m=ageb[np.isnan(emar)==False]
    emar_m=emar[np.isnan(emar)==False]
    
    everm=np.zeros(4)
    for i in range(4): everm[i]=np.average(emar_m[(agebc_m==240+i*60)],weights=wgt_m[(agebc_m==240+i*60)])
    
    #Relationship educated
    wgt_e=wgt[(np.isnan(emar)==False) & (np.isnan(ecoh)==False)]
    agebc_e=ageb[(np.isnan(emar)==False) & (np.isnan(ecoh)==False)]
    emar_e=emar[(np.isnan(emar)==False) & (np.isnan(ecoh)==False)]
    ecoh_e=ecoh[(np.isnan(emar)==False) & (np.isnan(ecoh)==False)]
    coll=coll[(np.isnan(emar)==False) & (np.isnan(ecoh)==False)]
    
    everr_e=np.zeros(4)
    everr_ne=np.zeros(4)
    for i in range(4): 
        
        summ=emar_e[(agebc_e==240+i*60) & (coll==1)]+ecoh_e[(agebc_e==240+i*60) & (coll==1)]
        summ[summ>=1]=1
        everr_e[i]=np.average(summ,weights=wgt_e[(agebc_e==240+i*60) & (coll==1)])
        
        summ=emar_e[(agebc_e==240+i*60) & (coll==0)]+ecoh_e[(agebc_e==240+i*60) & (coll==0)]
        summ[summ>=1]=1
        everr_ne[i]=np.average(summ,weights=wgt_e[(agebc_e==240+i*60) & (coll==0)])
        
        
    everm_e=np.zeros(4)
    everm_ne=np.zeros(4)
    for i in range(4): 
        
        summ=emar_e[(agebc_e==240+i*60) & (coll==1)]
        summ[summ>=1]=1
        everm_e[i]=np.average(summ,weights=wgt_e[(agebc_e==240+i*60) & (coll==1)])
        
        summ=emar_e[(agebc_e==240+i*60) & (coll==0)]
        summ[summ>=1]=1
        everm_ne[i]=np.average(summ,weights=wgt_e[(agebc_e==240+i*60) & (coll==0)])
        
    everc_e=np.zeros(4)
    everc_ne=np.zeros(4)
    for i in range(4): 
        
        summ=ecoh_e[(agebc_e==240+i*60) & (coll==1)]
        summ[summ>=1]=1
        everc_e[i]=np.average(summ,weights=wgt_e[(agebc_e==240+i*60) & (coll==1)])
        
        summ=ecoh_e[(agebc_e==240+i*60) & (coll==0)]
        summ[summ>=1]=1
        everc_ne[i]=np.average(summ,weights=wgt_e[(agebc_e==240+i*60) & (coll==0)])
    
    
   
    
    ecoh1,emar1=ecoh.copy(),emar.copy()
    ecoh1[np.isnan(ecoh)]=0
    emar1[np.isnan(emar)]=0
    ec=np.cumsum(ecoh1,axis=1)
    em=np.cumsum(emar1,axis=1)
    
    firstmar=np.any(((em>ec) & (em>0))==True,axis=1)
    firstcoh=np.any(((ec>em) & (ec>0))==True,axis=1)
    
    weight=weightsa.copy()
    weight[(firstmar+firstcoh)==False]=0.0
    
    ratio_mar=np.average(firstmar[colla==0],weights=weight[colla==0])/np.average(firstmar[colla==1],weights=weight[colla==1])
 

    
    ##########################   
    #BUILD HAZARD RATES   
    #########################    n
       
    #Get Duration bins   
    bins_d=np.linspace(0,1200,int((100/period)+1))   
    bins_d_label=np.linspace(1,len(bins_d)-1,len(bins_d)-1)   
    
    #Transform Duration in Years   
    dco['dur']=dco['timef']-dco['time']+1
    dco['dury'] = pd.cut(x=dco['dur'], bins=bins_d,labels=bins_d_label)          
    dco['dury']=dco['dury'].astype(float) 
    
    dma['dur']=dma['timef']-dma['time']+1
    dma['dury'] = pd.cut(x=dma['dur'], bins=bins_d,labels=bins_d_label)          
    dma['dury']=dma['dury'].astype(float) 
    
    dma['cohl'] = pd.cut(x=dma['countc'], bins=bins_d,labels=bins_d_label)          
    dma['cohl']=dma['cohl'].astype(float) 
    dma.loc[dma['countc']==0,'cohl']=0
    dma.loc[dma['countc']<0,'cohl']=np.nan
    

    #Hazard of Separation   
    hazs=list()   
   
    hazs=hazards(dco,1,'dury','fail',hazs,int(6/period),'weight')   
       
    #Hazard of Marriage   
    hazm=list()   
    hazm=hazards(dco,1,'dury','marry',hazm,int(6/period),'weight')   
       
    #Hazard of Divorce   
    hazd=list()   
    
    hazd=hazards(dma,1,'dury','fail',hazd,int(12/period),'weight')  
    
    #Hazard of Divorce-Educated
    hazde=list()   
    
    data=dma[dma['college']==1]
    hazde=hazards(data,1,'dury','fail',hazde,int(12/period),'weight')  
    
  
    
    ###############################################
    #Cox Regression: prearital cohabitation divorce
    ###############################################
    cph = CoxPHFitter() 
    #dma['age']=dma['yeara']-dma['birthy']
    dma['age']=round((12*(1997-dma['birthy'])-dma['birthm']+dma['time'])/12)
    data_m_panda=pd.DataFrame(data=dma.dropna(),columns=['fail','dury','cohl','college','age'])
    data_m_panda['ecoh']=0.0
    data_m_panda.loc[data_m_panda['cohl']>0.1,'ecoh']=1.0   
    data_m_panda['lcoh']=np.log(data_m_panda['cohl']+0.001)    
    #data_m_panda['age2']=data_m_panda['age']**2
    #data_m_panda['age3']=data_m_panda['age']**3
    #data_m_panda['age2']=data_m_panda['age']**2   
    
            
    data_m_panda=data_m_panda.drop(columns=['cohl','age']) 
    
    cox_div=cph.fit(data_m_panda, duration_col='dury', event_col='fail')  
    param=cox_div.params_
    beta_edu=cox_div.hazard_ratios_['college']
    #Get premartial cohabitation
 
    
    ref_dut=[np.exp(np.log(i+0.001)*param['lcoh']+param['ecoh'])/np.exp(np.log(0.001)*param['lcoh']) for i in range(15)]
        
   
    ref_dut[0]=1.0
    
    
    #################################
    #GET FLS
    ###############################
    filter_col_stat = [col for col in dpi if col.startswith('MAr_C')]
    filter_col_lab = [col for col in dpi if col.startswith('h_')]
    spike_cols = [col for col in dpi.columns if not '1994' or '2018' in col]
     

    
    status=dpi[filter_col_stat]
    listv= [x for x in list(status.columns.values) if '1994' not in x and
                                                      '1995' not in x and
                                                      '1996' not in x and
                                                      '2018' not in x]

       
    status=status[listv][dpi['sex']==2]
    labor=dpi[filter_col_lab][dpi['sex']==2]
    laborm=dpi[filter_col_lab][dpi['sex']==1]
    
    

    
    
    #Get age
    agei=np.array(12*(1997-dpi['birthy'])-dpi['birthm'],dtype=np.int16)[dpi['sex']==2]
    agej=np.array(12*(1997-dpi['birthy'])-dpi['birthm'],dtype=np.int16)[dpi['sex']==1]
    dime=np.array(len(np.array(labor)[0,:]),dtype=np.int16)
    dimem=np.array(len(np.array(laborm)[0,:]),dtype=np.int16)
    ageb=np.reshape(np.repeat(agei,dime),(len(agei),dime))+np.linspace(1,dime,dime,dtype=np.int16)
    agebb=np.reshape(np.repeat(agej,dimem),(len(agej),dimem))+np.linspace(1,dimem,dimem,dtype=np.int16)
    
    #Get weights
    weightsa=np.array(dpi['weight'])[dpi['sex']==2]
    wgt=np.reshape(np.repeat(weightsa,dime),(len(weightsa),dime))
    
    #Get education
    edua=np.array(dpi['college'])[dpi['sex']==2]
    edu=np.reshape(np.repeat(edua,dime),(len(edua),dime))
    
    
    #Get partner education
    #eduap=np.array(dpi['edusp'])[dpi['sex']==2]
    #edup=np.reshape(np.repeat(eduap,dime),(len(eduap),dime))
    
    #Drop nan values
    edu=np.array(edu)[np.isnan(labor)==False]
    status=np.array(status)[np.isnan(labor)==False]
    agebf=np.array(ageb)[np.isnan(labor)==False]
    agebm=np.array(agebb)[np.isnan(laborm)==False]
    wgt=np.array(wgt)[np.isnan(labor)==False]
    labor=np.array(labor)[np.isnan(labor)==False]
    laborm=np.array(laborm)[np.isnan(laborm)==False]
    
    flsm,flsc=np.zeros((3)),np.zeros((3))
    #FLFP at age 25 and 30 and 35  24-26//29-31//34-36
    labor=(labor/120.0)*0.8038
    flsm[0]=np.average(labor[(status>=200) &  (agebf>=300-12) & (agebf<312+12) ],
                 weights=wgt[(status>=200) &  (agebf>=300-12) & (agebf<312+12)  ])
    
    flsc[0]=np.average(labor[(status>=100) & (status<200)  & (agebf>=300-12) & (agebf<312+12) ],
                 weights=wgt[(status>=100) & (status<200)  & (agebf>=300-12) & (agebf<312+12) ])
    
    flsm[1]=np.average(labor[(status>=200) &  (agebf>=360-12) & (agebf<372+12)  ],
                 weights=wgt[(status>=200) & (agebf>=360-12) & (agebf<372+12) ])
    
    flsc[1]=np.average(labor[(status>=100) & (status<200)  & (agebf>=360-12) & (agebf<372+12) ],
                weights=wgt[(status>=100) & (status<200)  & (agebf>=360-12) & (agebf<372+12) ])
    
    
    flsm[2]=np.average(labor[(status>=200) &  (agebf>=420-12) & (agebf<432+12)  ],
                 weights=wgt[(status>=200) & (agebf>=420-12) & (agebf<432+12) ])
    
    flsc[2]=np.average(labor[(status>=100) & (status<200)  & (agebf>=420-12) & (agebf<432+12) ],
                weights=wgt[(status>=100) & (status<200)  & (agebf>=420-12) & (agebf<432+12) ])
    
    data_rel=np.array(np.stack((labor,agebf,status,edu),axis=0).T)     
    data_rel_panda=pd.DataFrame(data=data_rel,columns=['lab','age','status','edu'])
    data_rel_panda.drop(data_rel_panda[data_rel_panda['status']<100].index, inplace=True)
    #data_rel.drop(data_rel[data_rel['edup']>21].index, inplace=True)
    #data_rel.drop(data_rel[data_rel['edup']<0].index, inplace=True)
    
    data_rel_panda['marriage']=1.0
    data_rel_panda.loc[(data_rel_panda['status']>=100) & (data_rel_panda['status']<200),'marriage']=0.0  
    data_rel_panda['labor']=1.0
    data_rel_panda.loc[data_rel_panda['lab']==0,'labor']=0.0  
    data_rel_panda.loc[data_rel_panda['lab']>160,'lab']=160.0
    
    FE_ols = smf.ols(formula='labor ~ edu+marriage+age', data = data_rel_panda.dropna()).fit()  
    #flsc[1]=FE_ols.params['marriage']
    #Women average 68-men 78
   
    
    ########################################
    #FREQENCIES OF CENSORING AGES
    #######################################
        
    def CountFrequency(my_list,weight):    
     
        # Creating an empty dictionary     
        freq = {}    
        for item in my_list:    
            if (item in freq):    
                freq[item] += np.sum(weight[item])   
            else:    
                freq[item] = np.sum(weight[item])   
           
           
        return freq   
           
    #Get censoring age
    filter_col = [col for col in dpi if col.startswith('MAr_stA')]
    censoring=dpi[filter_col]
    
    listv= [x for x in list(censoring.columns.values) if '1994' not in x and
                                                         '1995' not in x and
                                                         '1996' not in x and
                                                         '2018' not in x]
    censoring=np.array(censoring[listv])
    
    
    agei=np.array(12*(1997-dpi['birthy'])-dpi['birthm'],dtype=np.int16)
    dime=np.array(len(censoring[0,:]),dtype=np.int16)
    ageb=np.reshape(np.repeat(agei,dime),(len(agei),dime))+np.linspace(1,dime,dime,dtype=np.int16)
    
    #Keep at certin point
    ageb=ageb[:,220:]
    censoring=censoring[:,220:]
    
    #Where censored
    wcensor=np.argmin(censoring,axis=1)
    wcensor[np.min(censoring,axis=1)>=0]=len(censoring[0,:])-1
    
    #Take right age using a mask
    columns=np.linspace(1,len(censoring[0,:]),len(censoring[0,:]),dtype=np.int16)
    dime=len(wcensor)
    col=np.reshape(np.repeat(columns,dime),(dime,len(columns)),order='F')-1
    mask=(col-wcensor[...,None]==0)
    
    #Finally get it
    dpi['cage']=np.round(ageb[mask]/12)
    
    

       
    #Frequencies for age in the second wave   
    freq_c= CountFrequency(dpi['cage'].tolist(),dpi['weight']) 
    
    #Frequencies of college by gender
    Nfe=np.average(dpi['college'][dpi['sex']==2],weights=dpi['weight'][dpi['sex']==2])
    Nme=np.average(dpi['college'][dpi['sex']==1],weights=dpi['weight'][dpi['sex']==1])
    fem=np.average(dpi['sex']==2,weights=dpi['weight'])
    
    #Put stuff in a dictionary
    freq={'Nfe':Nfe,'Nme':Nme,'fem':fem,'freq_c':freq_c}
       
    #Create a dictionary for saving simulated moments   
    listofTuples = [("hazs" , hazs), ("hazm" , hazm),("hazd" , hazd),("hazde" , hazde),  
                    ("everc" , everc), ("everm" , everm),("everr_e" , everr_e),("everr_ne" , everr_ne),("everr_d" , everr_ne-everr_e),
                    ("flsc" , flsc),("flsm" , flsm),
                    ("beta_edu" , beta_edu),("ref_coh",ref_dut[0:5]),('ratio_mar',ratio_mar),
                    ("freq",freq),("malecorr",malecorr),("femalecorr",femalecorr)]   
    dic_mom=dict(listofTuples)   
       
    del dpi,dco,dma
    return dic_mom   
   
   
   
##############################################   
#Actual moments computation + weighting matrix   
################################################   
   
def dat_moments(sampling_number=5,weighting=False,covariances=False,relative=False,period=3,transform=1):   
       
       
       
           
    #Import Data   
    dpeople=pd.read_csv('full1.csv') 
    dcoh=pd.read_csv('cohab.csv') 
    dmar=pd.read_csv('mar.csv')      

       
    #dpeople['weight']=1.0
    #dcoh['weight']=1.0
    #dmar['weight']=1.0
   

    
    #Call the routine to compute the moments   
    dic=compute(dpeople.copy(),dcoh.copy(),dmar.copy(),period=period,transform=transform)   
    

    
    hazs=dic['hazs']   
    hazm=dic['hazm']   
    hazd=dic['hazd'] 
    hazde=dic['hazde'] 
    everc=dic['everc']   
    everm=dic['everm']
    everr_e=dic['everr_e']
    everr_ne=dic['everr_ne']
    everr_d=dic['everr_d']
    flsc=dic['flsc']
    flsm=dic['flsm']
    beta_edu=dic['beta_edu']
    ref_coh=dic['ref_coh']
    freq=dic['freq']
    ratio_mar=dic['ratio_mar']
    malecorr=dic['malecorr']
    femalecorr=dic['femalecorr']

       
    #Use bootstrap samples to compute the weighting matrix   
    n=len(dpeople)   
    n_c=len(dcoh)
    n_m=len(dmar)
    boot=sampling_number   
    nn=n*boot 
    nn_c=n_c*boot 
    nn_m=n_m*boot
       
    hazsB=np.zeros((len(hazs),boot))   
    hazmB=np.zeros((len(hazm),boot))   
    hazdB=np.zeros((len(hazd),boot))   
    hazdeB=np.zeros((len(hazde),boot))   
    evercB=np.zeros((len(everc),boot))   
    evermB=np.zeros((len(everm),boot))   
    everr_eB=np.zeros((len(everr_e),boot))   
    everr_neB=np.zeros((len(everr_ne),boot))   
    everr_dB=np.zeros((len(everr_d),boot))   
    flscB=np.zeros((len(flsc),boot)) 
    flsmB=np.zeros((len(flsm),boot)) 
    beta_eduB=np.zeros((1,boot))
    ref_cohB=np.zeros((len(ref_coh),boot)) 
    ratio_marB=np.zeros((1,boot))   
    malecorrB=np.zeros((1,boot))
    femalecorrB=np.zeros((1,boot))
       
    aa=dpeople.sample(n=nn,replace=True,weights='weight',random_state=4) 
    a_h=dcoh.sample(n=nn_c,replace=True,random_state=5) 
    a_d=dmar.sample(n=nn_m,replace=True,random_state=6) 
       
    #Make weights useless, we already used them for sampling   
    aa['weight']=1   
    for i in range(boot):   
       
        a1=aa[(i*n):((i+1)*n)].copy().reset_index()   
        a1h=a_h[(i*n_c):((i+1)*n_c)].copy().reset_index() 
        a1d=a_d[(i*n_m):((i+1)*n_m)].copy().reset_index() 
        dicti=compute(a1.copy(),a1h.copy(),a1d.copy(),period=period,transform=transform)  
        
        hazsB[:,i]=dicti['hazs']   
        hazmB[:,i]=dicti['hazm']   
        hazdB[:,i]=dicti['hazd']   
        hazdeB[:,i]=dicti['hazde']   
        evercB[:,i]=dicti['everc']   
        evermB[:,i]=dicti['everm']   
        everr_eB[:,i]=dicti['everr_e'] 
        everr_neB[:,i]=dicti['everr_ne']
        everr_dB[:,i]=dicti['everr_d']
        flscB[:,i]=dicti['flsc'] 
        flsmB[:,i]=dicti['flsm']  
        beta_eduB[:,i]=dicti['beta_edu']   
        ref_cohB[:,i]=dicti['ref_coh']   
        ratio_marB[:,i]=dicti['ratio_mar']
        malecorrB[:,i]=dicti['malecorr']
        femalecorrB[:,i]=dicti['femalecorr']
           
       
    #################################   
    #Confidence interval of moments   
    ################################   
    hazmi=np.array((np.percentile(hazmB,2.5,axis=1),np.percentile(hazmB,97.5,axis=1)))   
    hazsi=np.array((np.percentile(hazsB,2.5,axis=1),np.percentile(hazsB,97.5,axis=1)))   
    hazdi=np.array((np.percentile(hazdB,2.5,axis=1),np.percentile(hazdB,97.5,axis=1)))
    hazdei=np.array((np.percentile(hazdeB,2.5,axis=1),np.percentile(hazdeB,97.5,axis=1)))
    everci=np.array((np.percentile(evercB,2.5,axis=1),np.percentile(evercB,97.5,axis=1)))   
    evermi=np.array((np.percentile(evermB,2.5,axis=1),np.percentile(evermB,97.5,axis=1)))   
    everr_ei=np.array((np.percentile(everr_eB,2.5,axis=1),np.percentile(everr_eB,97.5,axis=1)))   
    everr_nei=np.array((np.percentile(everr_neB,2.5,axis=1),np.percentile(everr_neB,97.5,axis=1))) 
    everr_di=np.array((np.percentile(everr_dB,2.5,axis=1),np.percentile(everr_dB,97.5,axis=1))) 
    flsci=np.array((np.percentile(flscB,2.5,axis=1),np.percentile(flscB,97.5,axis=1))) 
    flsmi=np.array((np.percentile(flsmB,2.5,axis=1),np.percentile(flsmB,97.5,axis=1)))
    beta_edui=np.array((np.percentile(beta_eduB,2.5,axis=1),np.percentile(beta_eduB,97.5,axis=1)))   
    ref_cohi=np.array((np.percentile(ref_cohB,2.5,axis=1),np.percentile(ref_cohB,97.5,axis=1)))   
    ratio_mari=np.array((np.percentile(ratio_marB,2.5,axis=1),np.percentile(ratio_marB,97.5,axis=1)))   
    malecorri=np.array((np.percentile(malecorrB,2.5,axis=1),np.percentile(malecorrB,97.5,axis=1)))   
    femalecorri=np.array((np.percentile(femalecorrB,2.5,axis=1),np.percentile(femalecorrB,97.5,axis=1)))   
    #Do what is next only if you want the weighting matrix      
    if weighting:   
           
        #Compute optimal Weighting Matrix   
        col=np.concatenate((hazmB,hazsB,hazdB,evercB,evermB,everr_dB,flscB,flsmB),axis=0)       
        dim=len(col)   
        W_in=np.zeros((dim,dim))   
        for i in range(dim):   
            for j in range(dim):   
                W_in[i,j]=1/np.cov(col[i,:],col[j,:])[0][1]*(1/(boot-1))   
                 
        if not covariances:   
            W_in = np.diag(np.diag(W_in))   
           
        #Invert   
        W=np.linalg.inv(W_in)   
           
        # normalize   
        #W = W/W.sum()   
          
    elif relative:  
          
        #Compute optimal Weighting Matrix   
        col=np.concatenate((hazm,hazs,hazd,everc,everm,everr_d,flsc,flsm),axis=0)       
        dim=len(col)   
        W=np.zeros((dim,dim))   
        for i in range(dim):   
                W[i,i]=1.0/col[i]**2  
                 
    else:   
           
        #If no weighting, just use sum of squred deviations as the objective function           
        W=np.diag(np.ones(len(hazm)+len(hazs)+len(hazd)+len(everc)+len(everm)+len(everr_d)+len(flsc)+len(flsm)))#two is for fls+beta_unid   
           
    listofTuples = [("hazs" , hazs), ("hazm" , hazm),("hazd" , hazd),("hazde" , hazde),  
                    ("everc" , everc), ("everm" , everm),("everr_e" , everr_e),("everr_ne" , everr_ne),("everr_d" , everr_d),
                    ("flsc" , flsc),("flsm" , flsm),
                    ("beta_edu" , beta_edu),("ref_coh",ref_coh),("ratio_mar",ratio_mar),("malecorr",malecorr),("femalecorr",femalecorr),
                    ("hazsi" , hazsi), ("hazmi" , hazmi),("hazdi" , hazdi),  ("hazdei" , hazdei), 
                    ("everci" , everci), ("evermi" , evermi),("everr_ei" , everr_ei),("everr_nei" , everr_nei),("everr_di" , everr_di),
                    ("flsci" , flsci),("flsmi" , flsmi),
                    ("beta_edui" , beta_edui),("ref_cohi",ref_cohi),("ratio_mari",ratio_mari),
                    ("malecorri",malecorri),("femalecorri",femalecorri),("W",W)]   
    
    packed_stuff=dict(listofTuples)   
    #packed_stuff = (hazm,hazs,hazd,emar,ecoh,fls_ratio,W,hazmi,hazsi,hazdi,emari,ecohi,fls_ratioi,mar,coh,mari,cohi)   
       
       
    
    #Export Moments   
    with open('moments.pkl', 'wb+') as file:   
        pickle.dump(packed_stuff,file)     
           
 
    #Frequency of PSID types ratio divorce
    with open('freq.pkl', 'wb+') as file:   
        pickle.dump(freq,file) 
           
    del packed_stuff,freq   
           
###################################################################   
#If script is run as main, it performs a data comparison with SIPP   
###################################################################   
if __name__ == '__main__':   
       
    import matplotlib.pyplot as plt   
    import matplotlib.backends.backend_pdf   
    import pickle   
   
   
    #Get stuff about moments   
    dat_moments(period=1)   
    
   
   
   
   
   
       
   
   
   
           
           
   
   
   
   
   
