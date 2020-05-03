# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:11:28 2020

@author: Fabio
"""


import pandas as pd   
import numpy as np   
import pickle    
import statsmodels.formula.api as smf   
from lifelines import CoxPHFitter  
   
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
   
    
    ############################
    # RELATIONSHIP BY AGE
    ##########################

    
    #Get if ever cohabited or married or any by month
    for y in range(1997,2018):
        for m in range(1,13):
            
           
            dpi['mar'+str(12*(y-1997)+m)]=0.0
            dpi['coh'+str(12*(y-1997)+m)]=0.0
            
            
#            dpi.loc[dpi['MAr_stAtus_'+str(y)+'_'+str(m)+'_XrND']=='Married','mar'+str(12*(y-1997)+m)]=1.0
#            dpi.loc[dpi['MAr_stAtus_'+str(y)+'_'+str(m)+'_XrND']=='Never Married, Cohabiting','coh'+str(12*(y-1997)+m)]=1.0
#           
            dpi.loc[dpi['MAr_COHABItAtION_'+str(y)+'_'+str(m)+'_XrND']>=200,'mar'+str(12*(y-1997)+m)]=1.0
            dpi.loc[(dpi['MAr_COHABItAtION_'+str(y)+'_'+str(m)+'_XrND']>=100) &
                    (dpi['MAr_COHABItAtION_'+str(y)+'_'+str(m)+'_XrND']<200),'coh'+str(12*(y-1997)+m)]=1.0            
            
           
    #Create the variables ever cohabited and ever married   
    for y in range(1997,2018):
        for m in range(1,13): 
            
            
            #Create the variable of ever married or cohabit   
            dpi.loc[(dpi['MAr_stAtus_'+str(y)+'_'+str(m)+'_XrND']!=-4) &
                    (dpi['MAr_stAtus_'+str(y)+'_'+str(m)+'_XrND']!=-3)
                    ,'everm'+str(12*(y-1997)+m)]=0.0   
                        
                        
            dpi.loc[(dpi['MAr_stAtus_'+str(y)+'_'+str(m)+'_XrND']!=-4) &
                    (dpi['MAr_stAtus_'+str(y)+'_'+str(m)+'_XrND']!=-3)
                    ,'everc'+str(12*(y-1997)+m)]=0.0    

            dpi.loc[(dpi['everc'+str(max(12*(y-1997)+m-1,1))]>=0.1) |
                    (dpi['coh'+str(12*(y-1997)+m)]>=0.1),'everc'+str(12*(y-1997)+m)]=1.0  

            dpi.loc[(dpi['everm'+str(max(12*(y-1997)+m-1,1))]>=0.1) |
                    (dpi['mar'+str(12*(y-1997)+m)]>=0.1),'everm'+str(12*(y-1997)+m)]=1.0  
            
    #Get ever in a relationship at the "right" time
    for j in range(20,40,5):
        dpi['agec'+str(j)]=np.nan
        dpi['agem'+str(j)]=np.nan
        
    for y in range(1997,2018):
        for m in range(1,13): 
            for j in range(20,40,5):
                            
                dpi.loc[12*y+m-(12*dpi['birthy']+dpi['birthm'])==j*12,'agec'+str(j)]=dpi['everc'+str(12*(y-1997)+m)]
                dpi.loc[12*y+m-(12*dpi['birthy']+dpi['birthm'])==j*12,'agem'+str(j)]=dpi['everm'+str(12*(y-1997)+m)]
            
    for j in range(20,40,5):
        dpi['ager'+str(j)]=dpi['agem'+str(j)]+dpi['agec'+str(j)]
        big=(dpi['ager'+str(j)]>1.0)
        dpi['ager'+str(j)][big]=1.0
        
        
    
        
    #get the moments
    everc=[np.average(dpi['agec'+str(j)][np.isnan(dpi['agec'+str(j)])==False],
     weights=np.array(dpi['weight'])[np.isnan(dpi['agec'+str(j)])==False])   for j in range(20,40,5)]
                
    everm=[np.average(dpi['agem'+str(j)][np.isnan(dpi['agem'+str(j)])==False],
     weights=np.array(dpi['weight'])[np.isnan(dpi['agem'+str(j)])==False])   for j in range(20,40,5)]
                
    everr_e=[np.average(dpi['ager'+str(j)][(np.isnan(dpi['ager'+str(j)])==False) & (dpi['college']==1)],
     weights=np.array(dpi['weight'])[(np.isnan(dpi['ager'+str(j)])==False) & (dpi['college']==1)])   for j in range(20,40,5)]
                
    everr_ne=[np.average(dpi['ager'+str(j)][(np.isnan(dpi['ager'+str(j)])==False) & (dpi['college']==0)],
     weights=np.array(dpi['weight'])[(np.isnan(dpi['ager'+str(j)])==False) & (dpi['college']==0)])   for j in range(20,40,5)]
                
 
    ##########################   
    #BUILD HAZARD RATES   
    #########################    
       
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
    
    

    
    
    #Get age
    agei=np.array(12*(1997-dpi['birthy'])-dpi['birthm'],dtype=np.int16)[dpi['sex']==2]
    dime=np.array(len(np.array(labor)[0,:]),dtype=np.int16)
    ageb=np.reshape(np.repeat(agei,dime),(len(agei),dime))+np.linspace(1,dime,dime,dtype=np.int16)
    
    #Get weights
    weightsa=np.array(dpi['weight'])[dpi['sex']==2]
    wgt=np.reshape(np.repeat(weightsa,dime),(len(weightsa),dime))
    
    #Drop nan values
    status=np.array(status)[np.isnan(labor)==False]
    agebf=np.array(ageb)[np.isnan(labor)==False]
    wgt=np.array(wgt)[np.isnan(labor)==False]
    labor=np.array(labor)[np.isnan(labor)==False]
    
    flsm,flsc=np.zeros((2)),np.zeros((2))
    #FLFP at age 25 and 35 
    flsm[0]=1.0-np.average(labor[(status>=200) & (agebf>=300) & (agebf<312) ]==0,
                 weights=wgt[(status>=200) & (agebf>=300) & (agebf<312) ])
    
    flsc[0]=1.0-np.average(labor[(status>=100) & (status<200)  & (agebf>=300) & (agebf<312)]==0,
                 weights=wgt[(status>=100) & (status<200)  & (agebf>=300) & (agebf<312)])
    
    flsm[1]=1.0-np.average(labor[(status>=200) & (agebf>=420) & (agebf<432) ]==0,
                 weights=wgt[(status>=200) & (agebf>=420) & (agebf<432) ])
    
    flsc[1]=1.0-np.average(labor[(status>=100) & (status<420)  & (agebf>=360) & (agebf<432)]==0,
                weights=wgt[(status>=100) & (status<420)  & (agebf>=360) & (agebf<432)])
    
    
   
    
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
    listofTuples = [("hazs" , hazs), ("hazm" , hazm),("hazd" , hazd),  
                    ("everc" , everc), ("everm" , everm),("everr_e" , everr_e),("everr_ne" , everr_ne),
                    ("flsc" , flsc),("flsm" , flsm),
                    ("beta_edu" , beta_edu),("ref_coh",ref_dut[1:4]),
                    ("freq",freq)]   
    dic_mom=dict(listofTuples)   
       
    del dpi,dco,dma
    return dic_mom   
   
   
   
##############################################   
#Actual moments computation + weighting matrix   
################################################   
   
def dat_moments(sampling_number=5,weighting=False,covariances=False,relative=False,period=3,transform=1):   
       
       
       
           
    #Import Data   
    dpeople=pd.read_csv('full.csv') 
    dcoh=pd.read_csv('cohab.csv') 
    dmar=pd.read_csv('mar.csv')      

       
    dpeople['weight']=1.0
    dcoh['weight']=1.0
    dmar['weight']=1.0
   

    
    #Call the routine to compute the moments   
    dic=compute(dpeople.copy(),dcoh.copy(),dmar.copy(),period=period,transform=transform)   
    

    
    hazs=dic['hazs']   
    hazm=dic['hazm']   
    hazd=dic['hazd']   
    everc=dic['everc']   
    everm=dic['everm']
    everr_e=dic['everr_e']
    everr_ne=dic['everr_ne']
    flsc=dic['flsc']
    flsm=dic['flsm']
    beta_edu=dic['beta_edu']
    ref_coh=dic['ref_coh']
    freq=dic['freq']

       
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
    evercB=np.zeros((len(everc),boot))   
    evermB=np.zeros((len(everm),boot))   
    everr_eB=np.zeros((len(everr_e),boot))   
    everr_neB=np.zeros((len(everr_ne),boot))   
    flscB=np.zeros((len(flsc),boot)) 
    flsmB=np.zeros((len(flsm),boot)) 
    beta_eduB=np.zeros((1,boot))
    ref_cohB=np.zeros((len(ref_coh),boot)) 
       
       
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
        evercB[:,i]=dicti['everc']   
        evermB[:,i]=dicti['everm']   
        everr_eB[:,i]=dicti['everr_e'] 
        everr_neB[:,i]=dicti['everr_ne'] 
        flscB[:,i]=dicti['flsc'] 
        flsmB[:,i]=dicti['flsm']  
        beta_eduB[:,i]=dicti['beta_edu']   
        ref_cohB[:,i]=dicti['ref_coh']   
       
           
       
    #################################   
    #Confidence interval of moments   
    ################################   
    hazmi=np.array((np.percentile(hazmB,2.5,axis=1),np.percentile(hazmB,97.5,axis=1)))   
    hazsi=np.array((np.percentile(hazsB,2.5,axis=1),np.percentile(hazsB,97.5,axis=1)))   
    hazdi=np.array((np.percentile(hazdB,2.5,axis=1),np.percentile(hazdB,97.5,axis=1)))   
    everci=np.array((np.percentile(evercB,2.5,axis=1),np.percentile(evercB,97.5,axis=1)))   
    evermi=np.array((np.percentile(evermB,2.5,axis=1),np.percentile(evermB,97.5,axis=1)))   
    everr_ei=np.array((np.percentile(everr_eB,2.5,axis=1),np.percentile(everr_eB,97.5,axis=1)))   
    everr_nei=np.array((np.percentile(everr_neB,2.5,axis=1),np.percentile(everr_neB,97.5,axis=1)))   
    flsci=np.array((np.percentile(flscB,2.5,axis=1),np.percentile(flscB,97.5,axis=1))) 
    flsmi=np.array((np.percentile(flsmB,2.5,axis=1),np.percentile(flsmB,97.5,axis=1)))
    beta_edui=np.array((np.percentile(beta_eduB,2.5,axis=1),np.percentile(beta_eduB,97.5,axis=1)))   
    ref_cohi=np.array((np.percentile(ref_cohB,2.5,axis=1),np.percentile(ref_cohB,97.5,axis=1)))   
       
    #Do what is next only if you want the weighting matrix      
    if weighting:   
           
        #Compute optimal Weighting Matrix   
        col=np.concatenate((hazmB,hazsB,hazdB,evercB,evermB,everr_eB,everr_neB,flscB,flsmB,beta_eduB,ref_coh),axis=0)       
        dim=len(col)   
        W_in=np.zeros((dim,dim))   
        for i in range(dim):   
            for j in range(dim):   
                W_in[i,j]=(1/(boot-1))*np.cov(col[i,:],col[j,:])[0][1]   
                 
        if not covariances:   
            W_in = np.diag(np.diag(W_in))   
           
        #Invert   
        W=np.linalg.inv(W_in)   
           
        # normalize   
        W = W/W.sum()   
          
    elif relative:  
          
        #Compute optimal Weighting Matrix   
        col=np.concatenate((hazm,hazs,hazd,everc,everm,everr_e,everr_ne,flscB,flsm,beta_edu*np.ones(1),ref_coh),axis=0)       
        dim=len(col)   
        W=np.zeros((dim,dim))   
        for i in range(dim):   
                W[i,i]=1.0/col[i]**2  
                 
    else:   
           
        #If no weighting, just use sum of squred deviations as the objective function           
        W=np.diag(np.ones(len(hazm)+len(hazs)+len(hazd)+len(everc)+len(everm)+len(everr_e)+len(everr_ne)+len(flscB)+len(flsm)+len(ref_coh)+1))#two is for fls+beta_unid   
           
    listofTuples = [("hazs" , hazs), ("hazm" , hazm),("hazd" , hazd),  
                    ("everc" , everc), ("everm" , everm),("everr_e" , everr_e),("everr_ne" , everr_ne),
                    ("flsc" , flsc),("flsm" , flsm),
                    ("beta_edu" , beta_edu),("ref_coh",ref_coh),
                    ("hazsi" , hazsi), ("hazmi" , hazmi),("hazdi" , hazdi),  
                    ("everci" , everci), ("evermi" , evermi),("everr_ei" , everr_ei),("everr_nei" , everr_nei),
                    ("flsci" , flsci),("flsmi" , flsmi),
                    ("beta_edui" , beta_edui),("ref_cohi",ref_cohi),("W",W)]   
    
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
    
   
   
   
   
   
       
   
   
   
           
           
   
   
   
   
   
