# -*- coding: utf-8 -*-    
"""    
Created on Wed Dec 18 12:52:29 2019    
    
@author: Fabio    
"""    
    
import pandas as pd    
import numpy as np    
import pickle     
import statsmodels.formula.api as smf    
    
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
def compute(hi,d_hrs,d_divo,period=3,transform=1):    
    #compute moments, period    
    #says how many years correspond to one    
    #period    
    
    #Get Date at Interview    
    hi.insert(0, 'IDN', range(0,  len(hi)))    
    hi['res']=hi['NUMUNION']+hi['NUMCOHMR']    
        
    #Get Duration bins    
    bins_d=np.linspace(0,1200,int((100/period)+1))    
    bins_d_label=np.linspace(1,len(bins_d)-1,len(bins_d)-1)    
        
    ##########################    
    #Gen cohabitation Dataset    
    #########################    
        
    #Get date at interview    
    hi['int']=hi['IDATMM']+(hi['IDATYY']-1900)*12    
       
    #Gen age at interview   
    hi['ageint']=round((((hi['IDATYY']-1900)*12+hi['IDATMM'])-hi['birth_month'])/12,0)   
        
    #Take only if cohabitations    
    coh=hi[(hi['NUMUNION']-hi['NUMMAR']>0) |  (hi['NUMCOHMR']>0)].copy()    
        
        
    #Create number of cohabitations    
    coh['num']=0.0   
    for i in range(9):    
        if(np.any(coh['HOWBEG0'+str(i+1)])=='coh'):   
            coh.loc[coh['HOWBEG0'+str(i+1)]=='coh','num']=coh.loc[coh['HOWBEG0'+str(i+1)]=='coh','num']+1.0    
                
    #Expand the data        
    cohe=coh.loc[coh.index.repeat(np.array(coh.num,dtype=np.int32))]    
        
        
    #Link each cohabitation to relationship number    
    cohe['rell'] = cohe.groupby(['IDN']).cumcount()+1    
    cohe['cou']=1    
    cohe['rel']=None    
    for i in range(9):    
        if(np.any(coh['HOWBEG0'+str(i+1)])=='coh'):   
            cohe.loc[(cohe['HOWBEG0'+str(i+1)]=='coh') & (cohe['rell']==cohe['cou']),'rel']=i+1    
            cohe.loc[cohe['HOWBEG0'+str(i+1)]=='coh','cou']= cohe.loc[cohe['HOWBEG0'+str(i+1)]=='coh','cou']+1    
            
    #Get beginning and end of relationhip    
    cohe['beg']=-1    
    cohe['endd']=-1    
    cohe['how']=-1    
    cohe['mar']=-1        
    for i in range(9):    
        cohe.loc[(i+1==cohe['rel']),'beg']=cohe.loc[(i+1==cohe['rel']),'BEGDAT0'+str(i+1)]    
        cohe.loc[(i+1==cohe['rel']),'endd']=cohe.loc[(i+1==cohe['rel']),'ENDDAT0'+str(i+1)]    
        cohe.loc[(i+1==cohe['rel']),'how']=cohe.loc[(i+1==cohe['rel']),'HOWEND0'+str(i+1)]    
        cohe.loc[(i+1==cohe['rel']),'mar']=cohe.loc[(i+1==cohe['rel']),'MARDAT0'+str(i+1)]    
        #add here an indicator of whether it should be unilateral duvorce scenario    
            
    #Get how relationship end    
    cohe['fine']='censored'    
    cohe.loc[cohe['how']=='sep','fine']='sep'    
    cohe.loc[cohe['how']=='div','fine']='mar'    
    cohe.loc[(cohe['how']=='intact') & (cohe['mar']>1),'fine']='mar'    
        
    #Replace censored date if still together    
    cohe['end']=-1    
    cohe.loc[cohe['fine']=='sep','end']=cohe.loc[cohe['fine']=='sep','endd']    
    cohe.loc[cohe['fine']=='mar','end']=cohe.loc[cohe['fine']=='mar','mar']    
    cohe.loc[cohe['fine']=='censored','end']=cohe.loc[cohe['fine']=='censored','int']    
        
    #Duration    
    cohe['dur']=cohe['end']-cohe['beg']    
        
    #Keep if no error for duration    
    cohe=cohe[(cohe['dur']>0) & (cohe['dur']<2000)]    
        
    #Transform Duration in Years    
    cohe['dury'] = pd.cut(x=cohe['dur'], bins=bins_d,labels=bins_d_label)     
        
    cohe['dury']=cohe['dury'].astype(float)   
    
    #Beginning in years
    cohe['begy'] = cohe['beg']/12+1900
   
    
    cohea=pd.DataFrame(data=cohe,columns=['state','begy','dury','fine','unil','rel','birth']) 
    cohea=cohea.round(decimals=0)
    cohea['age']=cohea['begy']-cohea['birth']
    cohea['end']=0
    cohea.loc[(cohea['fine']=='mar'),'end']=1
    cohea['unid']=0
    cohea.loc[(cohea['begy']>=cohea['unil']),'unid']=1
    cohea=cohea[~((cohea['begy']<cohea['unil']) & (cohea['begy']+cohea['dury']>cohea['unil']))]
    cohea1=cohea.drop(['unil','rel','fine','birth'], axis=1) 
    cohea1['age2']=cohea1['age']**2
    cohea1['age3']=cohea1['age']**3
    cohea1=pd.get_dummies(cohea1, columns=['begy'])
    cohea1=pd.get_dummies(cohea1, columns=['state'])
    cohea1=cohea1.drop(['begy_1980.0'], axis=1) 
    cohea1=cohea1.drop(['state_California'], axis=1)     
        
    #Eliminate non useful things    
    del coh    
        
    ##########################    
    #Gen marriage Dataset    
    #########################    
        
    #Take only if marriages    
    mar=hi[hi['NUMMAR']>0].copy()    
        
    #Create number of cohabitations    
    mar['num']=0    
    for i in range(9):    
        mar.loc[mar['MARDAT0'+str(i+1)]>0,'num']=mar.loc[mar['MARDAT0'+str(i+1)]>0,'num']+1    
                
    #Expand the data        
    mare=mar.loc[mar.index.repeat(mar.num)]    
        
        
    #Link each marriage to relationship number    
    mare['rell'] = mare.groupby(['IDN']).cumcount()+1    
    mare['cou']=1    
    mare['rel']=None    
    for i in range(9):    
        mare.loc[(mare['MARDAT0'+str(i+1)]>0) & (mare['rell']==mare['cou']),'rel']=i+1    
        mare.loc[mare['MARDAT0'+str(i+1)]>0,'cou']= mare.loc[mare['MARDAT0'+str(i+1)]>0,'cou']+1    
            
    #Get beginning and end of relationhip    
    mare['beg']=-1    
    mare['endd']=-1    
    mare['how']=-1    
    mare['mar']=-1        
    for i in range(9):    
        mare.loc[(i+1==mare['rel']),'beg']=mare.loc[(i+1==mare['rel']),'MARDAT0'+str(i+1)]    
        mare.loc[(i+1==mare['rel']),'endd']=mare.loc[(i+1==mare['rel']),'ENDDAT0'+str(i+1)]    
        mare.loc[(i+1==mare['rel']),'how']=mare.loc[(i+1==mare['rel']),'HOWEND0'+str(i+1)]    
        
            
    #Get how relationship end    
    mare['fine']='censored'    
    mare.loc[mare['how']=='div','fine']='div'    
        
        
    #Replace censored date if still together    
    mare['end']=-1    
    mare.loc[mare['fine']=='div','end']=mare.loc[mare['fine']=='div','endd']    
    mare.loc[mare['fine']=='censored','end']=mare.loc[mare['fine']=='censored','int']    
        
    #Duration    
    mare['dur']=mare['end']-mare['beg']    
        
    #Keep if no error for duration    
    mare=mare[(mare['dur']>0) & (mare['dur']<2000)]    
        
    # #Transform Duration in Years    
    mare['dury'] = pd.cut(x=mare['dur'], bins=bins_d,labels=bins_d_label)     
        
    mare['dury']=mare['dury'].astype(float)     
     
     
    # #Transform data to be year- 
    # mare['begy']=mare['beg']/12+1900 
    # mare=mare.round({'begy': 0}) 
     
    # #Drop if marriage started after the change in the law 
    # mare=mare[mare['unil']>0] 
    # mare=mare[(mare['begy']<=mare['unil'])] 
    # mare['IDN1']=mare['IDN']*1000+mare['rel'] 
     
    # mare1=mare[['dury','fine','IDN1','IDN','begy','state','unil','birth','rel']] 
    # mare2=mare1.loc[mare1.index.repeat(mare1.dury)]  
     
    # #Beginning year 
    # mare2['t'] = mare2.groupby(['IDN1']).cumcount()+1 
    # mare2['t2']=mare2['t']**2 
    # mare2['t3']=mare2['t']**3 
    # mare2['year']=mare2['begy']+mare2['t']-1 
     
     
    # #dependent variables 
    
    # mare2['max'] = mare2.groupby(['IDN1'])['t'].transform('max') 
    # mare2['div']=0 
    # mare2.loc[(mare2['max']==mare2['t']) & (mare2['fine']=='div'),'div']=1 
     
    # #Unlateral divorce 
    # mare2['unid']=0 
    # mare2.loc[mare2['unil']<=mare2['year'],'unid']=1 
    # #regression 
    # FE_ols = smf.ols(formula='div~ unid+C(state)+C(year)+t', data = mare2[mare2['rel']<=2]).fit()    
     
    del mar    
        
    #############################    
    #Build relationship by month    
    ##############################    
        
    #Eliminate observation if info on beg-end not complete    
    #for i in range(9):    
     #   hi=hi[(np.isfinite(hi['BEGDAT0'+str(i+1)])) & (hi['BEGDAT0'+str(i+1)]<3999)]    
            
    #Get date in time at which the guy is 20,25...,50 (9)    
    for j in range(30):    
        hi['time_'+str(20+(j))]=hi['DOBY']*12+hi['DOBM']+(20+(j))*12    
            
    #Get the status    
    for j in range(30):    
            
        #Create the variable of Status    
        hi['status_'+str(20+(j))]='single'    
            
    
            
        for i in range(9):    
            if(np.any(hi['HOWBEG0'+str(i+1)])!=None):   
                
                #Get if in couple    
                hi.loc[(hi['time_'+str(20+(j))]>=hi['BEGDAT0'+str(i+1)]) & (hi['BEGDAT0'+str(i+1)]<3999) &    
                       (((hi['time_'+str(20+(j))]<=hi['ENDDAT0'+str(i+1)]) & (hi['ENDDAT0'+str(i+1)]>0))  |    
                        (hi['ENDDAT0'+str(i+1)]==0) | (hi['WIDDAT0'+str(i+1)]>0) )   
                       ,'status_'+str(20+(j))]='mar'    
                           
            if(np.any(hi['HOWBEG0'+str(i+1)])=='coh'):              
                #Substitute if actually cohabitation     
                hi.loc[(hi['time_'+str(20+(j))]>=hi['BEGDAT0'+str(i+1)]) & (hi['BEGDAT0'+str(i+1)]<3999) &    
                       (((hi['time_'+str(20+(j))]<=hi['ENDDAT0'+str(i+1)]) & (hi['ENDDAT0'+str(i+1)]>0))  |     
                        (hi['ENDDAT0'+str(i+1)]==0) | (hi['WIDDAT0'+str(i+1)]>0) ) &     
                        (hi['status_'+str(20+(j))]=='mar') &     
                       (hi['HOWBEG0'+str(i+1)]=='coh')    &     
                       ((hi['MARDAT0'+str(i+1)]==0) | (hi['MARDAT0'+str(i+1)]>hi['time_'+str(20+(j))]))         
                       ,'status_'+str(20+(j))]='coh'     
                       
    #Create the variables ever cohabited and ever married    
    for j in range(30):    
            
        #Create the variable of ever married or cohabit    
        hi['everm_'+str(20+(j))]=0.0    
        hi['everc_'+str(20+(j))]=0.0    
            
        for i in range(9):    
               
            #if(np.any(hi['HOWBEG0'+str(i+1)])=='coh'):   
                #Get if ever cohabited     
                #hi.loc[((hi['everc_'+str(20+(max(j-1,0)))]>=0.1) | ((hi['HOWBEG0'+str(i+1)]=='coh') & (hi['time_'+str(20+(j))]>=hi['BEGDAT0'+str(i+1)]))),'everc_'+str(20+(j))]=1.0    
            hi.loc[(hi['everc_'+str(20+(max(j-1,0)))]>=0.1),'everc_'+str(20+(j))]=1.0   
            try:  
                hi.loc[((hi['HOWBEG0'+str(i+1)]=='coh') & (hi['time_'+str(20+(j))]>=hi['BEGDAT0'+str(i+1)])),'everc_'+str(20+(j))]=1.0    
            except:  
                pass  
                   
                #Get if ever married    
            hi.loc[((hi['everm_'+str(20+(max(j-1,0)))]>=0.1) |  (hi['time_'+str(20+(j))]>=hi['MARDAT0'+str(i+1)])),'everm_'+str(20+(j))]=1.0    
                    
    # ######################################    
    # #Build employment by status in 1986    
    # ######################################    
    # empl=hi[(hi['M2DP01']=='FEMALE') & (hi['weeks']<99)].copy()    
    # empl['stat']='single'    
    # empl['dist']=99999    
    # for j in range(7):    
    #     empl.loc[np.abs(empl['time_'+str(20+(j)*5)]-86*12)<empl['dist'],'stat']=hi['status_'+str(20+(j)*5)]    
                
    ##########################    
    #BUILD HAZARD RATES    
    #########################     
        
    #Hazard of Separation    
    hazs=list()    
    hazs=hazards(cohe,'sep','dury','fine',hazs,int(6/period),'SAMWT')    
        
    #Hazard of Marriage    
    hazm=list()    
    hazm=hazards(cohe,'mar','dury','fine',hazm,int(6/period),'SAMWT')    
        
    #Hazard of Divorce    
    hazd=list()    
    hazd=hazards(mare,'div','dury','fine',hazd,int(12/period),'SAMWT')    
     
    #Eventually transform Hazards pooling more years together 
    if transform>1: 
         
        #Divorce 
        hazdp=list() 
        pop=1 
        for i in range(int(12/(period*transform))): 
            haz1=hazd[transform*i]*pop 
            haz2=hazd[transform*i+1]*(pop-haz1) 
            hazdp=[(haz1+haz2)/pop]+hazdp  
            pop=pop-(haz1+haz2) 
        hazdp.reverse()    
        hazdp=np.array(hazdp).T  
        hazd=hazdp 
             
        #Separation and Marriage 
        hazsp=list() 
        hazmp=list() 
        pop=1 
        for i in range(int(6/(period*transform))): 
            hazs1=hazs[transform*i]*pop 
            hazm1=hazm[transform*i]*pop 
             
            hazs2=hazs[transform*i+1]*(pop-hazs1-hazm1) 
            hazm2=hazm[transform*i+1]*(pop-hazs1-hazm1) 
            hazsp=[(hazs1+hazs2)/pop]+hazsp 
            hazmp=[(hazm1+hazm2)/pop]+hazmp 
            pop=pop-(hazs1+hazs2+hazm1+hazm2) 
             
        hazsp.reverse()    
        hazsp=np.array(hazsp).T  
        hazs=hazsp 
         
        hazmp.reverse()    
        hazmp=np.array(hazmp).T  
        hazm=hazmp 
        
    ########################################    
    #Construct share of each relationship    
    #######################################    
    mar=np.zeros(7)    
    coh=np.zeros(7)    
    emar=np.zeros(7)    
    ecoh=np.zeros(7)    
        
    for j in range(7):    
        mar[j]=np.average(hi['status_'+str(20+(j)*3)]=='mar', weights=np.array(hi['SAMWT']))    
        coh[j]=np.average(hi['status_'+str(20+(j)*3)]=='coh', weights=np.array(hi['SAMWT']))    
        emar[j]=np.average(hi['everm_'+str(20+(j)*3)], weights=np.array(hi['SAMWT']))    
        ecoh[j]=np.average(hi['everc_'+str(20+(j)*3)], weights=np.array(hi['SAMWT']))    
            
            
 
    #########################################    
    #Create the age at unilateral divorce+    
    #regression on the effect of unilateral divorce    
    ###########################################    
        
    #Number of relationships for the person    
    hi['numerl']=0.0    
        
    #List of variables to keep    
    keep_var=list()    
    keep_var=keep_var+['numerl']+['state']+['SAMWT']+['IDN']+['unil']+['M2DP01'] 
        
    for i in range(9):    
            
        #Make sure that some relationship of order i exist    
        if (np.any(hi['BEGDAT0'+str(i+1)])):    
                
            #Add relationship order    
            hi['order'+str(i+1)]=np.nan    
            hi.loc[np.isnan(hi['BEGDAT0'+str(i+1)])==False,'order'+str(i+1)]=i+1    
                
            #Add number of relationships    
            hi.loc[np.isnan(hi['BEGDAT0'+str(i+1)])==False,'numerl']+=1.0    
                
            #Get whether the relationship started in marriage or cohabitation    
            hi['imar'+str(i+1)]=np.nan    
            hi.loc[hi['HOWBEG0'+str(i+1)]=='coh','imar'+str(i+1)]=0.0    
            hi.loc[hi['HOWBEG0'+str(i+1)]=='mar','imar'+str(i+1)]=1.0    
                
            #Get age at relationship    
            hi['iage'+str(i+1)]=np.nan    
            hi.loc[np.isnan(hi['BEGDAT0'+str(i+1)])==False,'iage'+str(i+1)]=round((hi['BEGDAT0'+str(i+1)]-hi['birth_month'])/12)    
                
            #Get if unilateral divorce when relationship started    
            hi['unid'+str(i+1)]=np.nan    
            hi.loc[np.isnan(hi['BEGDAT0'+str(i+1)])==False,'unid'+str(i+1)]=0.0    
            hi.loc[(round(hi['BEGDAT0'+str(i+1)]/12+1900)>=hi['unil']) & (hi['unil']>0.1),'unid'+str(i+1)]=1.0    
                
            #Year Realationship Started    
            hi['year'+str(i+1)]=np.nan    
            hi.loc[np.isnan(hi['BEGDAT0'+str(i+1)])==False,'year'+str(i+1)]=round(hi['BEGDAT0'+str(i+1)]/12+1900)    
                           
            #Keep variables    
            keep_var=keep_var+['year'+str(i+1)]+['unid'+str(i+1)]+['iage'+str(i+1)]+['imar'+str(i+1)]+['order'+str(i+1)]    
        
            
            
    #New Dataset to reshape    
    hi2=hi[keep_var]    
        
    #Reahspe Dataset    
    years = ([col for col in hi2.columns if col.startswith('year')])    
    unids = ([col for col in hi2.columns if col.startswith('unid')])    
    iages = ([col for col in hi2.columns if col.startswith('iage')])    
    imars = ([col for col in hi2.columns if col.startswith('imar')])    
    order = ([col for col in hi2.columns if col.startswith('order')])    
        
    hi3 = pd.lreshape(hi2, {'year' : years,'unid' : unids,'iage' : iages,'imar' : imars,'order' : order})     
        
    #Eliminate if missing    
    hi3.replace([np.inf, -np.inf], np.nan)    
    hi3.dropna(subset=['imar','unid'])    
        
    #Regression    
    #FE_ols = smf.wls(formula='imar ~ unid+C(iage)+C(state)+C(year)+C(order)',weights=hi3['SAMWT'], data = hi3.dropna()).fit()    
    FE_ols = smf.ols(formula='imar ~ unid+C(iage)+C(state)+C(year)+C(order)', data = hi3[(hi3['order']<=2) & (hi3['iage']>=20) & (hi3['iage']<60)]).fit()  
    FE_ols = smf.ols(formula='imar ~ unid+C(iage)+C(state)+C(year)+C(order)', data = hi3[(hi3['order']<=5) ]).fit()  
    #FE_ols = smf.wls(formula='imar ~ unid+C(iage)+C(state)+C(year)+C(order)',weights=hi3[(hi3['order']<=3) & (hi3['iage']>=20) & (hi3['iage']<60)]['SAMWT'], data = hi3[(hi3['order']<=3) & (hi3['iage']>=20) & (hi3['iage']<60)]).fit()    
    beta_unid=FE_ols.params['unid']   
     
     
    ####################################à 
    #RISK OF SINGLENESS 
    ##################################### 
    # hi3['exy']=hi3['iage']-17 
    # hi3=hi3[(hi3['exy']>0) & (hi3['iage']<70)] 
    # hi3=hi3[hi3['order']==1].copy() 
    # hiw=hi3.loc[hi3.index.repeat(hi3.exy)]  
     
    # #Beginning year 
    # hiw['t'] = hiw.groupby(['IDN']).cumcount()+1 
    # hiw['t2']=hiw['t']**2 
    # hiw['t3']=hiw['t']**3 
    # hiw['max'] = hiw.groupby(['IDN'])['t'].transform('max') 
    # hiw['year2']=hiw['year']-(hiw['max']-hiw['t']) 
     
     
     
    # #dependent variables 
     
     
    # hiw['rel']=0 
    # hiw.loc[(hiw['max']==hiw['t']),'rel']=1 
     
    # #Unlateral divorce 
    # hiw['unid']=0 
    # hiw.loc[hiw['unil']<=hiw['year2'],'unid']=1 
    # #regression 
    # FE_ols = smf.ols(formula='rel~ unid+C(state)+C(year2)+t+t2', data = hiw).fit()    
     
        
    #Get age at which unilateral divorced was introduced    
    hi['age_unid']=0.0    
    hi.loc[hi['unil']==0,'age_unid']=1000.0    
    hi.loc[hi['unil']!=0,'age_unid']=hi['unil']-hi['birth'] 
          
        
   
        
    #From hi make '-1' if law changed before the guy starts    
    hi.loc[hi['age_unid']<18.0,'age_unid']=18.0  
        
        
 
    ##############################  
    #Compute hours using the psid  
    ################################  
      
    #Account for the survey to be retrospective  
    d_hrs['age']=d_hrs['age']-1.0  
      
    #Trim if hrs>2000  
    d_hrs.loc[d_hrs['wls']>=2000,'wls']=2000  
      
    #First keep the right birth cohorts  
    d_hrs['birth']=d_hrs['year']-d_hrs['age']  
    d_hrs=d_hrs[(d_hrs['birth']>=1951) & (d_hrs['birth']<1955)]  
   
    #Generate variables of interest  
    d_hrs['mar']=-1.0  
    d_hrs.loc[(d_hrs['mls']==1),'mar']=1.0  
    d_hrs.loc[(d_hrs['mls']>1) & (d_hrs['mls']<100),'mar']=0.0  
      
    #Get mean labor supply  
    mean_fls=np.average(d_hrs.loc[(d_hrs['age']>=20) & (d_hrs['age']<=60),'wls']) 
        
    #New dataset   
    d_hrs2=d_hrs[(d_hrs['mar']>=0) & (d_hrs['year']>=1977)]  
     
    #Get Ratio of Female to Male FLP #23-38-53 
    fls_ratio=np.zeros((2))    
    fls_ratio[0]=np.average(d_hrs2.loc[(d_hrs2['mar']==1.0) & (d_hrs['age']>=23) & 
                                    (d_hrs['age']<=38),'wls'])/np.average(d_hrs2.loc[(d_hrs2['mar']==0.0) & 
                                    (d_hrs['age']>=23) & (d_hrs['age']<=38),'wls'])    
                 
    fls_ratio[1]=np.average(d_hrs2.loc[(d_hrs2['mar']==1.0) & (d_hrs['age']>=38) & 
                                    (d_hrs['age']<=53),'wls'])/np.average(d_hrs2.loc[(d_hrs2['mar']==0.0) & 
                                    (d_hrs['age']>=38) & (d_hrs['age']<=53),'wls'])    
                     
                     
    #Get difference in male wages in marriage and cohabitation 
    weightm=d_hrs2.loc[(d_hrs2['mar']==1.0) & (np.isnan(d_hrs2['ln_ly'])==False),'wls'] 
    weightc=d_hrs2.loc[(d_hrs2['mar']==0.0) & (np.isnan(d_hrs2['ln_ly'])==False),'wls'] 
    wage_ratio=np.average(d_hrs2.loc[(d_hrs2['mar']==1.0) & (np.isnan(d_hrs2['ln_ly'])==False),'ln_ly'],weights=weightm)-np.average(d_hrs2.loc[(d_hrs2['mar']==0.0) & (np.isnan(d_hrs2['ln_ly'])==False),'ln_ly'],weights=weightc) 
                
    ####################################### 
    #Get divorce by income using PSID 
    ######################################## 
    divR=np.average(d_divo.loc[(d_divo['ln_ly']>d_divo['wtmedian']),'div']) 
    divP=np.average(d_divo.loc[(d_divo['ln_ly']<d_divo['wtmedian']),'div']) 
    marR=np.average(d_divo.loc[(d_divo['ln_ly']>d_divo['wtmedian']),'mar']) 
    marP=np.average(d_divo.loc[(d_divo['ln_ly']<d_divo['wtmedian']),'mar']) 
    div_ratio=(divR/marR)/(divP/marP) 
      
     
    ######################################## 
    #FREQENCIES 
    ####################################### 
         
    def CountFrequency(my_list):     
      
        # Creating an empty dictionary      
        freq = {}     
        for item in my_list:     
            if (item in freq):     
                freq[item] += 1    
            else:     
                freq[item] = 1    
          
        #for key, value in freq.items():     
         #   print ("% d : % d"%(key, value))     
            
        return freq    
     
     
    def CountFrequencyw(my_list,weight):     
      
        # Creating an empty dictionary      
        freq = {}     
        for item in my_list:     
            if (item in freq):     
                freq[item] += np.sum(weight[item])    
            else:     
                freq[item] = np.sum(weight[item])    
            
            
        return freq    
            
     
    #Modify age unid 
    freq_pc=dict() 
    freq_pc['male'] = CountFrequency(hi.loc[hi['M2DP01']=='MALE','age_unid'].tolist())  
    freq_pc['female'] = CountFrequency(hi.loc[hi['M2DP01']=='FEMALE','age_unid'].tolist())  
    freq_pc['share_female']=np.mean(hi['M2DP01']=='FEMALE') 
        
   
             
             
     
        
   
         
        
    #Create a dictionary for saving simulated moments    
    listofTuples = [("hazs" , hazs), ("hazm" , hazm),("hazd" , hazd),("emar" , emar),   
                    ("ecoh" , ecoh), ("fls_ratio" , fls_ratio),("wage_ratio" , wage_ratio),("div_ratio" , div_ratio), 
                    ("mean_fls" , mean_fls),("mar" , mar),("coh" , coh)]    
    dic_mom=dict(listofTuples)    
        
    del hi,hi2,hi3    
    return dic_mom    
    
    
    
##############################################    
#Actual moments computation + weighting matrix    
################################################    
    
def dat_moments(sampling_number=5,weighting=True,covariances=False,relative=False,period=3,transform=1):    
        
        
        
            
    #Import Data    
    data=pd.read_csv('histo_nsfh.csv')       
    data_h=pd.read_csv('hrs.csv')  
    data_d=pd.read_csv('divo.csv') 
     
        
    data_h['wgt']=1.0 
    data_d['wgt']=1.0 
    #data['SAMWT']=1.0 
    #Subset Data 
    data=data[data['eq']<=1].copy() 
    data=data[(data['birth']>=1951) & (data['birth']<=1955)].copy() 
    
    #Get share men+women by college
    dataf=data[data['M2DP01']=="FEMALE"]
    np.average(dataf['coll'], weights=np.array(dataf['SAMWT']))#0.237
    datam=data[data['M2DP01']=="MALE"]
    np.average(datam['coll'], weights=np.array(datam['SAMWT']))#0.382
    
    
    #Keep by education
    datae=data[data['coll']==1]
    datan=data[data['coll']==0]
    
    #Call the routine to compute the moments    
    dic=compute(datae.copy(),data_h.copy(),data_d.copy(),period=period,transform=transform)       
    emare=dic['emar']    
    ecohe=dic['ecoh'] 
    
    dic=compute(datan.copy(),data_h.copy(),data_d.copy(),period=period,transform=transform)       
    emarn=dic['emar']    
    ecohn=dic['ecoh']    
    
    dic=compute(data.copy(),data_h.copy(),data_d.copy(),period=period,transform=transform)       
    emar=dic['emar']    
    ecoh=dic['ecoh'] 
    
   
            
###################################################################    
#If script is run as main, it performs a data comparison with SIPP    
###################################################################    
if __name__ == '__main__':    
        
    import matplotlib.pyplot as plt    
    import matplotlib.backends.backend_pdf    
    import pickle    
    
    
    #Get stuff about moments    
    dat_moments(period=1)    
     
   
        
    
    
    
    
    
    
        
    
    
    
            
            
    
    
    
    
    
