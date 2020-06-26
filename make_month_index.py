# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:24:47 2020

@author: Fabio
"""

#Create month-year matches for labor

import pandas as pd   
import numpy as np   
import pickle    
import statsmodels.formula.api as smf   
from lifelines import CoxPHFitter  

dpeople=pd.read_csv('full.csv') 
dconv=pd.read_excel('Continuous_week_crosswalk_r27.xlsx')


week=np.array(dconv['Week Start:\nMonth'])
cyear=np.array(dconv['Calendar Year \nWeek Number '])
year=np.array(dconv['Week Start: \nYear'])
day=np.array(dconv['Week Start: \nDay'])

dwm=dict()
for y in range(1997,2018):   
    for m in range(1,57): 
        dwm['month_'+str(y)+'_'+str(m)]=-1
        where=(cyear==m) & (year==y) & (day<=21)
        if np.any(where):dwm['month_'+str(y)+'_'+str(m)]=week[where][0]
       
        
for y in range(1997,2018):   
    for m in range(1,13): 
        dpeople['h_'+str(y)+'_'+str(m)]=0.0
        dpeople['c_'+str(y)+'_'+str(m)]=0.0
        for w in range(1,56): 
            
            try:
                if np.any(dpeople.loc[(dwm['month_'+str(y)+'_'+str(w)]==m) & (dpeople['emp_hours_'+str(y)+'_'+str(w)+'_xrnd']>=0),'emp_hours_'+str(y)+'_'+str(w)+'_xrnd']):
                    
                    dpeople.loc[(dwm['month_'+str(y)+'_'+str(w)]==m) & (dpeople['emp_hours_'+str(y)+'_'+str(w)+'_xrnd']>=0),'h_'+str(y)+'_'+str(m)]= \
                    dpeople.loc[(dwm['month_'+str(y)+'_'+str(w)]==m) & (dpeople['emp_hours_'+str(y)+'_'+str(w)+'_xrnd']>=0),'h_'+str(y)+'_'+str(m)]+ \
                    dpeople.loc[(dwm['month_'+str(y)+'_'+str(w)]==m) & (dpeople['emp_hours_'+str(y)+'_'+str(w)+'_xrnd']>=0),'emp_hours_'+str(y)+'_'+str(w)+'_xrnd']
                                                                                            
                
                    dpeople.loc[(dwm['month_'+str(y)+'_'+str(w)]==m) & (dpeople['emp_hours_'+str(y)+'_'+str(w)+'_xrnd']>=0),'c_'+str(y)+'_'+str(m)]= \
                    dpeople.loc[(dwm['month_'+str(y)+'_'+str(w)]==m) & (dpeople['emp_hours_'+str(y)+'_'+str(w)+'_xrnd']>=0),'c_'+str(y)+'_'+str(m)]+ \
                    1.0
            except:
                print('no year')
                
for y in range(1997,2018):   
    for m in range(1,13): 
        dpeople.loc[dpeople['c_'+str(y)+'_'+str(m)]<3,'h_'+str(y)+'_'+str(m)]=np.nan#4*dpeople.loc[dpeople['h_'+str(y)+'_'+str(m)]>0,'h_'+str(y)+'_'+str(m)]/dpeople.loc[dpeople['c_'+str(y)+'_'+str(m)]>0,'c_'+str(y)+'_'+str(m)]
        
        
filter_col = [col for col in dpeople if col.startswith('h_')]
filter_col1 = [col for col in dpeople if col.startswith('MAr_COH')]
filter_col2 = [col for col in dpeople if col.startswith('MAr_stAtus')]
filter_col3=['PuBID_1997']+['PUBID_1997']+ ['id']+['birthy']+['birthm']+['sex']+['college']+['weight']
filter_colm=filter_col+filter_col1+filter_col2+filter_col3
final=dpeople[filter_colm]

#final.to_csv('full1.csv')