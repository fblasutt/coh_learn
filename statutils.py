# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:32:47 2020

@author: Fabio

Statistical Utilities for Structural Estimation

"""

def strata_sample(namevar,dsample,frac=0.1,weights=False,distr=False,tsample=False):
    
    #Take dsample, compute its distribution according to strata namevar with weights
    #and then sample from tsample of dsample. You can also feed already the distribution
    #of strata: in this case distr should be true and the distribution should be
    #fed as dsample
    
    import pandas as pd
    import numpy as np
    
    #Genrate array for pop count
    namevar_1=[None]*len(namevar)
    number=[None]*len(namevar)
    num=0
    for i in namevar:
        
        namevar_1[num]=namevar[num][1:-1]
        number[num]=str(num)
        num+=1
    

        
    #Genrate distribution
    if not distr:
        
        #Get weights right
        if not weights:
            dsample['weights']=1/len(dsample)
        else:
            dsample['weights']=dsample['weights']/dsample['weights']
        
        pop_count = dsample.groupby(namevar_1)['weights'].count()
        
    else:
         
        pop_count = dsample
        
    #Adjust fraction per group
    fraction_new=pop_count/sum(pop_count)*frac
    fraction_current_temp=tsample.groupby(namevar_1)[namevar_1[0]].count()
    fraction_current=fraction_current_temp/sum(fraction_current_temp)
        
    #Check whether work with initial data
    if not isinstance(tsample, pd.DataFrame):

        tsample=dsample
        

   
    #Preparation for actual sampling
    subset=''
    for r,i in zip(namevar,number):
        subset=subset+'(     tsample[{}] == pop_count.index[x][{}] )&'.format(r,i) 
        

        
    #Stratify Sample
    
    def map_fun(x):
        p = pop_count
        if np.array(fraction_new)[x]/np.array(fraction_current)[x]<1.0:
            return tsample[eval(subset[0:-1])].sample(frac=np.array(fraction_new)[x]/np.array(fraction_current)[x],random_state=5) 
        else:
            return tsample[eval(subset[0:-1])].sample(frac=1.0,random_state=5)
    
    stratified_sample= list(map(map_fun, range(len(pop_count))))

    
    return pd.concat(stratified_sample) 
    




if __name__ == '__main__':   
       
    import pandas as pd
    import numpy as np
    
    # Generate random population (100K)
    
    population = pd.DataFrame(index=range(0,100000))
    population['income'] = 0
    population['income'].iloc[39000:80000] = 1
    population['income'].iloc[80000:] = 2
    population['sex'] = np.random.randint(0,2,100000)
    population['age'] = np.random.randint(0,4,100000)
    
    
    tsample = pd.DataFrame(index=range(0,240000))
    tsample['income'] = 0
    tsample['income'].iloc[19000:40000] = 1
    tsample['income'].iloc[120000:] = 2
    tsample['sex'] = np.random.randint(0,2,240000)
    tsample['age'] = np.random.randint(0,4,240000)
    
    #pop_count = population.groupby(['income', 'sex', 'age'])['income'].count()
    
    
    pop_count = population.groupby(['income', 'sex', 'age'])['income'].count()
    
    sample2=strata_sample(["'income'", "'sex'", "'age'"],population,frac=0.134,tsample=tsample)
    
    sample3=strata_sample(["'income'", "'sex'", "'age'"],pop_count,frac=0.134,tsample=tsample,distr=True)
    