# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:32:47 2020

@author: Fabio

Statistical Utilities for Structural Estimation

"""

def strata_sample(namevar,dsample,frac=0.1,weights=False,distr=False,tsample=False,fraction=False):
    
    #Take dsample, compute its distribution according to strata namevar with weights
    #and then sample from tsample of dsample. You can also feed already the distribution
    #of strata: in this case distr should be true and the distribution should be
    #fed as dsample
    
    import pandas as pd
    
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
        
    #Check whether work with initial data
    if not isinstance(tsample, pd.DataFrame):

        tsample=dsample
        

   
    #Preparation for actual sampling
    subset=''
    for r,i in zip(namevar,number):
        subset=subset+'(     tsample[{}] == pop_count.index[x][{}] )&'.format(r,i) 
        

        
    #Stratify Sample
    

    if not fraction:  
        
        def map_fun(x):
            p = pop_count
            fraction=tsample[eval(subset[0:-1])].sample(frac=frac)
            return fraction
    
        stratified_sample= list(map(map_fun, range(len(pop_count))))
        return pd.concat(stratified_sample)
    else:
        
        def map_fun(x):
            p = pop_count
            t=tsample
            fraction=eval(subset[0:-1])
            return fraction
    
        stratified_sample= list(map(map_fun, range(len(pop_count))))
        return stratified_sample
    




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
    
    
    tsample1 = pd.DataFrame(index=range(0,240000))
    tsample1['income'] = 0
    tsample1['income'].iloc[19000:40000] = 1
    tsample1['income'].iloc[120000:] = 2
    tsample1['sex'] = np.random.randint(0,2,240000)
    tsample1['age'] = np.random.randint(0,4,240000)
    
    #pop_count = population.groupby(['income', 'sex', 'age'])['income'].count()
    
    
    pop_count = population.groupby(['income', 'sex', 'age'])['income'].count()
    
    sample2=strata_sample(["'income'", "'sex'", "'age'"],population,frac=0.134,tsample=tsample1)
    sample3=strata_sample(["'income'", "'sex'", "'age'"],pop_count,frac=0.134,tsample=tsample1,distr=True,fraction=True)
    
    sample4=list(map(lambda x: population[sample3[x]],range(len(sample3))))