#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:07:47 2019

@author: egorkozlov
"""

import os
from time import sleep
from timeit import default_timer
import pickle





def fun_check(x):
    #sleep(5.0)
    return sum([i**2 for i in x]), x[0]

#f = line.split()

try:        
    os.mkdir('Job')
except:
    pass



# optionally this can apply function f_apply to the results
def compute_for_values(values,f_apply=lambda x:x,timeout=24000.0,print_every=10.0,nfails=3):
      
    
    
    
    #assert type(values) is list
    
    names_in = list()
    names_out = list()
    
    # find names of in and out
    # clear files if something already exists
    for ival in range(len(values)):
        namein = 'in{}.pkl'.format(ival)
        
        [os.remove('Job/'+f) for f in os.listdir('Job') 
         if (f.endswith('.pkl') and f.startswith('in')) or 
            (f.endswith('.pkl') and f.startswith('out'))]
        
        names_in.append(namein)
        nameout = 'out{}.pkl'.format(ival)
        names_out.append(nameout)
        
        
        
    def create_in(fname,x):
        file_in = open('Job/'+fname,'wb+')  
        pickle.dump(x,file_in)
        file_in.close()
    
    
    # create bunch of in files and write values of them
    for fname, x in zip(names_in,values):
        create_in(fname,x)
        
        
        
    
    time_start = [0.0]*len(names_in)
    time_took = [0.0]*len(names_in)
    started = [False]*len(names_in)
    finished = [False]*len(names_in)
    fail_count = [0]*len(names_in)
    
    start = default_timer()
    tic = default_timer()
    
    
    while True:
        
        sleep(1.0)
        
        
        ld = os.listdir('Job')
        
        
        li_in =  [f for f in ld if f.endswith('.pkl') and f.startswith('in') ]
        li_out = [f for f in ld if f.endswith('.pkl') and f.startswith('out')]
         
        for i, name in enumerate(names_in):
            if (name not in li_in): 
                if not started[i]:
                    started[i] = True # mark as started
                    time_start[i] = default_timer()
                elif not finished[i]:
                    # check if takes too long
                    time_since = default_timer() - time_start[i]
                    
                    if time_since > timeout: # if does restart
                        print('timeout for i = {}, recreating'.format(i))
                        time_start[i] = 0
                        fail_count[i] += 1
                        started[i] = False
                        if fail_count[i] >= nfails: # if too many fails
                            # simulates output of function computation
                            nameout = 'out{}.pkl'.format(i)
                            create_in(nameout,1.0e6)
                        else:
                            # creates new in file in hope to get the answer again
                            create_in(name,values[i])
                        
            if (names_out[i] in li_out) and (not finished[i]) and (started[i]):
                finished[i] = True
                time_took[i] = default_timer() - time_start[i]
                

        if set(li_out) == set(names_out):
            print('Everything is computed!')
            break # job is done!
            
        # time stats  sometimes if not done
        toc = default_timer()
        
        
        if toc - tic > print_every:
            print('{} running, {} finished, {} not started, running for {:.1f} minutes'.
                  format(sum(started)-sum(finished),sum(finished),len(values)-sum(started),(toc-start)/60))
            tic = toc
            
            
    
    
    fout = list()
    for i, name in enumerate(names_out):
        file = open('Job/'+name,'rb')
        
        # this handles both lists and values
        val = pickle.load(file)
        print(val,fout)
        fout.append(f_apply(val))
        file.close()
        os.remove('Job/'+name)
        
        
        
    
    return fout
    
if __name__ == '__main__':
    vals = [[1,2],[4.0,5.0,6.0],[3.0],[-1,-2,-3]]
    vout = compute_for_values(vals)
    print(vout)
 
    
    
