# -*- coding: utf-8 -*- 
""" 
Implementation of TikTak code as described in: 
     
    'Benchmarking Global Optimizers' 
     by Antoine Arnoud,Fatih Guvenen and Tatjana Kleineberg' 
 
@author: Fabio 
""" 
import sobol_seq 
import numpy as np 
from scipy.optimize import minimize 
from p_client import compute_for_values 
from time import sleep 
import pickle 
from calibration_params import calibration_params 
 
def tiktak(*,N,N_st,xfix=None,skip_global=False,skip_local=False): 
     
    xl, xu, x0, keys, translator = calibration_params(xfix=xfix) 
    #Initial cheks 
    assert len(xl)==len(xu) 
     
    assert N>=N_st 
     
     
    ############################################ 
    #1 INITIALIZATION 
    ########################################### 
     
     
     
    if not skip_global: 
        #First Create a Sobol Sequence 
        init = sobol_seq.i4_sobol_generate(len(xl),N) # generate many draws from uniform 
        #init=init[:,0]    
         
        #Get point on the grid 
        x_init=xl*(1-init)+xu*init 
        x_init=x_init.T 
        x_init=x_init.squeeze() 
     
        #Get fitness of initial points 
         
        pts = [ ('compute',translator(x_init[:,j])) for j in range(N)] 
        fx_init = compute_for_values(pts) 
         
        fx_init = (np.array(fx_init)).squeeze() 
         # !! not the optimizer returns squared value of mdl_resid 
         
        #Sort in ascending order of fitness 
        order=np.argsort(fx_init) 
        fx_init=fx_init[order] 
        x_init=x_init[:,order] 
         
        filer('sobol_results.pkl',(fx_init,x_init),True) 
        print('saved the results succesfully') 
    else: 
        (fx_init,x_init) = filer('sobol_results.pkl',None,False) 
        print('loaded the results from the file') 
         
         
    #Take only the first N_st realization 
    fx_init=fx_init[0:N_st] 
    x_init=x_init[:,0:N_st] 
    
    if skip_local: 
        print('local minimizers are skipped') 
        return x_init[:,0] 
     
     
    #Create a file with sobol sequence points 
    filer('sobol.pkl',x_init,True)     
     
    #List containing parameters and save them in file 
    param=list([ (fx_init[0], x_init[:,0])]) 
    filer('wisdom.pkl',param,True) 
          
     
    vals = [('minimize',(i,N_st,xfix)) for i in range(N_st)] 
     
    compute_for_values(vals,timeout=3600.0) 
     
    param = filer('wisdom.pkl',None,write=False) 
     
    ############################################ 
    #3 TOPPING RULE 
    ########################################### 
    #print(999,ite) 
     
     
    return param[0] 
     
########################################## 
#Functions 
######################################### 
     
#Write on Functionsbtach worker_run.sh 
def filer(filename,array,write=True): 
     
    while True: 
        try: 
            if write: 
                with open(filename, 'wb+') as file: 
                    pickle.dump(array,file) 
            else: 
                with open(filename, 'rb') as file: 
                    array=pickle.load(file) 
                return array 
                 
            break 
        except KeyboardInterrupt: 
            raise KeyboardInterrupt() 
        except: 
            print('Problems opening the file {}'.format(filename)) 
            #sleep(0.5) 
     
 
########################################## 
# UNCOMMENT BELOW FOR TESTING 
#########################################  
#def ff(x): 
#    return 10*3+1+(x[0]**2-10*np.cos(2*np.pi*x[0]))+(x[1]**2-10*np.cos(2*np.pi*x[1]))+(x[2]**2-10*np.cos(2*np.pi*x[2])) 
#param=tiktak(1,100,30,np.array([-25.12,-7.12,-5.12]),np.array([15.12,50.12,1.12]),ff,1e-3,nelder=False,refine=False)