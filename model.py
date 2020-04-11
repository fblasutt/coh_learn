#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This collects routines for setting up the model.

Created on Tue Oct 29 17:01:07 2019

@author: egorkozlov
"""


#from platform import system

import numpy as np
from timeit import default_timer
import os
import psutil


#if system() != 'Darwin':
from setup import ModelSetup
from graph import graphs
from solver_couples import v_iter_couple
from solver_singles import v_iter_single
from integrator_singles import ev_single
from integrator_couples import ev_couple_m_c


class Model(object):
    def __init__(self,iterator_name='default-timed',verbose=False,
                 solve_till=None,display_v=False,**kwargs):
        self.mstart = self.get_mem()
        self.mlast = self.get_mem()
        self.verbose = verbose
        self.setup = ModelSetup(**kwargs)
        self.dtype = self.setup.dtype
        self.iterator, self.initializer = self._get_iterator(iterator_name)
        self.start = default_timer()
        self.last = default_timer()        
        self.time_dict = dict()        
        self.display_v = display_v
        
        if solve_till is not None:
            T = self.setup.pars['T']
            if solve_till < 0: solve_till = T + solve_till
            print('T is {}, but solving till T = {}'.format(T,solve_till))
        
        
        self.solve(till=solve_till)
        
    def get_mem(self):
        return psutil.Process(os.getpid()).memory_info().rss/1e6
        
        
    def time(self,whatisdone,verbose=True,mintime=0.5):
        # mintime removes actions that take too little
        
        total_time = default_timer() - self.start
        last_time = default_timer() - self.last
        
        total_mem = self.get_mem()
        last_mem  = self.get_mem() - self.mlast
        
        def r(x): return round(x,2)
        
        if verbose and last_time>mintime:
            print('{} is done in {} sec, total {} sec, mem is {} Mb'.format(whatisdone,r(last_time),r(total_time),r(total_mem)))
        self.last = default_timer()
        self.mlast = self.get_mem()
        
        if whatisdone in self.time_dict:
            self.time_dict[whatisdone] = self.time_dict[whatisdone] + [last_time]
        else:
            self.time_dict[whatisdone] = [last_time]
        
    def time_statistics(self,remove_worst=True,remove_single=False):
        
        print('Total time is {}'.format(default_timer() - self.start))
        for what, timelist in self.time_dict.items():
            
            if remove_single and len(timelist) == 1: continue
            
            time_arr = np.array(timelist)
            
            extra = ''
            if remove_worst and time_arr.size > 1:
                time_worst = time_arr.max()
                time_arr = time_arr[time_arr<time_worst]
                extra = ' (excl the worst)'
                
            av_time = round(np.mean(time_arr),2) 
            tot_time = round(np.sum(np.array(timelist)),2) 
            print('On average {} took {}, total {} sec'.format(what,av_time,tot_time,extra))
            
    
    def _get_iterator(self,name='default'):
        # this thing returns two functions: iterate and initialize
        # it can put a timer inside of them
        # it can also do many other things potentially
        
        
        # this is not the best organization but kind of works
        # this allows to use different methods for iteration/initialization
        # as long as they all are specified here or imported
        
        # first we define the iterator
         
        def v_iterator(setup,desc,t,EV=None):
            # this takes integrated future type-specific value function and returns
            # this period value function. Integration is done separately.
            # If None is feeded for EV this assumes that we are in the last period
            # and returns last period value
            #get_ipython().magic('reset -sf')
            # v_iter should work fine ifor EV == None
            
            ushift = self.setup.utility_shifters[desc]            
            
            if desc == 'Female, single' or desc == 'Male, single':
                
                female = (desc == 'Female, single')                
                V, c, x, s = v_iter_single(setup,t,EV,female,ushift)    

                
                if self.display_v: print('at t = {} for {} mean V[0,:] is {}'.format(t,desc,V[0,:].mean()))
                                        
                return {desc: {'V':V,'c':c,'x':x,'s':s}}   
                
             
            elif desc == 'Couple, M' or desc == 'Couple, C':
                
                V, VF, VM, c, x, s, fls, V_all_l = v_iter_couple(setup,t,EV,ushift)    

                      
                if self.display_v: print('at t = {} for {} mean V[0,:,:] is {}'.format(t,desc,V[0,:,:].mean()))
                        
                return {desc: {'V':V,'VF':VF,'VM':VM,'c':c,'x':x,'s':s,'fls':fls,'V_all_l':V_all_l}}
          
            
        # and the integrator   
        
        def v_integrator(setup,desc,t,V_next):
            
            if desc == 'Female, single' or desc == 'Male, single':
                female = (desc == 'Female, single')
                EV, dec = ev_single(setup,V_next,setup.agrid_s,female,t)
            elif desc == 'Couple, M':
                EV, dec = ev_couple_m_c(setup,V_next,t,True)
            elif desc == 'Couple, C':
                EV, dec = ev_couple_m_c(setup,V_next,t,False)
                
                
            if type(EV) is tuple:
                EV0 = EV[0]
                if self.display_v: print('at t = {} for {} mean EV[0,:,:,0] is {}'.format(t,desc,EV0[0,:,:,0].mean()))
            else:
                if self.display_v: print('at t = {} for {} EV[0,:] is {}'.format(t,desc,EV[0,:].mean()))
                
                
            return EV, dec
            
        
        
        # then we wrap them into two routines  
        
        if name == 'default' or name == 'default-timed':
            timed = (name == 'default-timed')
            def iterate(desc,t,Vnext):
                EV, dec = v_integrator(self.setup,desc,t,Vnext)
                if timed: self.time('Integration for {}'.format(desc))
                vout = v_iterator(self.setup,desc,t,EV)
                if timed: self.time('Optimization for {}'.format(desc))
                
                self.wrap_decisions(desc,dec,vout)
                
                return vout, dec
            def initialize(desc,t):
                vout = v_iterator(self.setup,desc,t,None)
                if timed: self.time('Initialization for {}'.format(desc))
                dec = {}
                self.wrap_decisions(desc,dec,vout)
                return vout, dec
        else:
            raise Exception('unsupported name')
            
            
            
        return iterate, initialize
    
    def wrap_decisions(self,desc,dec,vout):
        # This interpolates consumption, savings and labor supply decisions
        # on fine grid for theta that is used for integration and simulations.
        
        v = vout[desc]
        if desc == 'Couple, M' or desc == 'Couple, C':
            
            #cint = self.setup.v_thetagrid_fine.apply(v['c'],axis=2)
            sint = self.setup.v_thetagrid_fine.apply(v['s'],axis=2).astype(self.dtype)
            cint = self.setup.v_thetagrid_fine.apply(v['c'],axis=2).astype(self.dtype)
            xint = self.setup.v_thetagrid_fine.apply(v['x'],axis=2).astype(self.dtype)
            
            Vint = self.setup.v_thetagrid_fine.apply(v['V_all_l'],axis=2).astype(self.dtype)
            
            if Vint.ndim < 4: Vint = Vint[:,:,:,None]
            
            fls = Vint.argmax(axis=3).astype(np.int8)
            
            dec.update({'s':sint,'fls':fls,'c':cint,'x':xint})
            del sint,fls
        else:
            dec.update({'s':v['s'],'c':v['c'],'x':v['x']})
            del v
        
    
    def solve(self,till=None,save=False):
        
        show_mem = self.verbose
        
        T = self.setup.pars['T']
        self.V = list()
        self.decisions = list()
        
        
        if till is None: till = 0
        
        for t in reversed(range(T)):
            Vnow = dict()
            decnow = dict()
            
            Vnext = self.V[0] if t<T-1 else None
            
            for desc in self.setup.state_names:
                if t == T-1:
                    V_d, dec = self.initializer(desc,t)
                else:
                    V_d, dec = self.iterator(desc,t,Vnext)   
                   
                Vnow.update(V_d)
                decnow.update({desc:dec})
                
            self.V = [Vnow] + self.V
            self.decisions = [decnow] + self.decisions
            
            
            #if show_mem:
             #   print('The size of V is {} giga'.format(asizeof(self.V)/1000000000))
              #  print('The size of decisions is {} giga'.format(asizeof(self.decisions)/1000000000))
                
            if t == till: break
            
        if save:
            import pickle
            pickle.dump(self,open('model_save.pkl','wb+'))
        
    def graph(self,ai,zfi,zmi,psii,ti,thi):        
        #Draw some graph of Value and Policy Functions
        V=graphs(self,ai,zfi,zmi,psii,ti,thi)        
        return V
      
        
    
