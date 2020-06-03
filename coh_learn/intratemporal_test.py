#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:43:41 2020

@author: egorkozlov
"""
import numpy as np

#from scipy.optimize import minimize_scalar as minimize
from scipy.optimize import brentq

#
#
#def util(x,M):
#    assert M>0
#    c = M - x
#    uc = A*c**(1-sig)/(1-sig)
#    ux = alp*(x**lam + kap*(1-lbr)**lam)**((1-xi)/lam)/(1-xi)
#    return uc + ux
#
#
#def foc_res(x,M):
#    #return M - ((A/alp)**(1/sig))*(x**lam + kap*(1-lbr)**lam)**((lam+xi-1)/(lam*sig)) * \
#    #            x**((1-lam)/sig) - x
#    lhs = A*(M-x)**(-sig)
#    rhs = alp*(x**lam + kap*(1-lbr)**lam)**((1-lam-xi)/lam)*x**(lam-1)
#    #lhs = M - x
#    #rhs = ((A/alp)**(1/sig)) * (x**lam + kap*(1-lbr)**lam)**((lam+xi-1)/(lam*sig)) * (x**((1-lam)/sig))
#    return lhs-rhs
#
#def num_sol(mgrid):
#    
#    mgrid = np.atleast_1d(mgrid)
#        
#    xout = np.zeros(mgrid.shape)
#    cout = np.zeros(mgrid.shape)
#    uout = np.zeros(mgrid.shape)
#    
#    for i, M in enumerate(mgrid):
#        obj = lambda x : -util(x,M)    
#        xout[i] = minimize(obj,method='bounded',bounds=(0,M)).x
#        cout[i] = M - xout[i]
#        uout[i] = util(xout[i],M)
#        
#    return xout, cout, uout
#
#def rf_sol_scalar(m_in):
#    res = brentq(lambda x : foc_res(x,m_in), 1e-12,m_in-1e-12)
#    return res


def int_sol(m_in,newton=True,step=1e-6,*,A,alp,sig,xi,lam,kap,lbr):
    m_in = np.atleast_1d(m_in)    
    
    def foc_expression(x):
        return ((A/alp)**(1/sig))*(x**lam + kap*(1-lbr)**lam)**((lam+xi-1)/(lam*sig)) * \
                x**((1-lam)/sig) + x
    
    
    
    rf_res = lambda x : foc_expression(x) - m_in[0]
    
    x0 = 0
    
    for _ in range(5):
        try:
            x0 = brentq(rf_res,step,m_in[0]-step)
            break
        except:
            pass
        
        
    
    xgrid = np.linspace(x0,mgrid.max(),2000)
    
    
    
    
    m_implied = foc_expression(xgrid)
          
    x_interpolated = np.interp(m_in,m_implied,xgrid)
    
    if not newton:
        xout = x_interpolated
    else:
        # perform 1 iteration of Newton method
        def foc_deriv(x):
            logder = ((lam+xi-1)/(sig)) * (x**(lam-1))/(x**lam + kap*(1-lbr)**lam) + \
                ((1-lam)/sig)/x
            return 1 + (foc_expression(x) - x)*logder
    
        f_res = foc_expression(x_interpolated) - m_in
        f_der = foc_deriv(x_interpolated)
    
        x_improved = x_interpolated - f_res/f_der
    
        x_interpolated = x_improved
            
        xout = x_interpolated
    
    def util(x,M):
        c = M - x
        uc = A*c**(1-sig)/(1-sig)
        ux = alp*(x**lam + kap*(1-lbr)**lam)**((1-xi)/lam)/(1-xi)
        return uc + ux
    
    cout = m_in - xout
    uout = util(xout,m_in)
    
    return xout, cout, uout


if __name__ == '__main__':
    
   
    
    A = 1.2
    alp = 0.5
    sig = 1.5
    xi = 1.5
    lam = 0.4
    kap = 0.8
    lbr = 0.5
    
    
    mgrid = np.linspace(0.5,20,100)
    
        
    Minit = 10
    
    
        
    x, c, u = int_sol(mgrid,A=A,alp=alp,sig=sig,xi=xi,lam=lam,kap=kap,lbr=lbr)
    
    #fres = foc_res(interpolation_solution,mgrid)
    #print('Max FOC residual at the analitical solution is {}'.
    #      format(fres.max() ))
        
    


