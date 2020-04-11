#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:54:09 2019

@author: egorkozlov
"""

import matplotlib.pyplot as plt
import numpy as np


q1 = 5
AC1 = 12
MC1 = 12


q2 = 8
AC2 = 14
MC2 = 20

qmin = 5


A = np.array([[1,1/q1,q1],[1,1/q2,q2],[0,1,-qmin**2]])
B = np.array([[AC1,AC2,0]]).T

beta = list(np.dot( np.linalg.inv(A), B).squeeze())

a, b, c = beta[0], beta[1], beta[2]

acmin = a + b/qmin + c*qmin

ac = lambda x : a + b/x + c*x

xmin = 1.0
xmax = 10.0

xval = np.arange(xmin,xmax,0.1)
acval = [ac(x) for x in xval]

plt.grid(which='both',alpha=1.0)
plt.xticks(ticks=np.arange(1.0,10.0,1.0) )
#plt.yticks(ticks=list(range(9,24,1)))
plt.ylim((8,20))
plt.xlim((xmin,xmax))
plt.plot(xval,acval)

Amc = np.array([[1,q2],[1,qmin]])
Bmc = np.array([[MC2,acmin]]).T


bet = list(np.dot(np.linalg.inv(Amc),Bmc).squeeze())
d, e, f = bet[0], bet[1], 0

mc = lambda x : d + e*x + f*(x**2)

mcval = [mc(x) for x in xval]
plt.plot(xval,mcval)
plt.legend()