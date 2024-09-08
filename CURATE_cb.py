#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import all the necessary libraries

import pandas as pd
import numpy as np
from random import seed
from random import randint
import matplotlib.pyplot as plt
import scipy.optimize as opt
#from math import comb
import scipy
from random import seed
from random import randint
from scipy.special import comb
from scipy.integrate import quad
from scipy.stats import norm
from scipy import stats 
import networkx as nx
from scipy.optimize import fsolve
from itertools import combinations, permutations
import logging
_logger = logging.getLogger(__name__)
#from gsq.ci_tests import ci_test_bin, ci_test_dis
from collections import namedtuple
import warnings
from numpy import ma
from scipy.stats import mstats_basic
from scipy.stats._stats import _kendall_dis
import scipy.special as special
from scipy.optimize import minimize
import numpy as np
#from gsq.ci_tests import ci_test_bin, ci_test_dis
from scipy.stats import chi2



# In[37]:


#Select a dataset from: ['cancer', 'earthquake', 'survey']
#dataset = input("Enter the dataset name ? \n")
dataset = 'cancer'

#Select an algorithm from: ['curate', 'pc', 'privpc', 'svt', 'em']
#algo = input("Enter the algorithm name ? \n")
algo = 'curate'

#
delta_prime = 1e-14
delta_ad = 1e-14
delta_total = 1e-10

#N = total number of samples in the dataset 100K
#q = sub-sampling rate
#T = test threshold
N = 100000
#q = float(input("Subsampling rate : \n"))
q = 1.0
#T = float(input("Threshold : \n"))
T = 0.05
n = q*N
alpha = T
beta = 0.2
maxi = 0
for i in range(2,int(n)):
    val = (np.abs(((9*i*i)+(9*i))/((4*i)+14))-np.abs(((9*i*i)/((4*i)+10))))*(2/(np.sqrt(2*np.pi)*(i-1)))
    if val>=maxi:
        maxi=val


if algo in ['curate']:
    epsilonpriv = None
    eps_total = float(input("Total Budget ? \n"))
    eps_rem = eps_total
    eps_limit = eps_total
    print(eps_total)
elif algo in ['privpc','svt','em']:
    epsilonpriv = float(input("Budget for each CI test ? \n"))
    print(epsilonpriv)
else:
    epsilonpriv = None
    

if dataset in ['cancer','earthquake','asia']:
    taskval = 'bin'
else:
    taskval = 'dis'

ground_truth = {
    "asia": [(0, 1), (1, 0), (1, 5), (2, 3), (2, 4), (3, 2), (3, 5), (4, 2), (4, 7), (7, 4)],
    "cancer": [(0, 2), (1, 2), (2, 3), (4, 2)],
    "earthquake": [(0, 2), (1, 2), (2, 3), (2, 4)],
    "survey": [(2, 3), (2, 4), (4, 2), (4, 5), (5, 3), (5, 4)],
    # "sachs": [(0, 1), (0, 7), (1, 0), (1, 3), (1, 7), (2, 7), (2, 8), (3, 1), (3, 7), (3, 8), (3, 10), (4, 7), (4, 8), (5, 6), (5, 9), (6, 5), (6, 9), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 8), (7, 10), (8, 2), (8, 3), (8, 4), (8, 7), (8, 10), (9, 5), (9, 6), (10, 3), (10, 7), (10, 8)],
    "sachs": [(0, 1), (1, 0), (1, 3), (1, 7), (2, 8), (3, 1), (3, 7), (3, 8), (3, 10), (4, 7), (4, 10), (5, 6), (5, 9), (6, 5), (6, 9), (7, 1), (7, 3), (7, 10), (8, 2), (8, 3), (8, 7), (9, 5), (9, 6)],
    "child": [(0, 18), (1, 7), (2, 7), (2, 8), (2, 10), (2, 16), (3, 9), (3, 17), (4, 10), (4, 17), (4, 18), (5, 12), (6, 14), (8, 2), (8, 12), (9, 3), (11, 13), (11, 14), (11, 15), (11, 16), (11, 18), (12, 8), (13, 11), (13, 18), (13, 19), (14, 6), (14, 11), (15, 1), (15, 11), (16, 1), (16, 2), (16, 11), (17, 3), (17, 4), (17, 5), (19, 5), (19, 13)],
    # 'child': [(1, 7), (2, 7), (2, 8), (3, 9), (3, 17), (4, 10), (5, 12), (6, 14), (9, 3), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (13, 11), (13, 19), (14, 6), (14, 11), (15, 1), (15, 11), (16, 1), (16, 2), (16, 11), (17, 2), (17, 3), (17, 4), (17, 5), (17, 11), (18, 4), (18, 11), (19, 5), (19, 11), (19, 13)],
    # 'alarm': [(0, 5), (1, 4), (2, 4), (3, 4), (3, 5), (3, 6), (4, 1), (4, 2), (4, 3), (4, 5), (5, 4), (5, 6), (6, 35), (7, 8), (9, 11), (10, 9), (10, 11), (11, 9), (12, 33), (13, 14), (14, 13), (14, 33), (14, 36), (16, 25), (17, 24), (17, 30), (18, 19), (19, 20), (20, 33), (21, 22), (22, 21), (22, 23), (23, 20), (24, 17), (24, 23), (24, 30), (24, 31), (26, 29), (27, 28), (28, 27), (28, 29), (29, 25), (30, 15), (30, 17), (30, 24), (30, 31), (31, 19), (31, 24), (31, 30), (31, 32), (32, 15), (32, 31), (32, 33), (34, 8), (34, 9), (34, 11), (34, 33), (34, 35), (35, 36)],
    "alarm": [(0, 5), (0, 26), (1, 4), (2, 4), (3, 4), (3, 6), (4, 1), (4, 2), (4, 3), (5, 0), (5, 6), (6, 3), (6, 5), (6, 35), (7, 8), (7, 26), (9, 10), (9, 11), (10, 9), (10, 11), (11, 9), (11, 10), (12, 33), (13, 14), (14, 33), (14, 36), (16, 30), (16, 31), (17, 16), (17, 30), (17, 31), (18, 14), (18, 19), (19, 18), (19, 20), (20, 19), (20, 33), (21, 22), (22, 21), (22, 23), (24, 23), (25, 16), (25, 30), (27, 28), (28, 27), (28, 29), (29, 26), (29, 28), (30, 15), (30, 16), (30, 17), (30, 25), (30, 31), (31, 30), (31, 32), (32, 15), (32, 33), (34, 8), (34, 33), (34, 35), (35, 36)],
}


# In[39]:




# In[18]:


def onlinebudgeting(budget, edges, order):
    #eps_total = budget
    if dataset in ['cancer', 'earthquake']:
        d = 5
        if order == 1:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))))))
            d1 = edges*comb(d-2,1)
            d2 = edges*comb(d-2,2)
            d3 = edges*comb(d-2,3)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0])
                                                           +(d2*x[1]*x[1])
                                                           +(d3*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[0]*x[0])))+
                                                            (np.sqrt(2*np.log(1/delta_prime)*(d2*x[1]*x[1])))+
                                                            (np.sqrt(2*np.log(1/delta_prime)*(d3*x[2]*x[2]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s), method = 'SLSQP',
                                        bounds = bnds, constraints=cons)
        elif order == 2:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))))))
            d2 = edges*comb(d-2,2)
            d3 = edges*comb(d-2,3)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d2*x[0]*x[0])
                                                           +(d3*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d2*x[0]*x[0])))+
                                                            (np.sqrt(2*np.log(1/delta_prime)*(d3*x[1]*x[1]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b))
            result = opt.minimize(fun, (s,s), method = 'SLSQP',
                                        bounds = bnds, constraints=cons)
        elif order == 3:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x)-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x)-1))))/delta)))))))
            d3 = edges*comb(d-2,3)
            cons = ({'type': 'ineq', 'fun': lambda x:  budget - (((d3*x*x))
                                                           +np.sqrt(2*np.log(1/delta_prime)*
                                                                   ((d3*x*x))))})
            a = 0
            s = 0
            b = None
            bnds = (a,b)
            result = opt.minimize(fun, s, method = 'SLSQP',
                                  constraints=cons)
                                        #bounds = bnds, constraints=cons)
    if dataset in ['survey']:
        d = 5
        if order == 1:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))))))
            
            d1 = edges*comb(d-2,1)
            d2 = edges*comb(d-2,2)
            d3 = edges*comb(d-2,3)
            d4 = edges*comb(d-2,4)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0])
                                                           +(d2*x[1]*x[1])
                                                           +(d3*x[2]*x[2])
                                                           +(d4*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d2*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d3*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d4*x[3]*x[3]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
            
        elif order == 2:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))))))
            
            d2 = edges*comb(d-2,2)
            d3 = edges*comb(d-2,3)
            d4 = edges*comb(d-2,4)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d2*x[0]*x[0])
                                                           +(d3*x[1]*x[1])
                                                           +(d4*x[2]*x[2]))
                                                           +np.sqrt(2*np.log(1/delta_prime)*(d2*x[0]*x[0])
                                                                   +(np.sqrt(2*np.log(1/delta_prime)*(d3*x[1]*x[1])))
                                                                   +(np.sqrt(2*np.log(1/delta_prime)*(d4*x[2]*x[2])))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
            
        elif order == 3:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))))))
            
            d3 = edges*comb(d-2,3)
            d4 = edges*comb(d-2,4)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d3*x[0]*x[0])
                                                           +(d4*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d3*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d4*x[1]*x[1]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b))
            result = opt.minimize(fun, (s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
            
            
        elif order == 4:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))))))
            d1 = edges*comb(d-2,4)
            cons = ({'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0]))
                                                           +np.sqrt(2*np.log(1/delta_prime)*
                                                                   ((d1*x[0]*x[0]))))})
            a = 0
            s = 0
            b = None
            result = opt.minimize(fun, s, method = 'SLSQP',
                                  constraints=cons)
                                        #bounds = bnds, constraints=cons)
                                        
    elif dataset in ['asia']:
        d = 8
        if order == 1:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))))))
            d1 = edges*comb(d-2,1)
            d2 = edges*comb(d-2,2)
            d3 = edges*comb(d-2,3)
            d4 = edges*comb(d-2,4)
            d5 = edges*comb(d-2,5)
            d6 = edges*comb(d-2,6)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0])
                                                           +(d2*x[1]*x[1])
                                                           +(d3*x[2]*x[2])
                                                           +(d4*x[3]*x[3])
                                                           +(d5*x[4]*x[4])
                                                           +(d6*x[5]*x[5]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d2*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d3*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d4*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d5*x[4]*x[4]))
                                                           +((np.sqrt(2*np.log(1/delta_prime)*d6*x[5]*x[5]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
            
        elif order == 2:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))))))
            d1 = edges*comb(d-2,2)
            d2 = edges*comb(d-2,3)
            d3 = edges*comb(d-2,4)
            d4 = edges*comb(d-2,5)
            d5 = edges*comb(d-2,6)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0])
                                                           +(d2*x[1]*x[1])
                                                           +(d3*x[2]*x[2])
                                                           +(d4*x[3]*x[3])
                                                           +(d5*x[4]*x[4]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d2*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d3*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d4*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d5*x[4]*x[4])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
            
        elif order == 3:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))))))
            d1 = edges*comb(d-2,3)
            d2 = edges*comb(d-2,4)
            d3 = edges*comb(d-2,5)
            d4 = edges*comb(d-2,6)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0])
                                                           +(d2*x[1]*x[1])
                                                           +(d3*x[2]*x[2])
                                                           +(d4*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d2*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d3*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d4*x[3]*x[3])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 4:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))))))
            d1 = edges*comb(d-2,4)
            d2 = edges*comb(d-2,5)
            d3 = edges*comb(d-2,6)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0])
                                                           +(d2*x[1]*x[1])
                                                           +(d3*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d2*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d3*x[2]*x[2])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 5:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))))))
            d1 = edges*comb(d-2,5)
            d2 = edges*comb(d-2,6)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0])
                                                           +(d2*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d2*x[1]*x[1])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b))
            result = opt.minimize(fun, (s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 6:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))))))
            d1 = edges*comb(d-2,6)
            cons = ({'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0]))
                                                           +np.sqrt(2*np.log(1/delta_prime)*
                                                                   ((d1*x[0]*x[0]))))})
            a = 0
            s = 0
            b = None
            result = opt.minimize(fun, s, method = 'SLSQP',
                                  constraints=cons)
    elif dataset in ['sachs']:
        d = 11
        if order == 1:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))))))
                        
            d1 = comb(d,2)*comb(d-2,1)
            d2 = comb(d,2)*comb(d-2,2)
            d3 = comb(d,2)*comb(d-2,3)
            d4 = comb(d,2)*comb(d-2,4)
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0])
                                                           +(d2*x[1]*x[1])
                                                           +(d3*x[2]*x[2])
                                                           +(d4*x[3]*x[3])
                                                           +(d5*x[4]*x[4])
                                                           +(d6*x[5]*x[5])
                                                           +(d7*x[6]*x[6])
                                                           +(d8*x[7]*x[7])
                                                           +(d9*x[8]*x[8]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d2*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d3*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d4*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d5*x[4]*x[4]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d6*x[5]*x[5]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d7*x[6]*x[6]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d8*x[7]*x[7]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d9*x[8]*x[8])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 2:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))))))
                        
            d2 = comb(d,2)*comb(d-2,2)
            d3 = comb(d,2)*comb(d-2,3)
            d4 = comb(d,2)*comb(d-2,4)
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d2*x[0]*x[0])
                                                           +(d3*x[1]*x[1])
                                                           +(d4*x[2]*x[2])
                                                           +(d5*x[3]*x[3])
                                                           +(d6*x[4]*x[4])
                                                           +(d7*x[5]*x[5])
                                                           +(d8*x[6]*x[6])
                                                           +(d9*x[7]*x[7]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d2*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d3*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d4*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d5*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d6*x[4]*x[4]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d7*x[5]*x[5]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d8*x[6]*x[6]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d9*x[7]*x[7])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 3:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))))))
            d3 = comb(d,2)*comb(d-2,3)
            d4 = comb(d,2)*comb(d-2,4)
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d3*x[0]*x[0])
                                                           +(d4*x[1]*x[1])
                                                           +(d5*x[2]*x[2])
                                                           +(d6*x[3]*x[3])
                                                           +(d7*x[4]*x[4])
                                                           +(d8*x[5]*x[5])
                                                           +(d9*x[6]*x[6]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d3*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d4*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d5*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d6*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d7*x[4]*x[4]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d8*x[5]*x[5]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d9*x[6]*x[6])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 4:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))))))
            d4 = comb(d,2)*comb(d-2,4)
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d4*x[0]*x[0])
                                                           +(d5*x[1]*x[1])
                                                           +(d6*x[2]*x[2])
                                                           +(d7*x[3]*x[3])
                                                           +(d8*x[4]*x[4])
                                                           +(d9*x[5]*x[5]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d4*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d5*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d6*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d7*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d8*x[4]*x[4]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d9*x[5]*x[5])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 5:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))))))
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d5*x[0]*x[0])
                                                           +(d6*x[1]*x[1])
                                                           +(d7*x[2]*x[2])
                                                           +(d8*x[3]*x[3])
                                                           +(d9*x[4]*x[4]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d5*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d6*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d7*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d8*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d9*x[4]*x[4])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 6:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))))))
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d6*x[0]*x[0])
                                                           +(d7*x[1]*x[1])
                                                           +(d8*x[2]*x[2])
                                                           +(d9*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d6*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d7*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d8*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d9*x[3]*x[3])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 7:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))))))
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d7*x[0]*x[0])
                                                           +(d8*x[1]*x[1])
                                                           +(d9*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d7*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d8*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d9*x[2]*x[2])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 8:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))))))
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d8*x[0]*x[0])
                                                           +(d9*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d8*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d9*x[1]*x[1])))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b))
            result = opt.minimize(fun, (s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 9:
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))))))
            d9 = comb(d,2)*comb(d-2,9)
            cons = ({'type': 'ineq', 'fun': lambda x:  budget - (((d9*x[0]*x[0]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[0]*x[0]))))})
            a = 0
            s = 0
            b = None
            result = opt.minimize(fun, s, method = 'SLSQP',
                                  constraints=cons)
    elif dataset in ['child']:
        if order == 1:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[15])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[16])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[17])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[15])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[16])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[17])-1))))/delta)))))))
            d1 = comb(d,2)*comb(d-2,1)
            d2 = comb(d,2)*comb(d-2,2)
            d3 = comb(d,2)*comb(d-2,3)
            d4 = comb(d,2)*comb(d-2,4)
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  x[9] - x[10]},
                {'type': 'ineq', 'fun': lambda x:  x[10] - x[11]},
                {'type': 'ineq', 'fun': lambda x:  x[11] - x[12]},
                {'type': 'ineq', 'fun': lambda x:  x[12] - x[13]},
                {'type': 'ineq', 'fun': lambda x:  x[13] - x[14]},
                {'type': 'ineq', 'fun': lambda x:  x[14] - x[15]},
                {'type': 'ineq', 'fun': lambda x:  x[15] - x[16]},
                {'type': 'ineq', 'fun': lambda x:  x[16] - x[17]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d1*x[0]*x[0])
                                                           +(d2*x[1]*x[1])
                                                           +(d3*x[2]*x[2])
                                                           +(d4*x[3]*x[3])
                                                           +(d5*x[4]*x[4])
                                                           +(d6*x[5]*x[5])
                                                           +(d7*x[6]*x[6])
                                                           +(d8*x[7]*x[7])
                                                           +(d9*x[8]*x[8])
                                                           +(d10*x[9]*x[9])
                                                           +(d11*x[10]*x[10])
                                                           +(d12*x[11]*x[11])
                                                           +(d13*x[12]*x[12])
                                                           +(d14*x[13]*x[13])
                                                           +(d15*x[14]*x[14])
                                                           +(d16*x[15]*x[15])
                                                           +(d17*x[16]*x[16])
                                                           +(d18*x[17]*x[17]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d2*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d3*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d4*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d5*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d6*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d7*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d8*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[9]*x[9])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[10]*x[10])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[11]*x[11])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[12]*x[12])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[13]*x[13])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[14]*x[14])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[15]*x[15])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[16]*x[16])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[17]*x[17]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)

        elif order == 2:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[15])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[16])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[15])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[16])-1))))/delta)))))))
        
            d2 = comb(d,2)*comb(d-2,2)
            d3 = comb(d,2)*comb(d-2,3)
            d4 = comb(d,2)*comb(d-2,4)
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  x[9] - x[10]},
                {'type': 'ineq', 'fun': lambda x:  x[10] - x[11]},
                {'type': 'ineq', 'fun': lambda x:  x[11] - x[12]},
                {'type': 'ineq', 'fun': lambda x:  x[12] - x[13]},
                {'type': 'ineq', 'fun': lambda x:  x[13] - x[14]},
                {'type': 'ineq', 'fun': lambda x:  x[14] - x[15]},
                {'type': 'ineq', 'fun': lambda x:  x[15] - x[16]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d2*x[0]*x[0])
                                                           +(d3*x[1]*x[1])
                                                           +(d4*x[2]*x[2])
                                                           +(d5*x[3]*x[3])
                                                           +(d6*x[4]*x[4])
                                                           +(d7*x[5]*x[5])
                                                           +(d8*x[6]*x[6])
                                                           +(d9*x[7]*x[7])
                                                           +(d10*x[8]*x[8])
                                                           +(d11*x[9]*x[9])
                                                           +(d12*x[10]*x[10])
                                                           +(d13*x[11]*x[11])
                                                           +(d14*x[12]*x[12])
                                                           +(d15*x[13]*x[13])
                                                           +(d16*x[14]*x[14])
                                                           +(d17*x[15]*x[15])
                                                           +(d18*x[16]*x[16]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d2*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d3*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d4*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d5*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d6*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d7*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d8*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[9]*x[9])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[10]*x[10])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[11]*x[11])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[12]*x[12])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[13]*x[13])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[14]*x[14])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[15]*x[15])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[16]*x[16]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 3:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[15])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[15])-1))))/delta)))))))
        
            d3 = comb(d,2)*comb(d-2,3)
            d4 = comb(d,2)*comb(d-2,4)
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  x[9] - x[10]},
                {'type': 'ineq', 'fun': lambda x:  x[10] - x[11]},
                {'type': 'ineq', 'fun': lambda x:  x[11] - x[12]},
                {'type': 'ineq', 'fun': lambda x:  x[12] - x[13]},
                {'type': 'ineq', 'fun': lambda x:  x[13] - x[14]},
                {'type': 'ineq', 'fun': lambda x:  x[14] - x[15]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d3*x[0]*x[0])
                                                           +(d4*x[1]*x[1])
                                                           +(d5*x[2]*x[2])
                                                           +(d6*x[3]*x[3])
                                                           +(d7*x[4]*x[4])
                                                           +(d8*x[5]*x[5])
                                                           +(d9*x[6]*x[6])
                                                           +(d10*x[7]*x[7])
                                                           +(d11*x[8]*x[8])
                                                           +(d12*x[9]*x[9])
                                                           +(d13*x[10]*x[10])
                                                           +(d14*x[11]*x[11])
                                                           +(d15*x[12]*x[12])
                                                           +(d16*x[13]*x[13])
                                                           +(d17*x[14]*x[14])
                                                           +(d18*x[15]*x[15]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d3*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d4*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d5*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d6*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d7*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d8*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[9]*x[9])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[10]*x[10])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[11]*x[11])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[12]*x[12])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[13]*x[13])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[14]*x[14])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[15]*x[15]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 4:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta)))))))
        
            d4 = comb(d,2)*comb(d-2,4)
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  x[9] - x[10]},
                {'type': 'ineq', 'fun': lambda x:  x[10] - x[11]},
                {'type': 'ineq', 'fun': lambda x:  x[11] - x[12]},
                {'type': 'ineq', 'fun': lambda x:  x[12] - x[13]},
                {'type': 'ineq', 'fun': lambda x:  x[13] - x[14]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d4*x[0]*x[0])
                                                           +(d5*x[1]*x[1])
                                                           +(d6*x[2]*x[2])
                                                           +(d7*x[3]*x[3])
                                                           +(d8*x[4]*x[4])
                                                           +(d9*x[5]*x[5])
                                                           +(d10*x[6]*x[6])
                                                           +(d11*x[7]*x[7])
                                                           +(d12*x[8]*x[8])
                                                           +(d13*x[9]*x[9])
                                                           +(d14*x[10]*x[10])
                                                           +(d15*x[11]*x[11])
                                                           +(d16*x[12]*x[12])
                                                           +(d17*x[13]*x[13])
                                                           +(d18*x[14]*x[14]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d4*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d5*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d6*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d7*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d8*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[9]*x[9])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[10]*x[10])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[11]*x[11])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[12]*x[12])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[13]*x[13])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[14]*x[14]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 5:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))))))
            d5 = comb(d,2)*comb(d-2,5)
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  x[9] - x[10]},
                {'type': 'ineq', 'fun': lambda x:  x[10] - x[11]},
                {'type': 'ineq', 'fun': lambda x:  x[11] - x[12]},
                {'type': 'ineq', 'fun': lambda x:  x[12] - x[13]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d5*x[0]*x[0])
                                                           +(d6*x[1]*x[1])
                                                           +(d7*x[2]*x[2])
                                                           +(d8*x[3]*x[3])
                                                           +(d9*x[4]*x[4])
                                                           +(d10*x[5]*x[5])
                                                           +(d11*x[6]*x[6])
                                                           +(d12*x[7]*x[7])
                                                           +(d13*x[8]*x[8])
                                                           +(d14*x[9]*x[9])
                                                           +(d15*x[10]*x[10])
                                                           +(d16*x[11]*x[11])
                                                           +(d17*x[12]*x[12])
                                                           +(d18*x[13]*x[13]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d5*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d6*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d7*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d8*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[9]*x[9])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[10]*x[10])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[11]*x[11])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[12]*x[12])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[13]*x[13]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 6:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))))))
            d6 = comb(d,2)*comb(d-2,6)
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  x[9] - x[10]},
                {'type': 'ineq', 'fun': lambda x:  x[10] - x[11]},
                {'type': 'ineq', 'fun': lambda x:  x[11] - x[12]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d6*x[0]*x[0])
                                                           +(d7*x[1]*x[1])
                                                           +(d8*x[2]*x[2])
                                                           +(d9*x[3]*x[3])
                                                           +(d10*x[4]*x[4])
                                                           +(d11*x[5]*x[5])
                                                           +(d12*x[6]*x[6])
                                                           +(d13*x[7]*x[7])
                                                           +(d14*x[8]*x[8])
                                                           +(d15*x[9]*x[9])
                                                           +(d16*x[10]*x[10])
                                                           +(d17*x[11]*x[11])
                                                           +(d18*x[12]*x[12]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d6*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d7*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d8*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[9]*x[9])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[10]*x[10])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[11]*x[11])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[12]*x[12]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 7:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))))))
            d7 = comb(d,2)*comb(d-2,7)
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  x[9] - x[10]},
                {'type': 'ineq', 'fun': lambda x:  x[10] - x[11]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d7*x[0]*x[0])
                                                           +(d8*x[1]*x[1])
                                                           +(d9*x[2]*x[2])
                                                           +(d10*x[3]*x[3])
                                                           +(d11*x[4]*x[4])
                                                           +(d12*x[5]*x[5])
                                                           +(d13*x[6]*x[6])
                                                           +(d14*x[7]*x[7])
                                                           +(d15*x[8]*x[8])
                                                           +(d16*x[9]*x[9])
                                                           +(d17*x[10]*x[10])
                                                           +(d18*x[11]*x[11]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d7*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d8*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[9]*x[9])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[10]*x[10])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[11]*x[11]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 8:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))))))
            d8 = comb(d,2)*comb(d-2,8)
            d9 = comb(d,2)*comb(d-2,9)
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  x[9] - x[10]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d8*x[0]*x[0])
                                                           +(d9*x[1]*x[1])
                                                           +(d10*x[2]*x[2])
                                                           +(d11*x[3]*x[3])
                                                           +(d12*x[4]*x[4])
                                                           +(d13*x[5]*x[5])
                                                           +(d14*x[6]*x[6])
                                                           +(d15*x[7]*x[7])
                                                           +(d16*x[8]*x[8])
                                                           +(d17*x[9]*x[9])
                                                           +(d18*x[10]*x[10]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d8*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[9]*x[9])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[10]*x[10]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 9:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))))))
            d9 = comb(d,2)*comb(d-2,9)
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d9*x[0]*x[0])
                                                           +(d10*x[1]*x[1])
                                                           +(d11*x[2]*x[2])
                                                           +(d12*x[3]*x[3])
                                                           +(d13*x[4]*x[4])
                                                           +(d14*x[5]*x[5])
                                                           +(d15*x[6]*x[6])
                                                           +(d16*x[7]*x[7])
                                                           +(d17*x[8]*x[8])
                                                           +(d18*x[9]*x[9]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d9*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[9]*x[9]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 10:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))))))
            d10 = comb(d,2)*comb(d-2,10)
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d10*x[0]*x[0])
                                                           +(d11*x[1]*x[1])
                                                           +(d12*x[2]*x[2])
                                                           +(d13*x[3]*x[3])
                                                           +(d14*x[4]*x[4])
                                                           +(d15*x[5]*x[5])
                                                           +(d16*x[6]*x[6])
                                                           +(d17*x[7]*x[7])
                                                           +(d18*x[8]*x[8]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d10*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[8]*x[8]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 11:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))))))
            d11 = comb(d,2)*comb(d-2,11)
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d11*x[0]*x[0])
                                                           +(d12*x[1]*x[1])
                                                           +(d13*x[2]*x[2])
                                                           +(d14*x[3]*x[3])
                                                           +(d15*x[4]*x[4])
                                                           +(d16*x[5]*x[5])
                                                           +(d17*x[6]*x[6])
                                                           +(d18*x[7]*x[7]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d11*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[7]*x[7]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 12:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))))))
            d12 = comb(d,2)*comb(d-2,12)
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d12*x[0]*x[0])
                                                           +(d13*x[1]*x[1])
                                                           +(d14*x[2]*x[2])
                                                           +(d15*x[3]*x[3])
                                                           +(d16*x[4]*x[4])
                                                           +(d17*x[5]*x[5])
                                                           +(d18*x[6]*x[6]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d12*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[6]*x[6]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 13:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))))))
            d13 = comb(d,2)*comb(d-2,13)
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d13*x[0]*x[0])
                                                           +(d14*x[1]*x[1])
                                                           +(d15*x[2]*x[2])
                                                           +(d16*x[3]*x[3])
                                                           +(d17*x[4]*x[4])
                                                           +(d18*x[5]*x[5]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d13*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[5]*x[5]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 14:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))))))
            d14 = comb(d,2)*comb(d-2,14)
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d14*x[0]*x[0])
                                                           +(d15*x[1]*x[1])
                                                           +(d16*x[2]*x[2])
                                                           +(d17*x[3]*x[3])
                                                           +(d18*x[4]*x[4]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d14*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[4]*x[4]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b),
            (a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 15:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))))))
            d15 = comb(d,2)*comb(d-2,15)
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d15*x[0]*x[0])
                                                           +(d16*x[1]*x[1])
                                                           +(d17*x[2]*x[2])
                                                           +(d18*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d15*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[3]*x[3]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b),
            (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 16:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))))))
            d16 = comb(d,2)*comb(d-2,16)
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d16*x[0]*x[0])
                                                           +(d17*x[1]*x[1])
                                                           +(d18*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d16*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[2]*x[2]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b), (a,b))
            result = opt.minimize(fun, (s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 17:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))))))
            d17 = comb(d,2)*comb(d-2,17)
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  budget - (((d17*x[0]*x[0])
                                                           +(d18*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d17*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[1]*x[1]))))})
            a = 0
            s = 0
            b = None
            bnds = ((a,b), (a,b))
            result = opt.minimize(fun, (s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
        elif order == 18:
            d = 20
            fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))))))
            d18 = comb(d,2)*comb(d-2,18)
            cons = ({'type': 'ineq', 'fun': lambda x:  budget - (((d18*x[0]*x[0]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d18*x[0]*x[0]))))})
            a = 0
            s = 0
            b = None
            result = opt.minimize(fun, s, method = 'SLSQP',
                                  constraints=cons)

    return result.x
#delta, _ = quad(lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), 0, 6 / np.sqrt(n))
#delta = (1/np.sqrt(2*np.pi))*(2.9/np.sqrt(n))
delta = (maxi/np.sqrt(n))
if algo == 'curate':
    if dataset in ['cancer', 'earthquake']:
        d = 5
        fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))))))
                        
        
        d0 = (comb(d,2))
        d1 = comb(d,2)*comb(d-2,1)
        d2 = comb(d,2)*comb(d-2,2)
        d3 = comb(d,2)*comb(d-2,3)
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x: eps_total - (((d0*x[0]*x[0])
                                                           +(d1*x[1]*x[1])
                                                           +(d2*x[2]*x[2])
                                                           +(d3*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d1*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d2*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d3*x[3]*x[3]))))})
        a = 0
        s = 0
        b = None
        bnds = ((a,b), (a,b), (a,b), (a,b))
        results = opt.minimize(fun, (s,s,s,s), method = 'SLSQP',
                                        bounds = bnds, constraints=cons)

    elif dataset in ['survey']:
        d = 6
        fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))))))
        d0 = (comb(d,2))
        d1 = comb(d,2)*comb(d-2,1)
        d2 = comb(d,2)*comb(d-2,2)
        d3 = comb(d,2)*comb(d-2,3)
        d4 = comb(d,2)*comb(d-2,4)
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  eps_total - (((d0*x[0]*x[0])
                                                           +(d1*x[1]*x[1])
                                                           +(d2*x[2]*x[2])
                                                           +(d3*x[3]*x[3])
                                                           +(d4*x[4]*x[4]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[0]*x[0]))
                                                                   +(np.sqrt(2*np.log(1/delta_prime)*(d1*x[1]*x[1])))
                                                                   +(np.sqrt(2*np.log(1/delta_prime)*(d2*x[2]*x[2])))
                                                                   +(np.sqrt(2*np.log(1/delta_prime)*(d3*x[3]*x[3])))
                                                                   +(np.sqrt(2*np.log(1/delta_prime)*(d4*x[4]*x[4])))))})
        a = 0
        s = 0
        b = None
        bnds = ((a,b), (a,b), (a,b), (a,b), (a,b))
        results = opt.minimize(fun, (s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
    elif dataset in ['asia']:
        d = 8
        fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))))))
                        
        
        d0 = (comb(d,2))
        d1 = comb(d,2)*comb(d-2,1)
        d2 = comb(d,2)*comb(d-2,2)
        d3 = comb(d,2)*comb(d-2,3)
        d4 = comb(d,2)*comb(d-2,4)
        d5 = comb(d,2)*comb(d-2,5)
        d6 = comb(d,2)*comb(d-2,6)
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  eps_total - (((d0*x[0]*x[0])
                                                           +(d1*x[1]*x[1])
                                                           +(d2*x[2]*x[2])
                                                           +(d3*x[3]*x[3])
                                                           +(d4*x[4]*x[4])
                                                           +(d5*x[5]*x[5])
                                                           +(d6*x[6]*x[6]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d1*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d2*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d3*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d4*x[4]*x[4]))
                                                           +((np.sqrt(2*np.log(1/delta_prime)*d5*x[5]*x[5])))
                                                           +((np.sqrt(2*np.log(1/delta_prime)*d6*x[6]*x[6]))))})
        a = 0
        s = 0
        b = None
        bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
        results = opt.minimize(fun, (s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
    elif dataset in ['sachs']:
        d = 11
        fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))))))
                        
        d0 = comb(d,2)
        d1 = comb(d,2)*comb(d-2,1)
        d2 = comb(d,2)*comb(d-2,2)
        d3 = comb(d,2)*comb(d-2,3)
        d4 = comb(d,2)*comb(d-2,4)
        d5 = comb(d,2)*comb(d-2,5)
        d6 = comb(d,2)*comb(d-2,6)
        d7 = comb(d,2)*comb(d-2,7)
        d8 = comb(d,2)*comb(d-2,8)
        d9 = comb(d,2)*comb(d-2,9)
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  eps_total - (((d0*x[0]*x[0])
                                                           +(d1*x[1]*x[1])
                                                           +(d2*x[2]*x[2])
                                                           +(d3*x[3]*x[3])
                                                           +(d4*x[4]*x[4])
                                                           +(d5*x[5]*x[5])
                                                           +(d6*x[6]*x[6])
                                                           +(d7*x[7]*x[7])
                                                           +(d8*x[8]*x[8])
                                                           +(d9*x[9]*x[9]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d1*x[1]*x[1]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d2*x[2]*x[2]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d3*x[3]*x[3]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d4*x[4]*x[4]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d5*x[5]*x[5]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d6*x[6]*x[6]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d7*x[7]*x[7]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d8*x[8]*x[8]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*d9*x[9]*x[9])))})
        a = 0
        s = 0
        b = None
        bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
        results = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)
    elif dataset in ['child']:
        d = 20
        fun = lambda x:(((0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[15])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[16])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[17])-1))))/delta)))
                        *(0.5+0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[18])-1))))/delta))))+
                        (1-(((0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[0])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[1])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[2])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[3])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[4])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[5])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[6])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[7])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[8])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[9])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[10])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[11])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[12])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[13])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[14])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[15])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[16])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[17])-1))))/delta)))
                        *(0.5-0.5*np.exp(((-T)*beta*(np.log(1+(q*(np.exp(x[18])-1))))/delta)))))))
        d0 = (comb(d,2))
        d1 = comb(d,2)*comb(d-2,1)
        d2 = comb(d,2)*comb(d-2,2)
        d3 = comb(d,2)*comb(d-2,3)
        d4 = comb(d,2)*comb(d-2,4)
        d5 = comb(d,2)*comb(d-2,5)
        d6 = comb(d,2)*comb(d-2,6)
        d7 = comb(d,2)*comb(d-2,7)
        d8 = comb(d,2)*comb(d-2,8)
        d9 = comb(d,2)*comb(d-2,9)
        d10 = comb(d,2)*comb(d-2,10)
        d11 = comb(d,2)*comb(d-2,11)
        d12 = comb(d,2)*comb(d-2,12)
        d13 = comb(d,2)*comb(d-2,13)
        d14 = comb(d,2)*comb(d-2,14)
        d15 = comb(d,2)*comb(d-2,15)
        d16 = comb(d,2)*comb(d-2,16)
        d17 = comb(d,2)*comb(d-2,17)
        d18 = comb(d,2)*comb(d-2,18)
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x:  x[2] - x[3]},
                {'type': 'ineq', 'fun': lambda x:  x[3] - x[4]},
                {'type': 'ineq', 'fun': lambda x:  x[4] - x[5]},
                {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},
                {'type': 'ineq', 'fun': lambda x:  x[6] - x[7]},
                {'type': 'ineq', 'fun': lambda x:  x[7] - x[8]},
                {'type': 'ineq', 'fun': lambda x:  x[8] - x[9]},
                {'type': 'ineq', 'fun': lambda x:  x[9] - x[10]},
                {'type': 'ineq', 'fun': lambda x:  x[10] - x[11]},
                {'type': 'ineq', 'fun': lambda x:  x[11] - x[12]},
                {'type': 'ineq', 'fun': lambda x:  x[12] - x[13]},
                {'type': 'ineq', 'fun': lambda x:  x[13] - x[14]},
                {'type': 'ineq', 'fun': lambda x:  x[14] - x[15]},
                {'type': 'ineq', 'fun': lambda x:  x[15] - x[16]},
                {'type': 'ineq', 'fun': lambda x:  x[16] - x[17]},
                {'type': 'ineq', 'fun': lambda x:  x[17] - x[18]},
                {'type': 'ineq', 'fun': lambda x:  eps_total - (((d0*x[0]*x[0])
                                                           +(d1*x[1]*x[1])
                                                           +(d2*x[2]*x[2])
                                                           +(d3*x[3]*x[3])
                                                           +(d4*x[4]*x[4])
                                                           +(d5*x[5]*x[5])
                                                           +(d6*x[6]*x[6])
                                                           +(d7*x[7]*x[7])
                                                           +(d8*x[8]*x[8])
                                                           +(d9*x[9]*x[9])
                                                           +(d10*x[10]*x[10])
                                                           +(d11*x[11]*x[11])
                                                           +(d12*x[12]*x[12])
                                                           +(d13*x[13]*x[13])
                                                           +(d14*x[14]*x[14])
                                                           +(d15*x[15]*x[15])
                                                           +(d16*x[16]*x[16])
                                                           +(d17*x[17]*x[17])
                                                           +(d18*x[18]*x[18]))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[0]*x[0])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[1]*x[1])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[2]*x[2])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[3]*x[3])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[4]*x[4])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[5]*x[5])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[6]*x[6])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[7]*x[7])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[8]*x[8])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[9]*x[9])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[10]*x[10])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[11]*x[11])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[12]*x[12])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[13]*x[13])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[14]*x[14])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[15]*x[15])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[16]*x[16])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[17]*x[17])))
                                                           +(np.sqrt(2*np.log(1/delta_prime)*(d0*x[18]*x[18]))))})
        a = 0
        s = 0
        b = None
        bnds = ((a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b),
            (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b), (a,b))
        results = opt.minimize(fun, (s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s), method = 'SLSQP',
                                         bounds = bnds, constraints=cons)


        
    


def bn_data(name, feature=None, size=10000):
    data = pd.read_csv(name+".csv")
    data = data.drop("Unnamed: 0", axis=1)
    data = data.astype('category')
    data = data.apply(lambda x: x.cat.codes)
    data = np.array(data)
    data = data.astype(int)

    node_size = data.shape[1]
    g_answer = nx.DiGraph()
    g_answer.add_nodes_from(np.arange(node_size))
    g_answer.add_edges_from(ground_truth[name])

    if feature is None:
        feature = data.shape[1]

    return data[:size, :feature], g_answer
KendalltauResult = namedtuple('KendalltauResult', ('correlation', 'pvalue'))

def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # If the check cannot be properly performed we fallback to omitting
        # nan values and raising a warning. This can happen when attempting to
        # sum things that are not numbers (e.g. as in the function `mode`).
        contains_nan = False
        nan_policy = 'omit'
        warnings.warn("The input array could not be properly checked for nan "
                      "values. nan values will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return (contains_nan, nan_policy)

def kendalltaua(x, y, initial_lexsort=None, nan_policy='propagate'):
    """
    Calculate Kendall's tau-a, a correlation measure for ordinal data.

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, values close to -1 indicate
    strong disagreement.  This is the 1945 "tau-b" version of Kendall's
    tau [2]_, which can account for ties and which reduces to the 1938 "tau-a"
    version [1]_ in absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not D, they will
        be flattened to D.
    initial_lexsort : bool, optional
        Unused (deprecated).
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'. Note that if the input contains nan
        'omit' delegates to mstats_basic.kendalltau(), which has a different
        implementation.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The two-sided p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    See also
    --------
    spearmanr : Calculates a Spearman rank-order correlation coefficient.
    theilslopes : Computes the Theil-Sen estimator for a set of points (x, y).
    weightedtau : Computes a weighted version of Kendall's tau.

    Notes
    -----
    The definition of Kendall's tau that is used is [2]_::

      tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))

    where P is the number of concordant pairs, Q the number of discordant
    pairs, T the number of ties only in `x`, and U the number of ties only in
    `y`.  If a tie occurs for the same pair in both `x` and `y`, it is not
    added to either T or U.

    References
    ----------
    .. [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 893, 1938.
    .. [2] Maurice G. Kendall, "The treatment of ties in ranking problems",
           Biometrika Vol. 33, No. 3, pp. 239-251. 1945.
    .. [3] Gottfried E. Noether, "Elements of Nonparametric Statistics", John
           Wiley & Sons, 1967.
    .. [4] Peter M. Fenwick, "A new data structure for cumulative frequency
           tables", Software: Practice and Experience, Vol. 24, No. 3,
           pp. 327-336, 1994.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> x2 = [1, 4, 7, 1, 0]
    >>> tau, p_value = stats.kendalltau(x1, x2)
    >>> tau
    -0.47140452079103173
    >>> p_value
    0.2827454599327748

    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs to `kendalltau` must be of the same size, "
                         "found x-size %s and y-size %s" % (x.size, y.size))
    elif not x.size or not y.size:
        print('here1 is wrong!')
        return KendalltauResult(np.nan, np.nan)  # Return NaN if arrays are empty

    # check both x and y
    cnx, npx = _contains_nan(x, nan_policy)
    cny, npy = _contains_nan(y, nan_policy)
    contains_nan = cnx or cny
    if npx == 'omit' or npy == 'omit':
        nan_policy = 'omit'

    if contains_nan and nan_policy == 'propagate':
        return KendalltauResult(np.nan, np.nan)

    elif contains_nan and nan_policy == 'omit':
        x = ma.masked_invalid(x)
        y = ma.masked_invalid(y)
        return mstats_basic.kendalltau(x, y)

    if initial_lexsort is not None:  # deprecate to drop!
        warnings.warn('"initial_lexsort" is gone!')

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.) * (2*cnt + 5)).sum(), cnt)

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)
    
    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.where(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1, cntx = count_rank_tie(x)     # ties in x, stats
    ytie, y0, y1, cnty = count_rank_tie(y)     # ties in y, stats

    tot = (size * (size - 1)) // 2

    if xtie == tot or ytie == tot:
        return (KendalltauResult(np.nan, np.nan), cntx, cnty)

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    # tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    tau = con_minus_dis / tot
    # Limit range to fix computational errors
    tau = min(1., max(-1., tau))

    # con_minus_dis is approx normally distributed with this variance [3]_
    var = (size * (size - 1) * (2.*size + 5) - x1 - y1) / 18. + (
        xtie * ytie) / (2. * size * (size - 1)) + x0 * y0 / (9. *
        size * (size - 1) * (size - 2))
    pvalue = special.erfc(np.abs(con_minus_dis) / np.sqrt(var) / np.sqrt(2))

    # Limit range to fix computational errors
    return KendalltauResult(min(1., max(-1., tau)), pvalue), cntx, cnty

def cal_precision(e1, e2):
    cnt = 0.0
    for e in e1:
        if e in e2:
            cnt = cnt + 1
    if cnt == 0:
        if len(e1) == 0:
            return 1
        return 0
    return cnt / len(e1)

def cal_recall(e1, e2):
    cnt = 0.0
    for e in e2:
        if e in e1:
            cnt = cnt + 1
    if cnt == 0:
        return 0
    return cnt / len(e2)

def cal_f1(e1, e2):
    precision = cal_precision(e1, e2)
    recall = cal_recall(e1, e2)
    numerator = 2*precision*recall
    if numerator == 0:
        return 0
    return numerator/(precision+recall)
# -*- coding: utf-8 -*-


def g_square_bin(dm, x, y, s):
    """G square test for a binary data.

    Args:
        dm: the data matrix to be used (as a numpy.ndarray).
        x: the first node (as an integer).
        y: the second node (as an integer).
        s: the set of neibouring nodes of x and y (as a set()).

    Returns:
        p_val: the p-value of conditional independence.
    """

    def _calculate_tlog(x, y, s, dof, dm):
        nijk = np.zeros((2, 2, dof))
        s_size = len(s)
        z = []
        for z_index in range(s_size):
            z.append(s.pop())
            pass
        for row_index in range(0, dm.shape[0]):
            i = dm[row_index, x]
            j = dm[row_index, y]
            k = []
            k_index = 0
            for z_index in range(s_size):
                k_index += dm[row_index, z[z_index]] * int(pow(2, z_index))
                pass
            nijk[i, j, k_index] += 1
            pass
        nik = np.ndarray((2, dof))
        njk = np.ndarray((2, dof))
        for k_index in range(dof):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((2, 2 , dof))
        tlog.fill(np.nan)
        for k in range(dof):
            tx = np.array([nik[:,k]]).T
            ty = np.array([njk[:,k]])
            tdijk = tx.dot(ty)
            tlog[:,:,k] = nijk[:,:,k] * nk[k] / tdijk
            pass
        return (nijk, tlog)

    row_size = dm.shape[0]
    s_size = len(s)
    dof = int(pow(2, s_size))
    row_size_required = 10 * dof
    if row_size < row_size_required:
        return 1
    nijk = None
    if s_size < 6:
        if s_size == 0:
            nijk = np.zeros((2, 2))
            for row_index in range(0, dm.shape[0]):
                i = dm[row_index, x]
                j = dm[row_index, y]
                nijk[i, j] += 1
                pass
            tx = np.array([nijk.sum(axis = 1)]).T
            ty = np.array([nijk.sum(axis = 0)])
            tdij = tx.dot(ty)
            tlog = nijk * row_size / tdij
            pass
        if s_size > 0:
            nijk, tlog = _calculate_tlog(x, y, s, dof, dm)
            pass
        pass
    else:
        nijk = np.zeros((2, 2, 1))
        i = dm[0, x]
        j = dm[0, y]
        k = []
        for z in s:
            k.append(dm[:,z])
            pass
        k = np.array(k).T
        parents_count = 1
        parents_val = np.array([k[0,:]])
        nijk[i, j, parents_count - 1] = 1
        for it_sample in range(1, row_size):
            is_new = True
            i = dm[it_sample, x]
            j = dm[it_sample, y]
            tcomp = parents_val[:parents_count,:] == k[it_sample,:]
            for it_parents in range(parents_count):
                if np.all(tcomp[it_parents,:]):
                    nijk[i, j, it_parents] += 1
                    is_new = False
                    break
                pass
            if is_new is True:
                parents_count += 1
                parents_val = np.r_[parents_val, [k[it_sample,:]]]
                nnijk = np.zeros((2,2,parents_count))
                for p in range(parents_count - 1):
                    nnijk[:,:,p] = nijk[:,:,p]
                nnijk[i, j, parents_count - 1] = 1
                nijk = nnijk
                pass
            pass
        nik = np.ndarray((2, parents_count))
        njk = np.ndarray((2, parents_count))
        for k_index in range(parents_count):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((2, 2 , parents_count))
        tlog.fill(np.nan)
        for k in range(parents_count):
            tX = np.array([nik[:,k]]).T
            tY = np.array([njk[:,k]])
            tdijk = tX.dot(tY)
            tlog[:,:,k] = nijk[:,:,k] * nk[k] / tdijk
            pass
        pass
    log_tlog = np.log(tlog)
    G2 = np.nansum(2 * nijk * log_tlog)
    p_val = chi2.sf(G2, dof)
    if s_size == 0:
        nijk = nijk.reshape((nijk.shape[0], nijk.shape[1], 1))
        log_tlog = log_tlog.reshape((log_tlog.shape[0], log_tlog.shape[1], 1))
    
    return G2, p_val, nijk, log_tlog

def g_square_dis(dm, x, y, s):
    """G square test for discrete data.

    Args:
        dm: the data matrix to be used (as a numpy.ndarray).
        x: the first node (as an integer).
        y: the second node (as an integer).
        s: the set of neibouring nodes of x and y (as a set()).
        levels: levels of each column in the data matrix
            (as a list()).

    Returns:
        p_val: the p-value of conditional independence.
    """
    levels = np.amax(dm, axis=0) + 1
    def _calculate_tlog(x, y, s, dof, levels, dm):
        prod_levels = np.prod(list(map(lambda x: levels[x], s)))
        nijk = np.zeros((levels[x], levels[y], prod_levels))
        s_size = len(s)
        z = []
        for z_index in range(s_size):
            z.append(s.pop())
            pass
        for row_index in range(dm.shape[0]):
            i = dm[row_index, x]
            j = dm[row_index, y]
            k = []
            k_index = 0
            for s_index in range(s_size):
                if s_index == 0:
                    k_index += dm[row_index, z[s_index]]
                else:
                    lprod = np.prod(list(map(lambda x: levels[x], z[:s_index])))
                    k_index += (dm[row_index, z[s_index]] * lprod)
                    pass
                pass
            nijk[i, j, k_index] += 1
            pass
        nik = np.ndarray((levels[x], prod_levels))
        njk = np.ndarray((levels[y], prod_levels))
        for k_index in range(prod_levels):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((levels[x], levels[y], prod_levels))
        tlog.fill(np.nan)
        for k in range(prod_levels):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        return (nijk, tlog)

    row_size = dm.shape[0]
    s_size = len(s)
    dof = ((levels[x] - 1) * (levels[y] - 1)
           * np.prod(list(map(lambda x: levels[x], s))))
    row_size_required = 10 * dof
    nijk = None
    if s_size < 5:
        if s_size == 0:
            nijk = np.zeros((levels[x], levels[y]))
            for row_index in range(row_size):
                i = dm[row_index, x]
                j = dm[row_index, y]
                nijk[i, j] += 1
                pass
            tx = np.array([nijk.sum(axis = 1)]).T
            ty = np.array([nijk.sum(axis = 0)])
            tdij = tx.dot(ty)
            tlog = nijk * row_size / tdij
            pass
        if s_size > 0:
            nijk, tlog = _calculate_tlog(x, y, s, dof, levels, dm)
            pass
        pass
    else:
        nijk = np.zeros((levels[x], levels[y], 1))
        i = dm[0, x]
        j = dm[0, y]
        k = []
        for z in s:
            k.append(dm[:, z])
            pass
        k = np.array(k).T
        parents_count = 1
        parents_val = np.array([k[0, :]])
        nijk[i, j, parents_count - 1] = 1
        for it_sample in range(1, row_size):
            is_new = True
            i = dm[it_sample, x]
            j = dm[it_sample, y]
            tcomp = parents_val[:parents_count, :] == k[it_sample, :]
            for it_parents in range(parents_count):
                if np.all(tcomp[it_parents, :]):
                    nijk[i, j, it_parents] += 1
                    is_new = False
                    break
                pass
            if is_new is True:
                parents_count += 1
                parents_val = np.r_[parents_val, [k[it_sample, :]]]
                nnijk = np.zeros((levels[x], levels[y], parents_count))
                for p in range(parents_count - 1):
                    nnijk[:, :, p] = nijk[:, :, p]
                    pass
                nnijk[i, j, parents_count - 1] = 1
                nijk = nnijk
                pass
            pass
        nik = np.ndarray((levels[x], parents_count))
        njk = np.ndarray((levels[y], parents_count))
        for k_index in range(parents_count):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((levels[x], levels[y], parents_count))
        tlog.fill(np.nan)
        for k in range(parents_count):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        pass
    log_tlog = np.log(tlog)
    G2 = np.nansum(2 * nijk * log_tlog)
    if dof == 0:
        p_val = 1
    else:
        p_val = chi2.sf(G2, dof)

    if s_size == 0:
        nijk = nijk.reshape((nijk.shape[0], nijk.shape[1], 1))
        log_tlog = log_tlog.reshape((log_tlog.shape[0], log_tlog.shape[1], 1))
    return G2, p_val, nijk, log_tlog


def findJs(x, y, S, alpha, dm, task=taskval, verbose = False):
    if task == 'bin':
        indep_test_func = g_square_bin
        dof = int(pow(2, len(S)))
    else:
        indep_test_func = g_square_dis
        levels = np.amax(dm, axis=0) + 1
        dof = ((levels[x] - 1) * (levels[y] - 1)
            * np.prod(list(map(lambda x: levels[x], S))))

    G2, pval, nijk, log_tlog = indep_test_func(dm, x, y, S)
    n = dm.shape[0]
    if pval == alpha:
        return 0
    if pval > alpha:
        positive = 1
    else:
        positive = -1
    nik = np.sum(nijk, axis=1)
    njk = np.sum(nijk, axis=0)
    nk = np.sum(njk, axis=0)
    dims = nijk.shape

    threshold = chi2.isf(alpha, dof)
    temp_nij = np.zeros(dims[0:2])
    temp_log = np.zeros(dims[0:2])
    direction = np.zeros(dims)

    step_nijk = nijk.copy()
    step_nik = nik.copy()
    step_njk = njk.copy()
    step_nk = nk.copy()
    step_G2 = 2 * nijk * log_tlog # 2x2xk
    step_G2 = np.nansum(np.nansum(step_G2, axis=0),axis=0)
    steps = 0
    change = False
    
    while(change == False):
        for k in range(dims[2]):
            temp_n = step_nk[k] - 1
            for i in range(dims[0]):
                temp_ni = step_nik[:, k].copy()
                temp_ni[i] = temp_ni[i] - 1
                for j in range(dims[1]):
                    temp_nj = step_njk[:, k].copy()
                    temp_nj[j] = temp_nj[j] - 1
                    temp_nij = step_nijk[:, :, k].copy()

                    if (temp_nij[i, j] == 0):
                        direction[i, j, k] = -np.Inf
                    else:
                        temp_nij[i, j] = temp_nij[i, j] - 1
                        temp_ni = temp_ni.reshape((-1, 1))
                        temp_nj = temp_nj.reshape((-1, 1)).T
                        temp_log = temp_n * (temp_nij / np.dot(temp_ni, temp_nj))
                        temp_G2 = 2 * temp_nij * np.log(temp_log)
                        direction[i,j,k] = positive * (np.nansum(temp_G2) - step_G2[k])

        choose_direction = np.unravel_index(direction.argmax(), direction.shape)
        if (positive * (np.sum(step_G2) + positive * direction[choose_direction]) >= positive * threshold):
            JSdist = positive * steps
            change = True
        elif (steps == n-1):
            JSdist = positive * steps
            change = True
        else:
            steps += 1
            step_nijk[choose_direction] = step_nijk[choose_direction] - 1
            step_G2[choose_direction[-1]] = step_G2[choose_direction[-1]] + positive * direction[choose_direction]
            step_nik = np.sum(step_nijk, axis=1)
            step_njk = np.sum(step_nijk, axis=0)
            step_nk = np.sum(step_njk, axis=0)
    return JSdist

def NumSig(JSscore, pval, alpha):

    Z = zip(JSscore, pval)
    Z = sorted(Z, reverse=True)

    JSscore_new , pval_new = zip(*Z)
    qscore = np.zeros(len(JSscore) + 1)
    if pval_new[0] >= alpha:
        qscore[0] = -JSscore_new[0] - 1
    else:
        qscore[0] = -JSscore_new[0]

    if pval_new[-1] < alpha:
        qscore[-1] = JSscore_new[-1] - 1
    else:
        qscore[-1] = JSscore_new[-1]

    for i in range(1, len(JSscore_new)):
        if pval_new[i-1] > alpha and pval_new[i] <= alpha:
            qscore[i] = min(JSscore_new[i-1], -JSscore_new[i]) - 1
        elif pval_new[i-1] > alpha and pval_new[i] > alpha:
            qscore[i] = -JSscore_new[i]
        else:
            qscore[i] = JSscore_new[i-1]

    return qscore


#def wrapped_ci_test_bin(dm, i, j, k, **kwargs):
#    p_val = ci_test_bin(dm, i, j, k, **kwargs)
#    return None, p_val

#def wrapped_ci_test_dis(dm, i, j, k, **kwargs):
#    p_val = ci_test_dis(dm, i, j, k, **kwargs)
#    return None, p_val

def bincondKendall(data_matrix, x, y, k, **kwargs):
    s_size = len(k)
    row_size = data_matrix.shape[0]
    if s_size == 0:
        (tau, pval), _, _ = kendalltaua(data_matrix[:,x], data_matrix[:,y])
        tau = tau * np.sqrt(9.0 * row_size * (row_size - 1) / (4*row_size+10))
        pval = norm.sf(np.abs(tau))
        return tau, pval
    z = []
    for z_index in range(s_size):
        z.append(k.pop())
        pass

    dm_unique = np.unique(data_matrix[:, z], axis=0)
    sumwk = 0
    sumweight = 0
    tau = 0
    pval = 0
    for split_k in dm_unique:
        index = np.ones((row_size),dtype=bool)
        for i in range(s_size):
            index = ((data_matrix[..., z[i]] == split_k[i]) & index)

        new_dm = data_matrix[index, :]
        nk = new_dm.shape[0]
        if nk <= 2:
            continue
        (condtau, condpval), cntx, cnty = kendalltaua(new_dm[:, x], new_dm[:, y])
        if np.isnan(condpval):
            continue
        sigma0_sq = (4.0 * nk + 10) / (9.0 * nk * (nk-1.0))
        tau += condtau / sigma0_sq
        sumwk += 1.0 / sigma0_sq

    tau /= np.sqrt(sumwk)
    pval = norm.sf(np.abs(tau))

    return tau, pval

def discondKendall(data_matrix, x, y, k, **kwargs):
    s_size = len(k)
    row_size = data_matrix.shape[0]
    if s_size == 0:
        (tau, pval), _, _ = kendalltaua(data_matrix[:,x], data_matrix[:,y])
        tau = tau * np.sqrt(9.0 * row_size * (row_size - 1) / (4*row_size+10))
        pval = norm.sf(np.abs(tau))
        return tau, pval
    z = []
    for z_index in range(s_size):
        z.append(k.pop())
        pass

    dm_unique = np.unique(data_matrix[:, z], axis=0)
    sumwk = 0
    sumweight = 0
    tau = 0
    pval = 0
    for split_k in dm_unique:
        index = np.ones((row_size),dtype=bool)
        for i in range(s_size):
            index = ((data_matrix[..., z[i]] == split_k[i]) & index)

        new_dm = data_matrix[index, :]
        nk = new_dm.shape[0]
        if nk <= 2:
            continue
        (condtau, condpval), cntx, cnty = kendalltaua(new_dm[:, x], new_dm[:, y])
        if np.isnan(condpval):
            continue
        sigma0_sq = (4.0 * nk + 10) / (9.0 * nk * (nk-1.0))
        tau += condtau / sigma0_sq
        sumwk += 1.0 / sigma0_sq

    tau /= np.sqrt(sumwk)
    pval = norm.sf(np.abs(tau))


    return tau, pval

def _create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.
    Args:
        node_ids: a list of node ids
    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

def estimate_skeleton(indep_test_func, data_matrix, alpha, **kwargs):
    """Estimate a skeleton graph from the statistis information.
    Args:
        indep_test_func: the function name for a conditional
            independency test.
        data_matrix: data (as a numpy array).
        alpha: the significance level.
        kwargs:
            'max_reach': maximum value of l (see the code).  The
                value depends on the underlying distribution.
            'method': if 'stable' given, use stable-PC algorithm
                (see [Colombo2014]).
            'init_graph': initial structure of skeleton graph
                (as a networkx.Graph). If not specified,
                a complete graph is used.
            other parameters may be passed depending on the
                indep_test_func()s.
    Returns:
        g: a skeleton graph (as a networkx.Graph).
        sep_set: a separation set (as an 2D-array of set()).
    [Colombo2014] Diego Colombo and Marloes H Maathuis. Order-independent
    constraint-based causal structure learning. In The Journal of Machine
    Learning Research, Vol. 15, pp. 3743782, 2014.
    """

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    l = 0
    count = 0
    while True:
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i,j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
            if len(adj_i) < l:
                continue
            for k in combinations(adj_i, l):
                _logger.debug('indep prob of %s and %s with subset %s'
                                % (i, j, str(k)))
                tau, p_val = indep_test_func(data_matrix, i, j, set(k), **kwargs)
                _logger.debug('tau is %s' % str(tau))
                if p_val > alpha:
                    count = count+1
                    if g.has_edge(i, j):
                        _logger.debug('p: remove edge (%s, %s)' % (i, j))
                        if method_stable(kwargs):
                            remove_edges.append((i, j))
                        else:
                            g.remove_edge(i, j)
                    sep_set[i][j] |= set(k)
                    sep_set[j][i] |= set(k)
                    break
            cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    return (g, sep_set, count)

def estimate_skeleton_EM(indep_test_func, data_matrix, alpha, eps=epsilonpriv, delta=delta_prime, task=taskval, **kwargs):

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    test_count = 0

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]

    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    l = 0
    count = 0
    while True:
        cont = False
        remove_edges = []
        for i in node_ids:
            adj_i = list(g.neighbors(i))
            if len(adj_i) < l + 1:
                continue
            count = count + 1
            ind_set, k_set, temp_count = findPrivInd(i, adj_i, l, data_matrix, eps/2, eps/2, indep_test_func, alpha, task, **kwargs)
            test_count += temp_count
            for j in range(len(ind_set)):
                if g.has_edge(i, ind_set[j]):
                    _logger.debug('p: remove edge (%s, %s)' % (i, j))
                    if method_stable(kwargs):
                        remove_edges.append((i, ind_set[j]))
                    else:
                        g.remove_edge(i, ind_set[j])
                sep_set[i][j] |= set(k_set[j])
                sep_set[j][i] |= set(k_set[j])
            cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    # advanced composition
    eps_prime = np.sqrt(2*count*np.log(1/delta))*eps + count*eps*eps
    delta_em = (count*delta_prime)+delta_ad


    return (g, sep_set, eps_prime, delta_em, test_count)


def findPrivInd(i, adj_i, l, data_matrix, epsilon1, epsilon2, indep_test_func, alpha, task, **kwargs):
    
    test_count = 0
    
    q1 = dict()
    q2 = dict()
    pval = dict()
    K_set = dict()
    for j in adj_i:
        R1 = adj_i.copy()
        R1.remove(j)
        max_pval = -1
        for k in combinations(R1, l):
            _, temp_pval = indep_test_func(data_matrix, i, j, set(k), **kwargs)
            test_count += 1
            if temp_pval > max_pval:
                max_pval = temp_pval
                max_k = k
        q1[j] = findJs(i, j, set(max_k), alpha, data_matrix, task)
        pval[j] = max_pval
        K_set[j] = max_k
    R2 = range(len(adj_i))

    q2 = NumSig(q1.values(), pval.values(), alpha)

    beta = EM(q2, epsilon2, 1, data_matrix, R2)
    Ind_set = []
    re_K = []
    for t in range(beta):
        Vr = EM(q1, epsilon1 / beta, 1, data_matrix, adj_i)
        Ind_set.append(Vr)
        re_K.append(K_set[Vr])
        q1[Vr] = -10000

    return Ind_set, re_K, test_count


def EM(q, epsilon, S, D, R):

    prob = dict()
    for i in range(len(R)):
        r = R[i]
        prob[r] = epsilon * q[r] / 2.0 / S
    prob = np.array(list(prob.values()))
    if np.any(np.isinf(prob)):
        return R[np.where(prob>0)[0][0]]
    prob = prob - np.max(prob) + 10
    prob = np.exp(prob)
    prob = prob / prob.sum()
    index = np.random.choice(R, p=prob.ravel())

    return index

#change eps to epsilonpriv

def estimate_skeleton_SVT(indep_test_func, data_matrix, alpha, eps=epsilonpriv, delta=delta_prime, **kwargs):

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    test_count = 0

    node_ids = range(data_matrix.shape[1])
    n = data_matrix.shape[0]
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    l = 0
    count = 0
    if delta is None:
        delta = 1e-4
    S, _ = quad(lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), 0, 6 / np.sqrt(n))
    sigma1 = 2 * S / eps
    sigma2 = 4 * sigma1

    T0 = alpha + np.random.laplace(0, sigma1)
    while True:
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i,j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
            if len(adj_i) < l:
                continue

            for k in combinations(adj_i, l):
                _logger.debug('indep prob of %s and %s with subset %s'
                                % (i, j, str(k)))
                v = np.random.laplace(0, sigma2)
                p_val = indep_test_func(data_matrix, i, j, set(k), **kwargs)[1] + v
                test_count += 1
                _logger.debug('p_val is %s' % str(p_val))

                if p_val < T0:
                    continue
                if p_val >= T0:
                    count += 1
                    if g.has_edge(i, j):
                        _logger.debug('p: remove edge (%s, %s)' % (i, j))
                        if method_stable(kwargs):
                            remove_edges.append((i, j))
                        else:
                            g.remove_edge(i, j)
                    sep_set[i][j] |= set(k)
                    sep_set[j][i] |= set(k)
                    T0 = alpha + np.random.normal(0, sigma1)
                    break
            cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    # advanced composition
    eps_em = (np.sqrt(2*count*np.log(1/delta))*eps) + (count*eps*eps)
    delta_em = (count*delta_prime)+delta_ad
 
    return (g, sep_set, eps_em, delta_em, test_count)


def estimate_skeleton_curate(epstotal, delta_prime, delta_ad, delta_total,indep_test_func, data_matrix, **kwargs):

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    test_count = 0
    test = 0
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    fixed_edges = set()
    if 'fixed_edges' in kwargs:
        _fixed_edges = kwargs['fixed_edges']
        if not isinstance(_fixed_edges, nx.Graph):
            raise ValueError
        if not _fixed_edges.number_of_nodes() == len(node_ids):
            raise ValueError('fixed_edges not matching data_matrix shape')
        for (i, j) in _fixed_edges.edges:
            fixed_edges.add((i, j))
            fixed_edges.add((j, i))

    l = 0
    track = []
    eps_track = []
    count = 0
    eps_rem = epstotal
    m = data_matrix.shape[0]
    row_rand = np.arange(m)
    np.random.shuffle(row_rand)
    delta_curate = 0
    dm_subsampled = data_matrix[row_rand[0:int(m*q)]]
    delta = (maxi/np.sqrt(n))
    deledge = 0
    initial = comb(d,2)
    epsiloncurate = []
    while True:
        if l == 0:
            epsilon = results.x[0]
        else:
            value = onlinebudgeting(eps_rem,edges,l)
            epsilon = value[0]
            
        #p = l
        eps = np.log(1+(q*(np.exp(epsilon)-1)))
        sigma = delta/eps
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            if (i, j) in fixed_edges:
                continue

            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i,j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
                if len(adj_i) < l:
                    continue
                for k in combinations(adj_i, l):
                    _logger.debug('indep prob of %s and %s with subset %s'
                                  % (i, j, str(k)))
                    v = np.random.laplace(0, sigma)
                    p_val = indep_test_func(dm_subsampled, i, j, set(k), **kwargs)[1] + v
                    _logger.debug('p_val is %s' % str(p_val))
                    if p_val < (T+(T*beta)) and p_val > (T-(T*beta)):
                        count = count+1
                        seed(10000)
                        # generate some integers
                        rand = randint(0,1)
                        if(rand==0):
                            if g.has_edge(i, j):
                                _logger.debug('p: remove edge (%s, %s)' % (i, j))
                                if method_stable(kwargs):
                                    remove_edges.append((i, j))
                                    deledge+=1
                                else:
                                    g.remove_edge(i, j)
                            sep_set[i][j] |= set(k)
                            sep_set[j][i] |= set(k)
                            break
                            
                        
                    if p_val > T+(T*beta):
                        count = count+1
                        if g.has_edge(i, j):
                            _logger.debug('p: remove edge (%s, %s)' % (i, j))
                            if method_stable(kwargs):
                                remove_edges.append((i, j))
                                deledge+=1
                            else:
                                g.remove_edge(i, j)
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
                cont = True
        track.append(count)
        eps_track.append(eps)
        epsiloncurate.append(((count*eps*eps+
                            np.sqrt(2*count*np.log(1/delta_prime)*eps*eps))))
        eps_rem = eps_rem - (count*eps*eps+
                            np.sqrt(2*count*np.log(1/delta_prime)*eps*eps))
        eps_total = eps_rem
        edges = initial - deledge 
        initial = initial - deledge 
        delta_curate = delta_curate + delta_prime + (count*delta_ad)
        if delta_curate > delta_total:
            break
        if np.sum(epsiloncurate)>=eps_limit:
            break
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    #return (g, sep_set,track, test, eps_track,p,delta_curate)
    return (g, sep_set,track, eps_track,p,epsiloncurate,delta_curate)

def estimate_skeleton_probe_examine(indep_test_func, data_matrix, alpha, eps=epsilonpriv, delta=delta_prime, bias=0.02, **kwargs):

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"
    
    test_count = 0
    
    node_ids = range(data_matrix.shape[1])
    n = data_matrix.shape[0]
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if not g.has_edge(i, j):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    l = 0
    count = 0
    budget_split = 1.0 / 2.0
    eps1 = eps * budget_split
    def noise_scale(x):
        return np.sqrt(x) / np.log(x * (np.exp(eps1)-1) + 1)

    #q = max(min(1. / minimize(noise_scale, [0.5], tol=1e-2).x[0], 1), 1. / 20.)
    eps2 = eps - eps1
    S, _ = quad(lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), 0, 6 / np.sqrt(n))
    sigma1 = 2.0 * S / np.sqrt(q) / np.log((np.exp(eps1)-1.)/q + 1)
    sigma2 = 2 * sigma1
    sigma3 = S / eps2
    # bias = 9 * sigma1

    T0 = alpha - bias + np.random.laplace(0, sigma1)
    row_rand = np.arange(n)
    np.random.shuffle(row_rand)
    dm_subsampled = data_matrix[row_rand[0:int(n*q)]]
    while True:
        cont = False
        remove_edges = []
        
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i,j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
            if len(adj_i) < l:
                continue

            for k in combinations(adj_i, l):
                _logger.debug('indep prob of %s and %s with subset %s'
                                % (i, j, str(k)))
                v = np.random.laplace(0, sigma2)
                p_val = indep_test_func(dm_subsampled, i, j, set(k), **kwargs)[1] + v
                test_count += 1
                _logger.debug('p_val is %s' % str(p_val))

                if p_val < T0:
                    continue
                if p_val >= T0:
                    count += 1
                    T0 = alpha - bias + np.random.laplace(0, sigma1)
                    np.random.shuffle(row_rand)
                    dm_subsampled = data_matrix[row_rand[0:int(n*q)]]
                    v = np.random.laplace(0, sigma3)
                    p_val = indep_test_func(data_matrix, i, j, set(k), **kwargs)[1] + v
                    test_count += 1
                    if p_val >= alpha:
                        if g.has_edge(i, j):
                            _logger.debug('p: remove edge (%s, %s)' % (i, j))
                            if method_stable(kwargs):
                                remove_edges.append((i, j))
                            else:
                                g.remove_edge(i, j)
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
            cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    #eps_prime1 = np.sqrt(2*count*np.log(2/delta))*eps2 + count*eps2*(np.exp(eps2)-1)
    #eps_prime2 = np.sqrt(2*count*np.log(2/delta))*eps1 + count*eps1*(np.exp(eps1)-1)
    #eps_prime = eps_prime1 + eps_prime2
    eps_priv = (count*eps1*eps1)+(count*eps2*eps2)+np.sqrt(2*np.log(1/delta_prime)*((count*eps1*eps1)+(count*eps2*eps2)))
    delta_priv = (q*count*delta_prime)+delta_ad

    return (g, sep_set, eps_priv, delta_priv, test_count)


# In[122]:


def estimate_cpdag(skel_graph, sep_set):
    """Estimate a CPDAG from the skeleton graph and separation sets
    returned by the estimate_skeleton() function.
    Args:
        skel_graph: A skeleton graph (an undirected networkx.Graph).
        sep_set: An 2D-array of separation set.
            The contents look like something like below.
                sep_set[i][j] = set([k, l, m])
    Returns:
        An estimated DAG.
    """
    dag = skel_graph.to_directed()
    node_ids = skel_graph.nodes()
    for (i, j) in combinations(node_ids, 2):
        adj_i = set(dag.successors(i))
        if j in adj_i:
            continue
        adj_j = set(dag.successors(j))
        if i in adj_j:
            continue
        if sep_set[i][j] is None:
            continue
        common_k = adj_i & adj_j
        for k in common_k:
            if k not in sep_set[i][j]:
                if dag.has_edge(k, i):
                    _logger.debug('S: remove edge (%s, %s)' % (k, i))
                    dag.remove_edge(k, i)
                if dag.has_edge(k, j):
                    _logger.debug('S: remove edge (%s, %s)' % (k, j))
                    dag.remove_edge(k, j)

    def _has_both_edges(dag, i, j):
        return dag.has_edge(i, j) and dag.has_edge(j, i)

    def _has_any_edge(dag, i, j):
        return dag.has_edge(i, j) or dag.has_edge(j, i)

    def _has_one_edge(dag, i, j):
        return ((dag.has_edge(i, j) and (not dag.has_edge(j, i))) or
                (not dag.has_edge(i, j)) and dag.has_edge(j, i))

    def _has_no_edge(dag, i, j):
        return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))

    # For all the combination of nodes i and j, apply the following
    # rules.
    old_dag = dag.copy()
    while True:
        for (i, j) in combinations(node_ids, 2):
            # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
            # such that k and j are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Look all the predecessors of i.
                for k in dag.predecessors(i):
                    # Skip if there is an arrow i->k.
                    if dag.has_edge(i, k):
                        continue
                    # Skip if k and j are adjacent.
                    if _has_any_edge(dag, k, j):
                        continue
                    # Make i-j into i->j
                    _logger.debug('R1: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 2: Orient i-j into i->j whenever there is a chain
            # i->k->j.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where k is i->k.
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                # Find nodes j where j is k->j.
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(succs_i & preds_j) > 0:
                    # Make i-j into i->j
                    _logger.debug('R2: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)

            # Rule 3: Orient i-j into i->j whenever there are two chains
            # i-k->j and i-l->j such that k and l are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                # For all the pairs of nodes in adj_i,
                for (k, l) in combinations(adj_i, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->j.
                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue
                    # Skip if not l->j.
                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue
                    # Make i-j into i->j.
                    _logger.debug('R3: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 4: Orient i-j into i->j whenever there are two chains
            # i-k->l and k->l->j such that k and j are nonadjacent.
            #
            # However, this rule is not necessary when the PC-algorithm
            # is used to estimate a DAG.

        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag


# In[123]:



# In[44]:


if dataset in ['asia', 'cancer', 'earthquake']:
    
        #dm, g_answer = bn_data(dataset, size=100000)
        dm, g_answer = bn_data(dataset, size=N)
        maxreach = max(min(int(np.log2(dm.shape[0]))-5, dm.shape[1]-2), 0)
        indeptest=bincondKendall
        taskval = 'bin'
        
else:
    
        #dm, g_answer = bn_data(dataset, size=100000)
        dm, g_answer = bn_data(dataset, size=N)
        maxreach = max(min(int(np.log2(dm.shape[0]))-5, dm.shape[1]-2), 0)
        indeptest=discondKendall
        taskval = 'dis'


# In[124]:


if algo == 'privpc':
    totaleps_priv = []
    totalf1_priv = []
    deltaprivpc = []
    for p in range(0,50):
        (G, sep_set, totalleakage, totaldelta, testnumber) = estimate_skeleton_probe_examine(indep_test_func=indeptest,
                                                             data_matrix=dm,
                                                             alpha=alpha,
                                                             eps=epsilonpriv,
                                                             delta=delta_prime,
                                                             max_reach=maxreach)
        g = estimate_cpdag(skel_graph=G, sep_set=sep_set)
        test_number = []
        deltaprivpc.append(totaldelta)
        test_number.append(testnumber)
        f1_score = cal_f1(g.edges, g_answer.edges)
        recall = cal_recall(g.edges, g_answer.edges)
        precision = cal_precision(g.edges, g_answer.edges)
        totaleps_priv.append(totalleakage)
        totalf1_priv.append(f1_score)
        
elif algo == 'svt':
    totaleps_svt = []
    totalf1_svt = []
    for p in range(0,50):
        (G_SVT, sep_set_SVT, eps_SVT, deltatotalsvt, test_number_SVT) = estimate_skeleton_SVT(indep_test_func=indeptest,
                                                             data_matrix=dm,
                                                             alpha=alpha,
                                                             eps=epsilonpriv,
                                                             delta=delta_prime,
                                                             max_reach=maxreach)
        g = estimate_cpdag(skel_graph=G_SVT, sep_set=sep_set_SVT)
        test_numbersvt = []
        deltasvt = []
        deltasvt.append(deltatotalsvt)
        test_numbersvt.append(test_number_SVT)
        f1_score = cal_f1(g.edges, g_answer.edges)
        recall = cal_recall(g.edges, g_answer.edges)
        precision = cal_precision(g.edges, g_answer.edges)
        totaleps_svt.append(eps_SVT)
        totalf1_svt.append(f1_score)

elif algo == 'em':
    for p in range(0,50):
        (G_EM, sep_set_EM, eps_EM, deltatotalem, test_number_EM) = estimate_skeleton_EM(indep_test_func=indeptest,
                                                             data_matrix=dm,
                                                             alpha=alpha,
                                                             eps=epsilonpriv,
                                                             delta=delta_prime,
                                                             max_reach=maxreach)
        totaleps_em = []
        totalf1_em = []
        g = estimate_cpdag(skel_graph=G_EM, sep_set=sep_set_EM)
        test_numberem = []
        deltaem = []
        test_numberem.append(test_number_EM)
        deltaem.append(deltatotalem)
        f1_score = cal_f1(g.edges, g_answer.edges)
        recall = cal_recall(g.edges, g_answer.edges)
        precision = cal_precision(g.edges, g_answer.edges)
        totaleps_em.append(eps_EM)
        totalf1_em.append(f1_score)
        


elif algo == 'pc':
    alpha = T
    (G, sep_set, test_number) = estimate_skeleton(indep_test_func=indeptest,
                                                             data_matrix=dm,
                                                             alpha=alpha,
                                                             max_reach=maxreach)
    g = estimate_cpdag(skel_graph=G, sep_set=sep_set)
    f1_score = cal_f1(g.edges, g_answer.edges)
    totalleakage = 'infinite'
else:
    totaleps_curate = []
    totalf1_curate = []
    for p in range(0,1):
        (G, sep_set, num, epsval, L,perorderleakage,deltacurate) =  estimate_skeleton_curate(epstotal = eps_total,
                                                                    delta_prime = 1e-14,
                                                                    delta_ad = 1e-14,
                                                                    delta_total = 1e-10,
                                                                     indep_test_func = indeptest,data_matrix = dm,
                                                                    max_reach = maxreach)
        g = estimate_cpdag(skel_graph=G, sep_set=sep_set)
        tests=[]
        f1_score = cal_f1(g.edges, g_answer.edges)
        totaleps_curate.append(np.sum(perorderleakage))
        totalf1_curate.append(f1_score)
        tests.append(np.sum(num))
        print("Every Order Leakage by CURATE: ", perorderleakage)
        
if algo == 'curate':
    print(algo)
    print("Total Leakage is: ",np.mean(totaleps_curate),np.std(totaleps_curate))
    print("The F1-score is: ",np.mean(totalf1_curate),np.std(totalf1_curate))
    #print("Average number of CI tests: ",int(np.mean(tests)))
    print("Average number of CI tests: ",int(np.mean(tests)))
elif algo == 'pc':
    print(algo)
    print("Total Tests : " ,test_number)
    print("F1 score : ",f1_score)
    print("Leakage : " ,totalleakage)

elif algo == 'privpc':
    print(algo)
    print("Total Leakage is: ",np.mean(totaleps_priv),np.std(totaleps_priv))
    print("The F1-score is: ", np.mean(totalf1_priv),np.std(totalf1_priv))
    print("Average number of CI tests: ",test_number)

elif algo== 'svt':
    print(algo)
    print("Total Leakage is: ",np.mean(totaleps_svt),np.std(totaleps_svt))
    print("The F1-score is: ",np.mean(totalf1_svt),np.std(totalf1_svt))
    print("Average number of CI tests: ",test_numbersvt)
elif algo == 'em':
    print(algo)
    print("Total Leakage is: ",np.mean(totaleps_em),np.std(totaleps_em))
    print("The F1-score is: ",np.mean(totalf1_em),np.std(totalf1_em))
    print("Average number of CI tests: ",test_numberem)      


# In[ ]:




