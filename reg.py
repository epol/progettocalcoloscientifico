#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Copyright (C) 2014 - Enrico Polesel
# 
# This is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# any later version.
# 
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software. If not, see <http://www.gnu.org/licenses/>.


import numpy
import numpy.linalg
import scipy.optimize

#import pylab as pl


def find_param(UTb, s, filters, filtersprime,err):
    residual = lambda mu : numpy.sum( numpy.dot(numpy.diag(filters(mu) - numpy.ones(s.size)) , UTb  ) **2)
    residualprime = lambda mu : 2*numpy.sum((UTb **2) * (filters(mu)-numpy.ones(s.size)) * filtersprime(mu) )
    to_be_zero = lambda mu: residual(mu) - err**2
    
    # newton zero search
    oldt = -1.
    t = 1e-4
    k=0 # iteration counter
    while ( ( abs(to_be_zero(t)) / err **2 > 1e-5 ) and k<200):
        oldt = t
        t = t - to_be_zero(t) / residualprime(t)
        if (t < 0):
            # because filters are not defined for t<0
            t = oldt /2
        # OLD debug: plot the zero search
        #mu = numpy.linspace (0,1.2*max(oldt,t) , 100)
        #dehs  = numpy.zeros(100)
        #for i in range (100):
        #    dehs[i] = to_be_zero(mu[i])
        #pl.plot(mu,dehs)
        #pl.plot(mu,residualprime(oldt)*mu + to_be_zero(oldt) - residualprime(oldt)*oldt,'g')
        #pl.plot(mu,numpy.zeros(mu.size),'y')
        #pl.plot(oldt,to_be_zero(oldt),'ro')
        #pl.plot(t*numpy.ones(20),numpy.linspace(-to_be_zero(t), + 2*to_be_zero(t),20),'r')
        #pl.show()
        k = k+1
    param = t
    #param = scipy.optimize.newton(to_be_zero,singular_bin_search(to_be_zero,s))#,fprime=residualprime)   #TODO: user fprime: the derivative of to_be_zero
    #param = singular_bin_search(to_be_zero,s)
    return param

def solve(A,b,err,method="newtik"):
    (n,m) = A.shape
    # SVD of A. WARNING:  A = U * diag(s) * V   (no V.T)
    U, s, V = numpy.linalg.svd(A)
    
    if method.lower() == "newtik":
        # new method filters
        filters = lambda mu : numpy.where ( s > mu , 1., s**2 /( mu**2 ))
        filtersprime = lambda mu : numpy.where ( s > mu , 0., -2. * s**2 / mu**3 )
        # standard tikhonov filters
        filtersTIK = lambda mu : s**2 / ( s**2 + mu **2 )
        filtersprimeTIK = lambda mu : - 2* mu * s**2 / ( ( s**2 + mu **2 ) **2 )
        # U^T *b
        UTb = numpy.dot( U.T, b)
        # curiosity: how much the parameter from discrepancy applied to this filters
        param = find_param(UTb,s,filters,filtersprime,err)
        # now compute the regularization parameter using the standard tikhonov filters
        paramTIK = find_param(UTb,s,filtersTIK,filtersprimeTIK,err)
        param = min(param,paramTIK) # paramTIK should always be the min
        # the solution
        x = numpy.dot ( V.T, numpy.dot( numpy.diag( filters(param) / s ) , UTb ) )
        return x
    elif method.lower() == "tik":
        # standard tikhonov filters
        filters = lambda mu : s**2 / ( s**2 + mu **2 )
        filtersprime = lambda mu : - 2* mu * s**2 / ( ( s**2 + mu **2 ) **2 )
        # U^T *b
        UTb = numpy.dot( U.T, b)
        # use newton to find a parameter using discrepancy principle
        param = find_param(UTb,s,filters,filtersprime,err)
        # the solution
        x = numpy.dot ( V.T, numpy.dot( numpy.diag( filters(param) / s ) , UTb ) )
        return x
    elif method.lower() == "tsvd":
        # (summer * v )[i] = sum _i ^n v[i]
        summer = numpy.zeros((m,m))
        summer [ numpy.triu_indices(m,1) ] = 1
        # U^T *b
        UTb = numpy.dot(numpy.transpose(U) , b)
        residual = numpy.dot( summer , UTb**2 )
        # use discrepancy principle
        alpha = numpy.argmax (residual < (err)**2)
        # truncated singual value vector
        ts = s
        ts [ alpha +1: ] = numpy.inf # because it's easier to use
        # the solution
        x = numpy.dot( V.T , numpy.dot(numpy.diag( ts ** (-1) ) , UTb) )
        return x
    else:
        raise Exception ("Unknow method "+method)

