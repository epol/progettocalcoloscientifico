#!/usr/bin/env python3
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

import pylab as pl


#def singular_bin_search(to_be_zero,s):
    ## we assume that residual is increasing in mu
    #(n,) = s.shape
    #first = 0
    #last = n
    #while last -first > 1:
        #mid = first + (last -first) /2
        #if to_be_zero((s[mid])**(-1)) > 0:
            #last = mid
        #else:
            #first = mid
    ##print ("bin param "+str((s[last-1])**(-1)))
    #return (s[last-1])**(-1)

def find_param(UTb, s, filters, filtersprime,err):
    residual = lambda mu : numpy.sum( numpy.dot(numpy.diag(filters(mu) - numpy.ones(s.size)) , UTb  ) **2)
    residualprime = lambda mu : 2*numpy.sum((UTb **2) * (filters(mu)-numpy.ones(s.size)) * filtersprime(mu) )
    #mus = numpy.logspace(-10,1,100)
    #residuals=numpy.zeros(shape=mus.shape)
    #print residual(0)
    #for i in range(0,100):
    #    residuals[i] = residual(mus[i])
    #pl.plot(mus,residuals)
    #pl.show()
    to_be_zero = lambda mu: residual(mu) - err**2
    
    # newton zero search
    oldt = -1.
    t = 1e-4
    k=0 # iteration counter
    #while ( abs(t - oldt) > 1e-4*t):
    while ( ( abs(to_be_zero(t)) / err **2 > 1e-5 ) and k<200):
        oldt = t
        t = t - to_be_zero(t) / residualprime(t)
        if (t < 0):
            # because filters are not defined for t<0
            t = oldt /2
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
    print ( "Discrepancy: " + str(to_be_zero(param))) 
    #param = scipy.optimize.newton(to_be_zero,singular_bin_search(to_be_zero,s))#,fprime=residualprime)   #TODO: user fprime: the derivative of to_be_zero
    #param = singular_bin_search(to_be_zero,s)
    print ('param '+str(param))
    return param

def solve(A,b,err,method="newtik"):
    (n,m) = A.shape
    U, s, V = numpy.linalg.svd(A)
    if method.lower() == "newtik":
        filters = lambda mu : numpy.where ( s > mu , 1., s**2 /( mu**2 ))
        filtersprime = lambda mu : numpy.where ( s > mu , 0., -2. * s**2 / mu**3 )
        filtersTIK = lambda mu : s**2 / ( s**2 + mu **2 )
        filtersprimeTIK = lambda mu : - 2* mu * s**2 / ( ( s**2 + mu **2 ) **2 )        
        UTb = numpy.dot( U.T, b)
        param = find_param(UTb,s,filters,filtersprime,err)
        paramTIK = find_param(UTb,s,filtersTIK,filtersprimeTIK,err)
        param = min(param,paramTIK)
        x = numpy.dot ( V.T, numpy.dot( numpy.diag( filters(param) / s ) , UTb ) )
        return x
    elif method.lower() == "tik":
        filters = lambda mu : s**2 / ( s**2 + mu **2 )
        filtersprime = lambda mu : - 2* mu * s**2 / ( ( s**2 + mu **2 ) **2 )
        UTb = numpy.dot( U.T, b)
        param = find_param(UTb,s,filters,filtersprime,err)
        x = numpy.dot ( V.T, numpy.dot( numpy.diag( filters(param) / s ) , UTb ) )
        return x
    elif method.lower() == "tsvd":
        summer = numpy.zeros((m,m))
        summer [ numpy.triu_indices(m,1) ] = 1
        UTb = numpy.dot(numpy.transpose(U) , b)
        residual = numpy.dot( summer , UTb**2 )
        alpha = numpy.argmax (residual < (err)**2)
        ts = s
        ts [ alpha +1: ] = numpy.inf # truncated s
        x = numpy.dot( V.T , numpy.dot(numpy.diag( ts ** (-1) ) , UTb) )
        print ("Discrepancy: "+str(numpy.sum( (numpy.dot(A,x) -b)**2 ) - err**2))
        return x
    else:
        raise Exception ("Unknow method "+method)

