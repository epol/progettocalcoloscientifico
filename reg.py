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


def regularize(U,s,V,b,filters,filtersprime,err):
    # U^T *b
    UTb = numpy.dot( U.T, b)
    residual = lambda mu : numpy.sqrt(numpy.linalg.norm( (numpy.diag(filters(mu) - numpy.ones(10)) * UTb  )) )
    residualprime = lambda mu : numpy.sum((UTb **2) * filters(mu) * filtersprime(mu) ) / residual(mu)
    to_be_zero = lambda mu: residual(mu) - err
    param = scipy.optimize.newton(to_be_zero,s[numpy.argmin(s>0) -1])#,fprime=residualprime)   #TODO: user fprime: the derivative of to_be_zero
    if (param < 0):
        param = 0
    x = numpy.dot ( numpy.dot ( V.T, numpy.diag( filters(param) / s ) ) , UTb )
    return x

def solve(A,b,err,method="newtik"):
    (n,m) = A.shape
    U, s, V = numpy.linalg.svd(A)
    if method.lower() == "newtik":
        filters = lambda mu : numpy.where ( s < mu , s**2 / mu**2 , 1)
        filtersprime = lambda mu : 2*numpy.where ( s < mu , -2 * s**2 / mu**3 , 0)
        x= regularize(U,s,V,b,filters,filtersprime,err)
        return x
    elif method.lower() == "tik":
        filters = lambda mu : s**2 / ( s**2 + mu **2 )
        filtersprime = lambda mu : - 2* mu * s**2 / ( ( s**2 + mu **2 ) **2 )
        x= regularize(U,s,V,b,filters,filtersprime,err)
        return x
    elif method.lower() == "tsvd":
        summer = numpy.zeros((m,m))
        summer [ numpy.tril_indices(m) ] = 1
        UTb = numpy.dot(numpy.transpose(U) , b)
        dirty_x_newbase = numpy.dot(numpy.diag( s ** (-1) ) , UTb)
        solutions_newbase = numpy.dot(summer,dirty_x_newbase)
        trigger = solutions_newbase < err 
        alpha = numpy.argmin( trigger ) -1
        x_newbase = dirty_x_newbase
        x_newbase [ alpha+1: ] = 0
        x = numpy.dot( V.T , x_newbase)
        return x
    else:
        raise Exception ("Unknow method "+method)

