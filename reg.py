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


def regularize(U,s,V,filters,err):
    residual = lambda mu : numpy.linalg.norm(numpy.dot(numpy.diag(filters(mu) - numpy.ones(10)) , U.T  ))
    to_be_zero = lambda mu: residual(mu) - err
    param = (to_be_zero,s[0])   #TODO: user fprime: the derivative of to_be_zero

def solve(A,b,err,method="newtik"):
    (n,m) = A.shape
    U, s, V = numpy.linalg.svd(A)
    new_b = numpy.transpose(U) * b
    dirty_x = V * ( s ** (-1) ) * new_b 
    if method.lower() == "newtik":
        filters = lambda mu : s / ( s**2 + mu **2 )
        
    elif method.lower() == "tik":
        pass
    elif method.lower() == "tsvd":
        summer = numpy.zeros((m,m))
        summer [ numpy.tril_indices(m) ] = 1
        solutions = summer*dirty_x
        trigger = solutions < err 
        alpha = numpy.argmax( trigger )
        x = solutions (alpha)
        return x
    else:
        raise Exception ("Unknow method "+method)

