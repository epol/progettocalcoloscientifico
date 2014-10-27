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



import sys
import csv

import numpy
import numpy.linalg
import pylab

# my library
import reg

# because i need pytave
sys.path.insert(0,"./lib/")
import pytave

# because i use regtools
pytave.addpath('./lib/regu/')

N = 200
rel_err = 1e-2

def example(name,N,rel_err,eta=1):
    # select the problem
    if name in [ 'phillips','shaw' ]:
        (A,b,x) = pytave.feval(3,name,N)
    elif name=='ilaplace':
        (A,b,x,tLAP) = pytave.feval(4,'i_laplace',N,1)
    else:
        raise Exception("I don't know problem "+name)
    
    # 1 dimension arrays are 2dim in octave. Let's flat them!
    b = b.flatten()
    x = x.flatten()
    
    # colors for the plot
    colors = {
        'real' : 'blue',
        'tsvd' : 'red',
        'tik' : 'green',
        'newtik' : 'black'
        }

    # calculate the error and disturb b with it
    err = numpy.linalg.norm(b,2) * rel_err
    e = numpy.random.randn(b.size)
    e = err / numpy.linalg.norm(e,2) *e
    tilde_b = b + e
    
    # dictionary for the computed solutions
    tilde_x = {}
    errors = {}

    # regularize the problem, 'real' is a dummy method for the real solution
    for method in [ 'real', 'tsvd', 'tik', 'newtik' ]:
        if method is not 'real':
            tilde_x[method] = reg.solve(A,tilde_b,eta*err,method)
            errors[method] = (numpy.linalg.norm(tilde_x[method] - x,2))/numpy.linalg.norm(x,2)
            #print ( method + " " + str((numpy.linalg.norm(tilde_x[method] - x,2))/numpy.linalg.norm(x,2)) )
        else:
            tilde_x[method] = x
        #pylab.plot(tilde_x[method],colors[method],label=method)

    # plot the solutions
    for method in [ 'real', 'tsvd', 'tik', 'newtik' ]:
        pylab.plot(tilde_x[method],colors[method],label=method)
    if name is 'shaw':
        pylab.legend(loc='upper left')
    else:
        pylab.legend(loc='upper right')
    pylab.title(name+" " + str( rel_err*100 ) + "%")
    pylab.savefig(name+'_'+str(int(rel_err*1000))+'.png')
    #pylab.show()
    pylab.clf()
    
    return errors

def space_err(name,N):
    table = {}
    for rel_err in [ 1e-1, 5e-2, 1e-2, 1e-3 ]:
        table[rel_err] = example(name,N,rel_err)
    
    return table

if __name__ == "__main__":
    data = {}
    for name in ['phillips','shaw','ilaplace' ]:
        data[name] = space_err(name,200)
    
    for name in data.keys():
        print ("Problem "+str(name)+"\n")
        for err in data[name].keys():
            print (str(err)+"\t")
            for method in [ 'tsvd', 'tik', 'newtik' ]:
                print (str(data[name][err][method])+"\t")
        print ("\n")
    
    for name in data.keys():
        outfile = open(name+'.csv','w')
        outwriter = csv.writer (outfile)
        for err in [.001,.01,.05,.1]:
            row = [str(err*100)+'%']
            for method in [ 'tsvd', 'tik', 'newtik' ]:
                row.append(str(data[name][err][method]))
            outwriter.writerow(row)
        outfile.close()
    
    for name in data.keys():
        outfile = open(name+'.tex','w')
        outfile.write('\\begin{tabular} { c | c | c | c }\n')
        outfile.write('err. & TSVD & st. Tik & new Tik \\\\ \hline \n')
        for err in [.001,.01,.05,.1]:
            outfile.write ( str(err*100)+'\\% & $'+str(data[name][err]['tsvd'])+'$ & $' +str(data[name][err]['tik'])+'$ & $' +str(data[name][err]['newtik'])+'$ \\\\\n' )
        outfile.write('\\end{tabular}\n')
        outfile.close()
    
    
