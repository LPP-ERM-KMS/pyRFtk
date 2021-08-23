################################################################################
#                                                                              #
# Copyright 2018-2020                                                          #
#                                                                              #
#                   Laboratory for Plasma Physics                              #
#                   Royal Military Academy                                     #
#                   Brussels, Belgium                                          #
#                                                                              #
#                   ITER Organisation                                          #
#                                                                              #
# Author : frederic.durodie@rma.ac.be                                          #
#                          @gmail.com                                          #
#                          @ccfe.ac.uk                                         #
#                          @telenet.be                                         #
#                          .lpprma@tlenet.be                                   #
#                                                                              #
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by   #
# the European Commission - subsequent versions of the EUPL (the "Licence");   #
#                                                                              #
# You may not use this work except in compliance with the Licence.             #
# You may obtain a copy of the Licence at:                                     #
#                                                                              #
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-11-12                  #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the Licence is distributed on an "AS IS" basis,            #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the Licence for the specific language governing permissions and          #
# limitations under the Licence.                                               #
#                                                                              #
################################################################################

__updated__ = '2020-04-29 18:34:58'

"""
Created on 11 Apr 2020

@author: frederic
"""

import numpy as np
import matplotlib.pyplot as pl


from pyRFtk2.rfObject import rfObject 
from Utilities.printMatrices import strM

#===============================================================================

def fill(a, fs):
    
    S = np.array([[i/10 + j/100 for j in range(1,a.N+1)] 
                  for i in range(1,a.N+1)], dtype=np.complex)

    for k, f in enumerate(fs):
        a.setS(f, S + k)
        
    
#===============================================================================

def show(a):
    print(a)
    for f, S in a:
        print('%7.3f %s    ' % (f, a.funit), end='')
        print(strM(S))
        
#===============================================================================

def test_create(ports, fs, **kwargs):
    
    quiet = kwargs.pop('quiet', False)
    
    if not quiet:
        print('# -----------------------------------------------------------')
        print('# __init__')
        print('# -----------------------------------------------------------')
    
    b = rfObject(Portnames=ports.split(), **kwargs)
    fill(b,fs)
    if not quiet:
        show(b)
    
    return b
    
#===============================================================================

def test_copy():
    
    print('# -----------------------------------------------------------')
    print('# __init__')
    print('# -----------------------------------------------------------')

    b = rfObject()
    fill(b, [10,20,30])
    a = b.copy()
    print(a)
    print(b)
    
#===============================================================================

def test_setS():
    print('# -----------------------------------------------------------')
    print('# setS')
    print('# -----------------------------------------------------------')

    ports = 'a b c'.split()
    b = rfObject(Portnames=ports)
    fill(b, [30, 35, 40])
    show(b)
    
#===============================================================================

def iterator(ports='a b c', fs=[30, 40, 50]):
    print('# -----------------------------------------------------------')
    print('# __iter__')
    print('# -----------------------------------------------------------')
    tports = ports.split()
    b = rfObject(Portnames=tports)
    fill(b, fs)
    show(b)
    
#===============================================================================

def sortports():
    print('# -----------------------------------------------------------')
    print('# sortports')
    print('# -----------------------------------------------------------')
    
    ports = 'a b c'.split()
    b = rfObject(Portnames=ports)
    fill(b, [20])
    show(b)
    
    b.sortports(order=['c'])
    print(b.Portnames)
    show(b)

    b.sortports(order='b c a'.split())
    print(b.Portnames)
    show(b)
    
#===========================================================================

def test_pickle(portstr, fs, fs2):
    print('# -----------------------------------------------------------')
    print('# __getstate__  and  pickle')
    print('# -----------------------------------------------------------')
    
    from pickle import dumps, loads
    
    b = test_create(Portnames=portstr, fs=fs, quiet=True)
    b._interp_(fs2, insert=True)
        
    c = loads(dumps(b))
    
    print(b.__eq__(c, True))
    
#===========================================================================

def test_read_tsf(fpath):
    print('# -----------------------------------------------------------')
    print('# from_tsf')
    print('# -----------------------------------------------------------')
    
    b = test_create(fs=[], quiet=True)
    b.read_tsf(fpath, funit='MHz')

#===========================================================================

def interpolate():
    print('# -----------------------------------------------------------')
    print('# interpolate')
    print('# -----------------------------------------------------------')
    
    b = test_create(Portnames = 'a b', fs=[20, 40, 60, 80], quiet=True)
    b._interp_([30, 50, 70], insert=True)
    show(b)

