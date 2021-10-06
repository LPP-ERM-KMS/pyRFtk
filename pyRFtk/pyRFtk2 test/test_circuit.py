"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 12 Feb 2021

@author: frederic
"""
__updated__ = "2021-08-17 13:12:03"

import numpy as np
import matplotlib.pyplot as pl
import sys

from pyRFtk2 import circuit
from pyRFtk2 import junction
from pyRFtk2 import rfTRL

from pyRFtk2.Utilities import printM
from scipy.constants import speed_of_light as c0

# test a connect to a non-existing port
L = 2
f0 = c0/(5*L) # so L is a quarter wave

aTL = rfTRL(L=2, Zbase=30, Z0TL=30)
print(aTL)
bTL = rfTRL(L=2, Zbase=30, Z0TL=[30,30])

ct = circuit(Zbase=30)
ct.addblock('TL1', aTL)
ct.addblock('STB', aTL)
ct.terminate('STB.2',Z=0)
ct.connect('TL1.2', 'STB.1', 'C')
print(ct)
S = ct.getS(f0)
print(ct)
sys.exit(0)

# compare
ct1 = circuit(Zbase=30)
ct1.addblock('TL1', aTL)
ct1.addblock('STB', aTL)
ct1.addblock('C', [[0j,1],[1,0j]])
ct1.connect('TL1.2', 'STB.1', 'C.1')
ct1.terminate('STB.2',Z=0)
S = ct1.getS(f0)
print(ct1)

sys.exit()

# define a quarter wave impedance transformer

dx=0.05
Z1, Z2 = 30, 50
f0, E, Zbase = 30E6, {'1':1, '2': -(Z2-Z1)/(Z2+Z1)}, Z1
QWL = 3e8/f0/4
ZQW = np.sqrt(Z1*Z2)
ZDX = 0.05

aTL = rfTRL(L=2 + QWL, Zbase = 30, dx=dx, 
            Z0TL=[[0.00, 1-ZDX, 1+ZDX, 1 + QWL - ZDX, 1 + QWL + ZDX, 2 + QWL],
                  [  Z1,    Z1,   ZQW,           ZQW,            Z2,      Z2]], 
            OD=0.230, rho=2E-8)

ctQWL = circuit(Zbase=30)
ctQWL.addblock('QWL', aTL, ['p30','p50'], xpos = [0, 2+QWL])
ctQWL.addblock('LWQ', aTL, ['p30','p50'], xpos = [4+2*QWL,2+QWL])
ctQWL.connect('QWL.p50','LWQ.p50')
ctQWL.terminate('LWQ.p30',Z=40)
print(ctQWL)
S30 = ctQWL.getS(f0)
print(ctQWL)

print(ctQWL.maxV(f0, {'QWL.p30':1}, ctQWL.Zbase, 'ctQWL_base'),'\n')



bTL = rfTRL(L=4 + 2* QWL, Zbase = 30, dx=dx,   
            Z0TL=np.array([[ 0.00,              Z1  ],
                           [ 1 - ZDX,           Z1  ],
                           [ 1 + ZDX,           ZQW ],
                           [ 1 + QWL - ZDX,     ZQW ],
                           [ 1 + QWL + ZDX,     Z2  ],
                           [ 3 + QWL - ZDX,     Z2  ],
                           [ 3 + QWL + ZDX,     ZQW ],
                           [ 3 + 2 * QWL - ZDX, ZQW ],
                           [ 3 + 2 * QWL + ZDX, Z1  ],
                           [ 4 + 2 * QWL,       Z1  ]]).T, 
            OD=0.230, rho=2E-8)

ctQWL2 = circuit(Zbase=30)
ctQWL2.addblock('QWL', bTL, ['p1','p2'], xpos=[0,4+2*QWL])
ctQWL2.terminate('QWL.p2',Z=40)

S30 = ctQWL2.getS(f0)
print(ctQWL2)

print(ctQWL2.maxV(f0, {'QWL.p1':1}, ctQWL2.Zbase, 'ctQWL2_base'),'\n')

pl.show()



