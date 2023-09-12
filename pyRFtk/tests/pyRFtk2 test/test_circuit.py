"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 12 Feb 2021

@author: frederic
"""
__updated__ = "2021-12-15 15:57:43"

import numpy as np
import matplotlib.pyplot as pl
import sys
import copy
from scipy.constants import speed_of_light as c0

from pyRFtk2 import circuit, rfTRL, rfObject, rfGTL, rfRLC
from pyRFtk2.config import setLogLevel
from pyRFtk2.Utilities import printM, str_dict
from pyRFtk2.CommonLib import plotVSWs

alltests = False
tests = [
    'add block port renaming',
#    'copy/deepcopy rfObject',
#    'copy/deepcopy',
#    'update nested blocks and uniqueness',
#    'replace/update blocks',
#    'port reordering',
#    'connect to a non-existing port',
]

#===============================================================================
#add block port renaming
if alltests or 'add block port renaming' in tests:
    print('#\n# test -- add block port renaming\n#\n')
    
    
    
    setLogLevel('DEBUG')
    
    C1 = circuit()
    C1.addblock('TL1',rfTRL(L=1, ports=['a','b']), xpos= [0., 1.])
    C1.addblock('TL2',rfTRL(L=2), xpos= [1., 3.], ports=['c','d'])
    C1.addblock('RLC', rfRLC(Cp=100e-12),ports=['m','n'])
    C1.connect('TL1.b','TL2.c')
    C1.connect('TL2.d','RLC.m')
    print(C1)

    Umax1, where1, VSWs1 = C1.maxV(f=50e6, E={'TL1.a':1, 'RLC.n':1})
    print(str_dict(VSWs1))
    setLogLevel('CRITICAL')
    
    C2 = circuit()
    C2.addblock('TLs', C1, ports=['A','B'])
    print(C2)
    
    Umax2, where2, VSWs2 = C2.maxV(f=50e6, E={'TLs.A':1, 'TLs.B':1})
    print(str_dict(VSWs2))
    
    plotVSWs(VSWs2, maxlev=10, ID="C1")
    plotVSWs(VSWs2, maxlev=10, ID="C2")

    pl.show()
    
#===============================================================================
#copy/deepcopy rfObject
if alltests or 'copy/deepcopy rfObject' in tests:
    print('#\n# test -- copy/deepcopy rfObject\n#\n')
    
    rfObj = rfObject(
        touchstone='../test data/4pj-L4PJ=403mm-lossy_normalized_50ohm_1.s4p',
        Zbase = 20)
    
    S = rfObj.getS(47.456)
    rfObj2 = rfObj.copy()
    
    print(rfObj, id(rfObj), id(rfObj.Ss), '\n', rfObj.getS(47.456), '\n')
    print(rfObj2, id(rfObj2), id(rfObj2.Ss), '\n', rfObj.getS(47.456), '\n')
    EQ = rfObj.__eq__(rfObj2, verbose=True)
    print(EQ)
    
#===============================================================================
#copy/deepcopy
if alltests or 'copy/deepcopy' in tests:
    print('#\n# test -- copy/deepcopy\n#\n')
    
    CT = circuit(Zbase=30)
    CT.addblock('TL1', rfTRL(L=1))
    CT1 = copy.copy(CT)
    CT1.addblock('TL2',rfTRL(L=2))
    CT.addblock('TL3',rfTRL(L=3))
    CT.connect('TL1.','TL3.1')
    
    print(CT)
    print(CT1)
    
    CT.addblock('CT1', CT1)
    CT2 = CT.copy()
    
    print(CT)
    print(CT2)
    
#===============================================================================
#replace/update blocks
if alltests or 'update nested blocks and uniqueness' in tests:
    print('#\n# test -- pdate nested blocks and uniqueness\n#\n')
    
    # create a block that will be shared and updated
    rfShared = circuit()
    rfShared.addblock('TL', rfTRL(L=1, Z0TL=rfShared.Zbase))
    rfShared.connect('TL.1','1')
    rfShared.connect('TL.2','2')
    
    rfCircuit = circuit()
    rfCircuit.addblock('TL1', rfShared)
    rfCircuit.addblock('TL2', rfShared)
    
    rfCircuit.getS(50e6)
    print(rfCircuit)
    
    rfCircuit.addblock('TL1.TL',rfTRL(L=0.5, Z0TL=rfShared.Zbase))
    
    rfCircuit.getS(50e6)
    print(rfCircuit)
   
    
    
#===============================================================================
#replace/update blocks
if alltests or 'replace/update blocks' in tests:
    print('#\n# test -- replace/update blocks\n#\n')

    fHz = 55e6
    L = c0/fHz/2
    
    print(' #\n # update top level block:\n #\n')
    ct = circuit()
    
    ct.addblock('TL1', rfTRL(L=L, Z0TL=ct.Zbase))
    ct.getS(fHz)
    print(ct)
    
    ct.addblock('TL1', rfTRL(L=2*L, Z0TL=ct.Zbase))
    ct.getS(fHz)
    print(ct)
    
    print(' #\n # update nested level block:\n #\n')
    cttop = circuit()
    cttop.addblock('Obj', ct, ports=['a','b'])
    cttop.getS(fHz)
    print(cttop)
    
    cttop.addblock('Obj.TL1', rfTRL(L=L, Z0TL=ct.Zbase))
    cttop.getS(fHz)
    print(cttop)
    
    
#===============================================================================
# port reordering 
if alltests or 'port reordering' in tests:
    print('#\n# test -- port reordering\n#\n')
    ct = circuit(Zbase = 30.)
    ct.addblock('TL1', rfTRL(L=1.0, Z0TL=20))
    ct.addblock('STB', rfTRL(L=1.0, Z0TL=30))
    ct.terminate('STB.2',Z=0)
    ct.connect('TL1.2', 'STB.1', 's')
    print(ct)
    ct.connect('TL1.1', 't')
    print(ct)
    
    ct.ports = sorted(ct.ports,reverse=False)
    print(ct.ports)
    print(ct.getS(50e6),'\n')
    
    ct.ports = sorted(ct.ports,reverse=True)
    print(ct.ports)
    print(ct.getS(50e6),'\n')
    
    print(ct)

#===============================================================================
# connect to a non-existing port
if alltests or 'connect to a non-existing port' in tests:
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



