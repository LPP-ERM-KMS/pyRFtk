"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 12 Feb 2021

@author: frederic
"""
__updated__ = "2022-04-21 10:52:29"

import numpy as np
import matplotlib.pyplot as pl
import sys
import copy
from scipy.constants import speed_of_light as c0
from Utilities.printMatrices import printMA 
from Utilities.findpath import findpath

from pyRFtk2 import rfBase, circuit, rfCircuit, rfTRL, rfObject, rfGTL, rfRLC, rfArcObj
from pyRFtk2.config import setLogLevel
from pyRFtk2.Utilities import printM, str_dict
from pyRFtk2.CommonLib import plotVSWs, strVSW

alltests = False
tests = [
#    'rfBase-maxV',
#    'rfCircuit-basic',
    'rfCircuit-junctions',
#    'rfArc',
#    'rfBase-basic',
#    'plotVSWs',
#    'rfGTL'
]

setLogLevel('CRITICAL')

def testhdr(t):
    testit = alltests or t in tests
    t1 = ' '.join([c for c in t])
    if testit:
        print('#'*100 +f'\n#\n# t e s t -- {t1}\n#\n')
    return testit

#===============================================================================
#
# r f G T L
#
if testhdr('rfGTL'):
    path2model = findpath('WHu20201021_fpj=453_arc.ciamod',
                          '/mnt/data/frederic/git/iter_d_wws896/ITER_D_WWS896'
                          '/src/CYCLE 2018/2020 Contract/Arc Detection')
    setLogLevel('DEBUG')
    ctGTL = rfGTL(path2model, 
                  objkey='VTL1',
                  Zbase=20, 
                  variables= {'RHO':0.022, 'LMTL' : 1.}
                 )
    printMA(ctGTL.getS(50e6))
    print(ctGTL)
    maxV, where, VSWs = ctGTL.maxV(50e6,{'ss':1, '4pj':0.})
    plotVSWs(VSWs, plotnodes=True)
    
#===============================================================================
#
# r f C i r c u i t - j u n c t i o n s
#
if testhdr('rfCircuit-junctions'):
    A = rfCircuit()
    A.addblock('TL1', rfTRL(L=0))
    A.addblock('TL2', rfTRL(L=0))
    A.connect('TL1.1','TL2.1','TA')
    print(A)
    
    B = rfCircuit()
    B.addblock('TL1', rfTRL(L=0))
    B.addblock('TL2', rfTRL(L=0))
    B.connect('TL1.1','TL2.1','TB')
    
    C = rfCircuit()
    C.addblock('A', A)
    C.addblock('B', B)
    C.addblock('TL3',rfTRL(L=0))
    C.connect('A.TA','B.TB','TL3.1')
    
    printMA(C.getS(1E6))
    print(C.asstr(-1))
    
#===============================================================================
#
# r f C i r c u i t - b a s i c
#
if testhdr('rfCircuit-basic'):
    acircuit = rfCircuit()
    print(acircuit.__str__(1))
    print('\n## rfCircuit: kwargs\n')
    acircuit = rfCircuit(Zbase=30., 
                         Portnames=['a','b','c'],
                         Id='rfCircuit-kwargs',
                         xpos = [0., 1., 2.],
                         )
    print(acircuit.__str__(1))
    
#===============================================================================
#
# r f A r c
#
if testhdr('rfArc'):
    
    tArc = rfArcObj(Larc= 20e-9, Zbase=30)
    print(tArc.__str__(1))
    
#===============================================================================
#
# r f B a s e - b a s i c 
#
if testhdr('rfBase-basic'):
    
    print('\n## no args, kwargs\n')
    base = rfBase()
    print(base.__str__(1))
    
    print('\n## no args, Zbase and S\n')
    base = rfBase(Zbase=20, Ss=[[0j,1],[1,0j]])
    print(base.__str__(1))
    
    print('\n## copy\n')

    printMA(base.getS(0, Zbase=10))
    print(base.__str__(1))
    
    otherbase = base.copy()
    printMA(otherbase.getS(0, Zbase=10))
    print(otherbase.__str__(1))
    
#===============================================================================
#
# r f B a s e - m a x V 
#
if testhdr('rfBase-maxV'):
    
    base = rfBase(Zbase=20., S=[[0j,1],[1,0j]], ports=['a','b'])
    
    print('\nUnknown ports:')
    try:
        base.maxV(10., {'c':1})
    except ValueError as e:
        print('[OK] caught '+repr(e))
    
    print('\nMissing ports:')
    Vmax, where, VSWs = base.maxV(10., {'b':1})
    print(Vmax, where)
    print(VSWs)
    
    print('\nAll ports:')
    base = rfBase(Zbase=20, 
                  S=[[0., np.exp((-0.1 + 1j)*1.)], [np.exp((-0.1 + 1j)*1.), 0.]], 
                  ports = ['a','b'])
    print(base.__str__(1))
    Vmax, where, VSWs = base.maxV(10., {'a':2, 'b':1})
    print(Vmax, where)
    print(VSWs)
    
#===============================================================================
#
# p l o t V S W s
#
if testhdr('plotVSWs'):
    
    XPOS = lambda _: dict([(_p, _x) for _p, _x in zip(_.ports, _.xpos)])
    
    TRL1 = rfTRL(L=1.1, ports=['1a','1b'])
    TRL2 = rfTRL(L=2.1, ports=['2a','2b'])
    print('TRL1.xpos:',XPOS(TRL1))
    
    print('construct CT1')
    CT1 = rfCircuit()
    CT1.addblock('TRL1', TRL1, relpos= 0. )
    CT1.addblock('TRL2', TRL2, relpos= 1.1 )
    CT1.connect('TRL1.1b','TRL2.2a')
    # CT1.resolve_xpos()
    print('CT1.xpos:',XPOS(CT1))
    
    TRL3 = rfTRL(L=1.3, ports=['3a','3b'])
    TRL4 = rfTRL(L=1.4, ports=['4a','4b'])
    CT2 = rfCircuit()
    CT2.addblock('TRL3', TRL3, relpos= 0. )
    CT2.addblock('TRL4', TRL4, relpos= 1.3)
    CT2.connect('TRL3.3b','TRL4.4a')
    CT2.terminate('TRL4.4b', RC=0.5j)
    # CT2.resolve_xpos()
    print('CT2.xpos:',XPOS(CT2))
    
    CT3 = rfCircuit()
    CT3.addblock('CT1', CT1, relpos= 0. )
    CT3.addblock('CT2', CT2, relpos= 0. )
    CT3.connect('CT1.TRL1.1a','CT2.TRL3.3a','ct1')
    # CT3.resolve_xpos()
    print('CT3.xpos:',XPOS(CT3))
    
    CT4 = rfCircuit(Id='Duh')
    CT4.addblock('TRL5', rfTRL(L=2.5, ports=['5a','5b']), relpos= 0. )
    CT4.addblock('CT3', CT3, relpos= 2.5 )
    CT4.connect('TRL5.5b','CT3.ct1')
    # CT4.resolve_xpos()
    print('CT4.xpos:',XPOS(CT4))
    
    print('TRL1:',TRL1)
    setLogLevel('DEBUG')
    maxV, where, VSWs = CT4.maxV(f=45e6, E={'TRL5.5a':1, 'CT3.CT1.TRL2.2b':0.5})
    setLogLevel('CRITICAL')
    print(f'maxV: {maxV}, {where}')
    print(strVSW(VSWs))
    plotVSWs(VSWs)
    
#===============================================================================

pl.show()
