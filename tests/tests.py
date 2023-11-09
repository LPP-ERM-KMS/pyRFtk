"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 12 Feb 2021

@author: frederic
"""
__updated__ = "2023-11-09 11:13:43"

import numpy as np
import matplotlib.pyplot as pl
import sys
sys.path.insert(0, '../src')
# import copy
# from scipy.constants import speed_of_light as c0

from pyRFtk.findpath import findpath                                            # @UnresolvedImport
from pyRFtk import rfBase, rfCircuit, rfTRL, rfGTL, rfRLC, rfArcObj             # @UnresolvedImport
from pyRFtk.config import setLogLevel                                           # @UnresolvedImport
from pyRFtk import printMA, printRI                                                      # @UnresolvedImport
from pyRFtk import plotVSWs, strVSW                                             # @UnresolvedImport

alltests = False
tests = [
    'connect_issue',
#     'rfRLC',
#     'DEMO_KoM',
#     'deembed',
#     'port-order',
#     'rfBase-maxV',
#     'rfCircuit-basic',
#     'rfCircuit-junctions',
#     'rfArc',
#     'rfBase-basic',
#     'plotVSWs',
#     'rfGTL',
]

setLogLevel('DEBUG')

def testhdr(t):
    testit = alltests or t in tests
    t1 = ' '.join([c for c in t])
    if testit:
        print('#'*100 +f'\n#\n# t e s t -- {t1}\n#\n')
    return testit

#===============================================================================
#
# connect_issue
#
if testhdr('connect_issue'):
    fHz = 10e6
    
    setLogLevel('DEBUG')
    ct = rfCircuit()
    tl = rfTRL(L=1)
    ct.addblock('TL1', tl)
    ct.addblock('TL2', tl)
    
    if False and (fHz > 0):
        # !! cannot yet get the S because this would freeze the circuit after
        #    which one could not alter connections or add blocks any more
        print('S for the two TLs separately')
        printMA(ct.getS(fHz))
        
    print(ct.asstr(-1))
    
    ct.connect('TL1.1','TL2.1','1')
    ct.connect('TL1.2','TL2.2','2')
    if fHz > 0:
        print('S for the two TLs connected')
        printMA(ct.getS(fHz))
    print(ct.asstr(-1))

    ct.terminate('1',Z=0)
    if fHz > 0:
        print('S for the two TLs connected and terminated')
        printMA(ct.getS(fHz))
    print(ct.asstr(-1))
    
    print('S for the two TLs connected and terminated')
    S = ct.getS(fHz)
    printMA(S)
    print(ct.asstr(-1))
    

    
    # these are 2 equal length TLs in parallel and thus are equivalent to
    # a single TL of the same length and half the Z0TL
    
    ctc = rfCircuit()
    ctc.addblock('TLC', rfTRL(L=tl.L, Z0TL=tl.TLP.Z0TL()/2))
    ctc.terminate('TLC.1', Z=0)
    SC = ctc.getS(fHz)
    printMA(SC)
    
    printMA(S - SC)
        
    
    
#===============================================================================
#
# rfRLC
#
if testhdr('rfRLC'):
    
    setLogLevel('CRITICAL')
    fHz = 50e6
    ct = rfCircuit()
    ct.addblock('C1s', rfRLC(Cs=10e-12))
    ct.addblock('C1p', rfRLC(Cp=20e-12))
    ct.connect('C1s.p', 'C1p.s')

    printRI(ct.getS(55e6))
    print(ct.asstr(full=-1))
    
    setLogLevel('DEBUG')
    # ct.blocks['C1s']['object'].set(Cs=20e-12)
    ct.set('C1s.Cs', 20e-12)
    
    printRI(ct.getS(55e6))
    setLogLevel('CRITICAL')
    print(ct.asstr(full=-1))
    
#===============================================================================
#
# p o r t - o r d e r
#
if testhdr('port-order'):
    
    fHz = 50e6
    ct = rfCircuit()
    TLS = []
    for k, L in enumerate([0.5,1.0,1.5]):
        TLS.append(f'TL{k+1}')
        ct.addblock(TLS[-1], rfTRL(L=L), ports=[f'{k+1}','t'])
    ct.connect(*[f'{t}.t' for t in TLS])
    
    ct.Portnames = [f'{t}.{t[-1]}' for t in TLS]

    print(ct.asstr(-1))
    SN = ct.getS(fHz)
    printMA(SN)
    
    ct.Portnames.reverse()
    ct.S = None
    ct.invM = None
    print(ct.Portnames)
    SR = ct.getS(fHz)
    printMA(SR)

    M = np.array([[0j,0j,1], [0j,1,0j], [1,0j,0j]])
    
    printMA(SN - M  @ SR @ M)
 
#===============================================================================
#
# d e e m b e d 
#
if testhdr('deembed'):
    
    # equal internal and exteral ports
    #
    #         +----------------------------+
    #   TL.1 -+           TL               +- TL.2
    #         +----------------------------+
    #
    # 
    #         +-----+                +---------+
    #   TL.1 -+ TLi +- TL.i = DTL.i -+   DTL   +- DTL.e = TL.2
    #         +-----+                +---------+
    
    ct = rfCircuit()
    ct.addblock('TL', rfTRL(L=1))
    ct.addblock('DTL', rfTRL(L=1), ports=['i','e'])
    print(ct.asstr(-1))
    ct.deembed([('DTL.i','TLi')], [('DTL.e','TL.2')])
    print(ct.asstr(-1))
    printMA(ct.getS(55E6))
    
    print('\n','-'*120,'\n')
    
    #
    # ct = rfCircuit()
    # ct.addblock('TL', rfTRL(L=1), ports=['1','t'])
    # # printMA(ct.getS(55e6))
    #
    # # more external than internal ports
    #
    # ct.addblock('TL1', rfTRL(L=1), ports=['t','2'])
    # ct.addblock('TL2', rfTRL(L=1), ports=['t','2'])
    # ct.connect('TL.t','TL1.t','TL2.t')
    #
    # ct.addblock('DTL1', rfTRL(L=1), ports=['t','e'])
    # ct.addblock('DTL2', rfTRL(L=1), ports=['t','e'])
    # ct.connect('DTL1.t','DTL2.t','DTLi')
    # print(ct.asstr(-1))
    #
    # ct.deembed([('DTLi','TL.i')], [('DTL1.e','TL1.2'), ('DTL2.e','TL2.2')])
    # print(ct.asstr(-1))
    # printMA(ct.getS(55E6))
    #

    # more internal than external ports
    
    ct = rfCircuit()
    
    ct.addblock('TL1', rfTRL(L=1), ports=['t','1'])
    ct.addblock('TL2', rfTRL(L=1), ports=['t','2'])
    ct.addblock('TL3', rfTRL(L=1), ports=['t','3'])
    ct.connect('TL1.t','TL2.t','TL3.t')
    ct.connect('TL2.2','TL3.3','TL_common')

    printMA(ct.getS(55E6))
    
    dt = rfCircuit()
    dt.addblock('DTL1', rfTRL(L=1), ports=['i1','t'])
    dt.addblock('DTL2', rfTRL(L=1), ports=['i2','t'])
    dt.connect('DTL1.t', 'DTL2.t', 'DTLext')
    
    printMA(dt.getS(55E6))
    
    CT = rfCircuit()
    CT.addblock('spider', ct)
    CT.addblock('deembed', dt)
    
    print(CT.asstr(1))

    CT.deembed({'deembed.DTL1.i1':'spider_i1', 'deembed.DTL2.i2':'spider_i2'}, 
               {'deembed.DTLext': 'spider.TL_common'})
    
    print(CT.asstr(1))
    printMA(CT.getS(55e6))
    
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

if testhdr('DEMO_KoM'):
    
    TRL1 = rfTRL(L=1.1, OD=0.230, ID=[0.100, 0.130], dx=360) # a conical TL
    TRL2 = rfTRL(L=1.1, Z0TL=40, dx=360)
    TRL3 = rfTRL(L=2, ports=['E', 'T'], Zbase=40, dx=360) # <- just for fun
    RLC2 = rfRLC(Cp=100e-12)
    
    ct = rfCircuit()
    ct.addblock('TL1', TRL1, ports=['T', 'E'], relpos=TRL3.L)
    ct.addblock('TL2', TRL2, ports=['T', 'E'], relpos=TRL3.L)
    ct.addblock('TL3', TRL3)
    ct.addblock('Cap', RLC2, ports=['E','oc'], relpos=TRL3.L + TRL2.L)
    ct.connect('TL1.T', 'TL2.T', 'TL3.T')
    ct.connect('TL1.E', 'Cap.E')
    ct.terminate('Cap.oc', Y=0)   # open circuit !
    ct.terminate('TL2.E', Z=10)  # finite impedance
    
    maxV, where, VSWs = ct.maxV(f=55e6, E={'TL3.E': 1})
    plotVSWs(VSWs) 
    print(f'max: {maxV:.3f}V {where}')

pl.show()
