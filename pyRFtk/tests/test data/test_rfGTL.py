################################################################################
#                                                                              #
# Copyright 2018-2022                                                         #
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
#                          .lpprma@telenet.be                                  #
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

__updated__ = '2022-02-09 15:16:23'

"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 6 Oct 2021

@author: frederic
"""

import numpy as np
import matplotlib.pyplot as pl
import os
import sys

from pprint import pprint

from Utilities import ReadDictData
from pyRFtk2 import circuit, rfTRL, rfRLC, rfGTL, rfObject
from pyRFtk2.CommonLib import plotVSWs

alltests = False
tests = [
    'hasattr_Solution',
    '__str__',
#     'legacy',
]

path2model = '/home/frederic/git/iter_d_wws896'                \
             '/ITER_D_WWS896/src/CYCLE 2018/SS/WHu20201021'   \
             '/S_STSR_XN7VDD_v1p0.modss'

#===============================================================================
# hasattr_Solution
if 'hasattr_Solution' in tests or alltests:
    print('#\n# test -- hasattr_Solution\n#\n')
    gtlCT = rfGTL(path2model, variables={'DX':0.01})
    print(f'hasattr_Solution : {hasattr(gtlCT, "Solution")}\n')
    
#===============================================================================
# __str__
if '__str__' in tests or alltests:
    print('#\n# test -- __str__\n#\n')
    gtlCT = rfGTL(path2model, variables={'DX':0.01})
    print(gtlCT,'\n')
    print(gtlCT.__str__(-1),'\n')

    
#===============================================================================
# lecagy 
if 'legacy' in tests or alltests:

    print('#\n# test -- legacy\n#\n')
    gtlCT = rfGTL(path2model, variables={'DX':0.01})
    
    fMHzs = np.linspace(35, 65, num=51)
    Zbase = 20
    
    print('getting gtlSs ', end='')
    gtlSs = gtlCT.getSgtl(fMHzs*1E6, Zbase=Zbase)
    
    print('getting tsfSs ', end='')
    tsfSs = gtlCT.getS(fMHzs*1E6, Zbase=Zbase)
    
    fig, axs = pl.subplots(2, 4, sharex=True, figsize=(12,8), num='test -- legacy #1')
    for r_ in range(2):
        for c_ in range(2):
            pl.sca(axs[r_][c_])
            pl.plot(fMHzs, np.abs(gtlSs[:,r_,c_]), 'r', label='gtl')
            pl.plot(fMHzs, np.abs(tsfSs[:,r_,c_]), 'b', label='tsf')
            pl.grid()
            if r_ == 1:
                pl.xlabel('frequency [MHz]')
            pl.title(f'| S$_{{{r_+1}{c_+1}}}$ |')
            pl.legend(loc='best')
            
            pl.sca(axs[r_][c_+2])
            pl.plot(fMHzs, np.angle(gtlSs[:,r_,c_], deg=1), 'r', label='gtl')
            pl.plot(fMHzs, np.angle(tsfSs[:,r_,c_], deg=1), 'b', label='tsf')
            pl.grid()
            if r_ == 1:
                pl.xlabel('frequency [MHz]')
            pl.title(f'phase( S$_{{{r_+1}{c_+1}}}$ ) [deg]')
            pl.legend(loc='best')
            
    pl.suptitle(f'{os.path.basename(path2model)} on {Zbase} Ohm')
    pl.tight_layout(rect = [0,0,1,0.96])
    
    print(gtlCT.getSgtl(50e6,Zbase=Zbase))
    print(gtlCT.getS(50e6, Zbase=Zbase))
    
    print()
    print(gtlCT)
    
    C1 = circuit(Id='C1')
    C1.addblock('TL1', rfTRL(L=1, Z0TL=40), xpos=[0.0, 1.0])
    C1.addblock('C', rfRLC(Cp=5e-12, Rp=40))
    C1.connect('C.s','TL1.2')
    
    C2 = circuit(Id='C2')
    C2.addblock('TL', C1, xpos=[0.0, 1.0])
    print(C2)
    
    Vmax, where, VSWs = C2.maxV(50E6, {'TL.TL1.1':0, 'TL.C.p':1})
    print(Vmax, where)
    plotVSWs(VSWs, num='test -- legacy #2')
    pprint(VSWs)
    
pl.show()
