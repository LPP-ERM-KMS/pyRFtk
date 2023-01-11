################################################################################
#                                                                              #
# Copyright 2018-2022                                                          #
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

__updated__ = '2021-10-01 13:55:04'

"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 1 Oct 2021

@author: frederic
"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.constants import speed_of_light as c0

from pyRFtk2 import circuit, rfTRL, rfRLC

L1, L2, LS = 1.0, 0.0, 2.0
fMHz = c0/4/LS

CT1 = circuit(Zbase=50)
CT1.addblock('TL1',rfTRL(Z0TL=30, L=L1))
CT1.addblock('TL2',rfTRL(Z0TL=30, L=L2)) 
CT1.addblock('TL3',rfTRL(Z0TL=20, L=LS)) 
CT1.connect('TL1.2','TL2.2','TL3.2')
CT1.terminate('TL3.1', Z=0)

print(CT1.getS(fMHz, Zbase=30))
print(CT1)

CT2 = circuit(Zbase=50)
CT2.addblock('Shunt1', CT1, ports=['a','b'])
CT2.addblock('Shunt2', CT1, ports=['a','b'])
print(CT2)

CT2.connect('Shunt1.b','Shunt2.a')
CT2.terminate('Shunt2.b',Z=30)
print(CT2)

tSol = CT2.Solution(fMHz, E={'Shunt1.a':1})

CT3 = circuit(Zbase=30)
CT3.addblock('Shunts', CT2)
CT3.addblock('C', rfRLC(Cs=50e-12))
CT3.connect('C.p', 'Shunts.Shunt1.a')
print(CT3)


tSol = CT3.Solution(fMHz, E={'C.s':1})



for node, v in sorted(tSol.items()):
    print(f'{node:30} : {v[0].real:7.3f}{v[0].imag:+7.3f}j')

#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    ...


