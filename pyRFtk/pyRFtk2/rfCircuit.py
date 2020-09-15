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

__updated__ = '2020-09-14 09:17:55'

"""
Created on 9 Apr 2020

@author: frederic

try the approach everything is a circuit

methods of the circuit:

__add__
__iadd__
__sub__ (deembed)
__isub__

setf(f, unit)
compact()  -> return S-matrix of free ports

Zbase, freq, funit

M is the circuit matrix (i
S is the compacted matrix (i.e only external ports)
Ports are the external ports

properties of a (sub)circuit:

{'toplevel': (None, 0, 0, 0), <-- tbd
 ...
 name:s : (pointer2rfobject, size:d, row:d, col:d),
 ...
}
"""

import numpy as np

from pyRFtk2.rfObject import rfObject

#===============================================================================
#
#  r f C i r c u i t
#

class rfCircuit(rfObject):
    
    #===========================================================================
    #
    #  _ _ i n i t _ _
    #
    def __init__(self, ports, **kwargs):
        
        #TODO: TBD Portnames in this context
        #      e.g. these could be the external ports
        
        super().__init__(ports, **kwargs)
        self.type = 'circuit'
    
    
#===============================================================================
#        
#  _ _ m a i n _ _
#

if __name__ == '__main__':
    
    a = rfCircuit('a b c'.split())
    
    print(a)
