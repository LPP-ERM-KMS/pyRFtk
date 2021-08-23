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

__updated__ = '2020-04-10 19:49:39'

"""
Created on 10 Apr 2020

@author: frederic
"""

import numpy as np
import matplotlib.pyplot as pl

from pyRFtk2.rfObject import rfObject

#===============================================================================
#
#  r f C h i l d
#
class rfChild(rfObject):
    """
    This is a test class to check and learn. 
    It is not meant to be anything funtional.
    """
    def __init__(self, Portnames, **kwargs):
        super().__init__(Portnames, **kwargs)
        self.type = 'child'
        self.mu0 = 3
        
    def __copy__(self):
        new_self = super().__copy__()
        new_self.mu0 = self.mu0
        return new_self
        
    def __repr__(self):
        s = super().__repr__() + '\n# mu0 = %r \n' % self.mu0
        return s
        
#===============================================================================
#        
#  _ _ m a i n _ _
#

if __name__ == '__main__':
    
    a = rfChild('a b c'.split())
    print(a)

