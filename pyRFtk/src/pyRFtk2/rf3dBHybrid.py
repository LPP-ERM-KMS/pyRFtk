###               see copyright notice at the end of the file                ###

__updated__ = '2023-03-30 10:36:30'

"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 11 Apr 2022

@author: frederic
"""

import numpy as np
degSign = u'\N{DEGREE SIGN}'

if __name__ == '__main__':
    from pyRFtk2 import rfBase
    from pyRFtk2.config import logit, logident
    from pyRFtk2.whoami import whoami
else:
    from . import rfBase
    from .config import logit, logident
    from .whoami import whoami

from scipy.constants import speed_of_light as c0

#===============================================================================
#
# r f 3 d B H y b r i d
#
class rf3dBHybrid(rfBase):
    
    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, *args, **kwargs):
        """
        portnames are: 'A1', 'A2' (for strip 'A') and          A1             A2
                       'B1', 'B2' (for strip 'B'); ports        |             |
                       with equal numerals face each other.     +-------------+
                                                                +-------------+
                                                                |             |
                                                               B1             B2
    
        return the Scatter object of a hybrid coupler defined by :
        
          Identifier          Id     []    'rf3dBHydrid_#'
          ref. impedance      Zbase  [Ohm] (defaults to rfBase.Zbase)
          coupler length      Lc     [m]   (defaults to quarter wave lenght when 
                                            f0Hz is given)
          coupler center f    f0Hz   [Hz]  (defaults to quarter wave f when Lc
                                            is given)
          coupling constant   kc     [ampl. fraction] (defaults to sqrt(0.5))                                  
          coupling constant   kcdB   [dB]  (defaults to 20.*np.log10(kc) = 3. db)
        
        Note:
        
            Zbase defines the 3dB hybrid coupler. So when including the hybrid
            coupler in a circuit with a different Zbase the one should be
            careful to properly set Zbase for the 3dB hybrid coupler. 
        """
        
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', printargs=False)

        type_rf3dBHybrid = type(self).__name__ == 'rf3dBHybrid'
        
        if not hasattr(self,'args'):
            self.args = args
            
        if not hasattr(self,'kwargs'):
            self.kwargs = kwargs.copy()
        
        ports = kwargs.pop('ports', ['A1','A2','B1','B2'])
        if len(ports) != 4:
            raise ValueError(
                f'{whoami(__package__)}: there must be exactly 4 ports' # @UndefinedVariable
            )
            
        super().__init__(**dict(kwargs, ports=ports))
        
        # pop the kwargs consumed by rfBase (self.kwused is initialized in rfBase)
        for kw in self.kwused:
            kwargs.pop(kw, None)
            
        self.f0Hz = kwargs.pop('f0Hz', None)
        self.Lc = kwargs.pop('Lc', None)
        self.kcdB = kwargs.pop('kcdB', None)  
        self.kc = kwargs.pop('kc', None)
        
        if 'sNp' in kwargs:
            # sNp = kwargs.pop('sNp', None)
            raise NotImplementedError(
                f'{whoami(__package__)}: sNp not yet implemented' # @UndefinedVariable
            )

        if self.kc:
            if self.kcdB:
                raise ValueError(
                    f'{whoami(__package__)}: only one of kc or kcdB can be given' # @UndefinedVariable
                )
            else:
                self.kcdB = 20 * np.log10(self.kc)
        else:
            if self.kcdB:
                self.kc = 10**(-np.abs(self.kcdB)/20)
            else:
                self.kc = np.sqrt(0.5)
                self.kcdB = -20 * np.log10(self.kc)
                
        if self.Lc:
            if self.f0Hz:
                raise ValueError(
                    f'{whoami(__package__)}: only one of Lc or f0Hz can be given' # @UndefinedVariable
                )
            self.f0Hz = c0/(4*self.Lc)
        else:
            if self.f0Hz:                
                self.Lc = c0/(4*self.f0Hz)
            
        _debug_ and logident('<')
            
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, full=0):
        spl = lambda kw: ('+' if kw in self.kwargs else '') + kw
        s = super().__str__()
        s += f'\n| {spl("kc")}= {self.kc:.4f};'
        s += f' {spl("kcdB")}= {self.kcdB:7.2f} dB;'
        if self.Lc or self.f0Hz:
            s += f' {spl("f0Hz")}= {self.f0Hz:_.1f} Hz;'
            s += f' {spl("Lc")}= {self.Lc:7.3f} m'
        else:
            s += f' Lc is quarter wave at any requested frequency' 
            
        if full:
            s += '\n'
            lmax = max([len(p) for p in self.ports])
            q = "'"
            for p, xpos in zip(self.ports, self.xpos):
                s += f'| {q+p+q:{lmax+2}s} @ {xpos:7.3f}m '+ '\n'
            s = s[:-1]
        s += '\n^'
        return s
        
    #===========================================================================
    #
    # g e t 1 S 
    #
    def get1S(self, f):
        
        _debug_ = logit['DEBUG']
        _debug_ and logident('>')
        
        blc = 2*np.pi*f / c0 * (self.Lc if self.Lc else c0/(4*f))
        cosblc, sinblc = np.cos(blc), np.sin(blc)
    
        s1k2 = np.sqrt(1 - self.kc**2)
    
        N = s1k2 * cosblc + 1j*sinblc
        A, B = s1k2 / N, 1j * self.kc * sinblc / N
      
        _debug_ and logident('<')
        
        return np.array(
            [[0., A, B, 0.],
             [A, 0., 0., B],
             [B, 0., 0., A],
             [0., B, A, 0.]])
          
#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    from Utilities.printMatrices import printMA
    my3dB = rf3dBHybrid(kcdB=80.5)
    print(my3dB.asstr(-1))
    printMA(my3dB.getS(40e6),pfmt='%14.10f %+12.7f' )

################################################################################
#                                                                              #
# Copyright 2018-present                                                       #
#                                                                              #
#                   Laboratory for Plasma Physics                              #
#                   Royal Military Academy                                     #
#                   Brussels, Belgium                                          #
#                                                                              #
#                   ITER Organisation                                          #
#                                                                              #
#                   EUROfusion                                                 #
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

