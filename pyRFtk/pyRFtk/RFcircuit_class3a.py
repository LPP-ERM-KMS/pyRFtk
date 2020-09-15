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

"""RFcircuit_class3a

Created on 12 Feb 2020

@author: frederic

The idea is to expand on circuit_class3a allowing nesting of circuits as well as
defining an RFobject class with whose sources can be a set of scatter3a objects,
a touchstone object or a GTL dict (see circuit_class3a.processGTL).

Question: should there be a distiction between RFcirccuit and RFobject


"""

__updated__ = '2020-02-14 13:23:38'

import numpy as np
import matplotlib.pyplot as pl
import copy
        
#===============================================================================
#
# R F c i r c u i t
#
class RFcircuit():
    """RFcircuit class
    
        kwargs:
        
            Zbase: <float> default 50 [Ohm]
        
            freqs: <list of floats> default empty list
        
            fUnit: 'Hz' | 'kHz' | 'MHz' | 'GHz' default 'MHz'
        
    """
    #===========================================================================
    #
    # _ _ i n i t _ _ 
    #
    def __init__(self, **kwargs):
        """constructor methods
        """
        self.kwargs = copy.deepcopy(kwargs)
        
        self.Zbase = kwargs.pop('Zbase', 50.)
        
        self.freqs = kwargs.pop('freqs',np.array([],dtype=float))
        if not hasattr(self.freqs, '__iter__'):
            self.freqs = [self.freqs]
            
        self.fUnit = kwargs.pop('fUnit', 'MHz')
        try:
            self.fScale = {'hz':1, 'khz':1E3, 'mhz':1e6, 'ghz':1e9}[
                self.fUnit.lower()]
        except KeyError:
            raise ValueError('parameter fUnit must be one of "Hz", "kHz", "MHz"'
                             ' or "GHz". Got %r.' % self.fUnit)
            
        if kwargs:
            raise AttributeError('Unknown property/keyword(s)): %r' % 
                                 [kw for kw in kwargs])
        
        self.blocks = [
            # {'name': <str>,
            #  'portnames': ['str'],
            #  'data': RFcircuit |               # has frequencies
            #          TouchStone |              # has frequencies
            #          [ Scatter ]               # has frequencies
            #          {'GTL' : GTL data,
            #           'freqs': [floats],
            #           'fUnit': Hz | kHz | MHz | GHz
            #          },
            #  'location': (row_1, col_1),
            # }
        ]
        self.portnames = []
        
    
    #===========================================================================
    #
    # _ _ r e p r _ _ 
    #
    def __repr__(self):
        s = 'RFCircuit(\n'
        for key in sorted(self.kwargs, key=lambda x: x.lower()):
            s += '  %s = %r,\n' % (key, self.kwargs[key])
        s += ')'
        return s

    #===========================================================================
    #
    # _ _ s t r _ _ 
    #
    def __str__(self):
        return self.__repr__()
        
    #===========================================================================
    #
    # P r o c e s s G T L 
    #
    def ProcessGTL(self, f, GTL):
        """ProcessGTL - process a GTL dict data structure
        
        returns Scatter object
            
        """
        ...
        
    #===========================================================================
    #
    # A d d 
    #
    def Add(self, **kw):
        """Add an RF object to the circuit
        
            GTL: cicuit_Class3a GTL object
        
            SZs: list of scatter3a.Scatter objects
        
            TSF: TouchStone_class.TouchStone object
            
            type of inputs:
        
                GTL
        
                SZs (with signals)
        
                GTL + TSFs
        
                SZs (without signals) + GTL
        """
        ...
        
    #===========================================================================
    #
    # G e t S 
    #
    def GetS(self, f=[], Zbase = None):
        """GetS
        """
        Zbase = self.Zbase if Zbase is None else Zbase
        
#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    print('Hello World !')
    circuit = RFcircuit(Zbase=20, fUnit='kHz', freqs=np.linspace(35,60,51))
    print(circuit)
    