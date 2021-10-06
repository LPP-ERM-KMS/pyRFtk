"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 30 Mar 2021

@author: frederic
"""
__updated__ = "2021-08-19 16:39:00"

import numpy as np

#TODO: integrate config imports ...

from .config import tLogger, logit
from .config import _newID
from .config import rcparams
from .config import fscale
from .config import FUNITS

#===============================================================================
#
# r f R L C
#
class rfRLC():
    """rfRLC
    
    (s) -- Cs -- Ls -- Rs --+-- (p)
                            |
                        +---+---+
                        |   |   |
                        Cp  Lp  Rp
                        |   |   |
                        +---+---+
                            |
                           Gnd
    kwargs:
        Zbase : reference impedance [50 Ohm]
        ports : port names [['s','p']]
        Rp : parallel resistance [+inf Ohm]
        Lp : parallel inductance [+inf H]
        Cp : parallel capacity [0 F]
        Rs : series resistance [0 Ohm]
        Ls : series inductance [0 H]
        Cs : series capacity [+inf F]  
        
        thus the default is :  (s) -- (p)
        
    """
    
    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, **kwargs):
        """kwargs:
            Zbase : reference impedance [50 Ohm]
            ports : port names [['s','p']]
            Rp : parallel resistance [+inf Ohm]
            Lp : parallel inductance [+inf H]
            Cp : parallel capacity [0 F]
            Rs : series resistance [0 Ohm]
            Ls : series inductance [0 H]
            Cs : series capacity [+inf F]
        """
        self.Zbase = kwargs.pop('Zbase',50.)
        self.ports = kwargs.pop('ports',['s','p'])
        self.Rs = kwargs.pop('Rs', 0.)
        self.Ls = kwargs.pop('Ls', 0.)
        self.Cs = kwargs.pop('Cs', +np.inf)
        self.Rp = kwargs.pop('Rp', +np.inf)
        self.Lp = kwargs.pop('Lp', +np.inf)
        self.Cp = kwargs.pop('Cp', 0.)
        self.f = None
        self.S = np.array([[0,1],[1,0]], dtype=complex)
    
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, full=False):
        return 'lumped RLC element\n^'
    
    #===========================================================================
    #
    # _ _ l e n _ _
    #
    def __len__(self):
        return len(self.S)
    
    #===========================================================================
    #
    # s e t
    #
    def set(self, **kwargs):
        
        for kw, val in kwargs.items():
            if not hasattr(self, kw):
                raise ValueError(f'rfTRL.set: parameter {kw} not present')
            
            setattr(self, kw, val)
            self.kwargs[kw] = val
        
        self.solved = {}
    
    #===========================================================================
    #
    # g e t S
    #
    def getS(self, fs, Zbase=None, params={}):
        
        if Zbase is None:
            Zbase = self.Zbase
        
        for p in ['Rs','Ls','Cs','Rp','Lp','Cp']:
            setattr(self, p, params.pop(p, getattr(self, p)))
            
        def get1S(f):
            self.f = f
            
            if f != 0:
                jw = 2j * np.pi * f
                zs = (self.Rs + jw * self.Ls + 1/jw/self.Cs)/Zbase
                yp = (1/self.Rp + 1/jw/self.Lp + jw * self.Cp)*Zbase
                
                self.S = np.linalg.inv(
                            [[     1   , 1 + yp],
                             [-(1 + zs),   1   ]]
                        ) @ np.array(
                            [[     1   , 1 - yp],
                             [  1 - zs ,  -1   ]])
            else:
                if self.Cs == +np.inf:
                    zs = self.Rs
                    if self.Lp == +np.inf:
                        # (s) -- Rs --+-- (p)
                        #             |
                        #             Rp
                        #             |
                        #            Gnd
                        yp = 1/self.Rp
                        self.S = np.linalg.inv(
                                [[     1   , 1 + yp],
                                 [-(1 + zs),   1   ]]
                            ) @ np.array(
                                [[     1   , 1 - yp],
                                 [  1 - zs ,  -1   ]])
                    else:
                        # (s) -- Rs --+-- (p)
                        #             |
                        #            Gnd
                        self.S = np.array(
                            [[ (self.Rs-Zbase)/(self.Rs + Zbase) , 0j ],
                             [               0j                  , -1 ]],
                            dtype=np.complex)                   
                else:
                    if self.Lp == +np.inf:
                        # (s) -- OO --+-- (p)
                        #             |
                        #             Rp
                        #             |
                        #            Gnd
                        if self.Rp != +np.inf:
                            rcp = (self.Rp - Zbase) / (self.Rp + Zbase)
                        else:
                            rcp = +1
                        self.S = np.array(
                            [[1, 0j],[0j, rcp ]],
                            dtype=np.complex)
                    else:
                        # (s) -- OO --+-- (p)
                        #             |
                        #            Gnd
                        self.S = np.array([[1, 0j], [0j, -1]], dtype=np.complex)
            return self.S
        
        Ss = []
        if hasattr(fs,'__iter__'):
            for f in fs:
                Ss.append(get1S(f))
        else:
            Ss = get1S(fs)
            
        return Ss

    #===========================================================================
    #
    # m a x V 
    #
    def maxV(self, f, E, Zbase=None, ID='<rfRLC>'):
        
        if Zbase is None:
            Zbase = self.Zbase
        
        raise NotImplementedError
        return 
        
    