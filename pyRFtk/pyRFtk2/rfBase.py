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

__updated__ = '2022-03-23 08:58:14'

"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 13 Feb 2022

@author: frederic
"""
if __name__ == '__main__':
    import sys
    print('running the test module "../pyRFtk2 test/pyRFtk2_tests.py"')
    sys.path.insert(0, '../pyRFtk2 test')
    import pyRFtk2_tests                       # @UnresolvedImport @UnusedImport
    sys.exit(0)
    
degSign = u'\N{DEGREE SIGN}'
OHM = u'\N{Greek Capital Letter Omega}'

import numpy as np
import copy
   
from .config import logident, logit, _newID
from .Utilities import whoami, strM
from .CommonLib import ConvertGeneral, _check_3D_shape_
        
#===============================================================================
#
# a d d _ d e b u g _ c o d e 
#
def add_debug_code(func):
    def funccaller(*args, **kwargs):
        debug = logit['DEBUG']
        debug and logident(f'> {func.__name__}',stacklev=2)
        retvals = func(*args, **kwargs)
        debug and logident('<', stacklev=2)
        return retvals
    return funccaller

#===============================================================================
#
# r f B a s e
#
class rfBase():
    """rfBase is the parent class of all RF objects
    
        __init__ implements the setting of Id, Zbase, portnames
        copy, __copy__, __deepcopy__
        __getstate__, __setstate__
        __str__
        __len__
    """
    
    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, **kwargs):
        """rfBase(
            [Id=],    # Identifier; default type(self).__name__(::ALFANUM::){6} 
            [Ss=],    # the S-matrix; default to np.zeros((0, len(ports), len(ports)))
            [fs=],    # the frequencies associated with S; default np.zeros((Ss.shape[0],))
            [ports=], # portnames; default to [f'{k}' for k in range(1,len(S)+1)]
            [Zbase=]  # reference impedance; default to 50 [Ohm]
            [xpos=]   # port positions; default [ 0. for k in range(len(self)]
           )
        """
        
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', printargs=True)
        
        self.kwargs = kwargs.copy()
        self.Id = kwargs.pop('Id', f'{type(self).__name__}_{_newID()}')
        self.Zbase = kwargs.pop('Zbase', 50.)
        
        if 'Ss' in kwargs:
            _, self.Ss = _check_3D_shape_(np.array(kwargs.pop('Ss'), dtype=complex)) # Ss is there
            _debug_ and logident(f'self.Ss.shape={self.Ss.shape}')
            _debug_ and logident(f'self.Ss={self.Ss}')
        
            if 'ports' in kwargs:
                self.ports = kwargs.pop('ports')
                if len(self.ports) != self.Ss.shape[1]:
                    raise ValueError(
                        'pyRFtk2.rfBase: inconsitent "ports" and "S" given'
                    )
            else:
                self.ports = [f'{_}' for _ in range(1,self.Ss.shape[1]+1)]
            
        
        elif 'ports' in kwargs:
            self.ports = kwargs.pop('ports')
            self.Ss = 1j*np.zeros((0, len(self.ports),len(self.ports)))
        
        else:
            self.ports = []
            self.Ss = 1j*np.zeros((0, 0, 0))
        
        self.fs = np.array(kwargs.pop('fs', range(self.Ss.shape[0])), dtype=float)
        _debug_ and logident(f'self.fs={self.fs}')
        if self.Ss.shape[0] != self.fs.shape[0]:
            raise ValueError(
                f'size mismatch between supplied fs [{self.fs.shape[0]}] and '
                f'Ss [{self.Ss.shape[0]}].'
            )
        
        self.xpos = kwargs.pop('xpos', [0.] * len(self))
        self.f, self.S = None, None
        
        _debug_ and logident('<')
        
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, full =0):
        s = f'{type(self).__name__} Id="{self.Id}" @ Zbase={self.Zbase:.2f} Ohm [{hex(id(self))}]'
        if full:
            s += '\n'
            if len(self):
                lmax = max([len(p) for p in self.ports])
                q = "'"
                for p, xpos in zip(self.ports, self.xpos):
                    s += f'| {q+p+q:{lmax+2}s} @ {xpos:7.3f}m '+ '\n'
                
                if len(self.Ss):
                    s += (f'| {len(self.fs)} frequencies '
                          f'from {np.min(self.fs)} to  {np.max(self.fs)} Hz\n')
                    try:
                        if self.f is not None and self.S is not None:
                            s  += f'+ last evaluation @ {self.f:_.1f}Hz\n'
                            for aline in strM(
                                    self.S, pfmt='%8.6f %+6.1f'+degSign, 
                                    pfun=lambda z: (np.abs(z), np.angle(z,deg=1))
                                    ).split('\n'):
                                s += '| ' + aline + '\n'
                    except ValueError:
                        s  += f'+ last evaluation @ {self.f}Hz [{type(self.f)}](ValueError on self.f)\n'
            else:
                s += '| <empty>\n'
            s += '^'
        return s
    
    #===========================================================================
    #
    # _ _ l e n _ _
    #
    def __len__(self):
        return len(self.ports)
    
    #===========================================================================
    #
    # c o p y ,  _ _ c o p y  _ _ ,   _ _ d e e p c o p y _ _
    #
    def copy(self):
        return self.__deepcopy__(self)
    
    def __copy__(self):
        return self.__deepcopy__(self)
    
    def __deepcopy__(self, memo=None):                         # @UnusedVariable
        other = type(self)()
        for attr, val in self.__dict__.items():
            try:
                # print(f'copying {attr}')
                other.__dict__[attr] = copy.deepcopy(val)
            except:
                print(f'{whoami(__package__)}: could not deepcopy {attr}')
                raise
        
        # change the Id
        other.Id += '(copy)'
        return other
    
    #===========================================================================
    #
    # g e t 1 S 
    #
    def get1S(self, f):
        _debug_ = logit['DEBUG'] 
        _debug_ and logident('>', printargs=True)
        
        _debug_ and logident(f'shape self.Ss={self.Ss.shape}')
        
        k = np.where(self.fs == f)[0]
        _debug_ and logident(f'k={k}')
        
        if len(k) == 0:
            raise ValueError(
                f'{whoami(__package__)}: f={f}Hz is not available'
            )
        
        _debug_ and logident(f'self.Ss={self.Ss}')
        S = self.Ss[k[0],:,:]
        _debug_ and logident(f'S[k]={S}')
        self.f, self.S = f, S
        
        _debug_ and logident('<')
        return S
    
    #===========================================================================
    #
    # g e t S 
    #
    def getS(self, fs=None, **kwargs):
        _debug_ = logit['DEBUG'] 
        _debug_ and logident('>', printargs=True)

        fs = self.fs if fs is None else fs
        _debug_ and logident(f'fs={fs}')
        
        if hasattr(fs,'__iter__'):
            Ss = [self.get1S(fk) for fk in fs]
            
            Zbase = kwargs.pop('Zbase', self.Zbase)
            R = ConvertGeneral(Zbase, Ss, self.Zbase) if Zbase != self.Zbase else Ss

        else:
            R = self.getS([fs], **kwargs)[0]
        
        _debug_ and logident('<')
        return R

    #===========================================================================
    #
    # m a x V
    #
    def maxV(self, f, E, Zbase=None, Id=None, xpos=0., **kwargs):
        """
        """
        _debug_ = logit['DEBUG']
        _debug_ and logident('>')
        
        Id = Id if Id else self.Id
        
        unknown = [p for p in E if p not in self.ports]
        if unknown:
            msg = (f'Unknown port{"s" if len(unknown) > 1 else ""} '
                   f'{", ".join(unknown)} for {self.Id}')
            _debug_ and logident(msg)
            raise ValueError(f'{whoami()}: {msg}')
        

        if isinstance(xpos, (int, float)):
            if hasattr(self, 'xpos'):
                xpos = list(xpos + np.array(self.xpos))
            else:
                xpos = [xpos] * len(self)
        
        elif hasattr(xpos, '__iter__') and len(xpos) == len(self):
            pass
        
        else:
            msg = (f' could not understand supplied xpos: {type(xpos)} '
                   f'{("[%d]"%len(xpos)) if hasattr(xpos,"__iter__") else ""}')
            _debug_ and logident(msg)
            raise ValueError(f'{whoami(__package__)}: {msg}')

        Zbase = Zbase if Zbase else self.Zbase
        S = self.getS(f, Zbase=Zbase, **kwargs)
        A = [E[p] if p in E else 0. for p in self.ports]
        B = S @ A
        absV = np.abs(A + B) 
        Vmax = np.max(absV)
        where = self.ports[list(np.where(absV == Vmax)[0])[0]]
        VSWs = {
            Id: dict( (p, Vp) for p, Vp in zip(self.ports, absV))
        }
        
        return Vmax, where, VSWs
