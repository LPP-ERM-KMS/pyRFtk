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

__updated__ = '2020-10-19 14:49:36'

"""
Created on 8 Apr 2020

@author: frederic

2020-04-08 rewrite of the (in)famous pyRFtk package

The rfObject is functionally a merger of scatter3a and TouchstoneClass3a
"""

import numpy as np
from scipy.interpolate import interp1d
import re

from pprint import pprint

from Utilities.printMatrices import printR, printM, printMA
from Utilities.tictoc import tic, toc
from Utilities.getlines import getlines

from pyRFtk2.CommonLib import read_tsf, convert_general, S_from_Y, S_from_Z

#===============================================================================
#
#    setup logging
#
import logging
logging.basicConfig(
    level=logging.DEBUG, 
    filename='rfObject.log',
    filemode = 'w',
    format='%(asctime)s - %(levelname)8s - %(message)s'
)
logging.info('[rfObject] ---------------------------------------------------- ')
logging.info('[rfObject] startlogging -- level %s')

#===============================================================================

rcparams = {
    'Zbase'  : 50,       # Ohm
    'funit'  : 'MHz',    # 'Hz' | 'kHz' | 'MHz' | 'GHz'
    'interp' : 'MA',     # 'MA' | 'RI' | 'dB'
    'interpkws' : {'kind': 3, 'fill_value':'extrapolate'},
}

FUNITS = {'HZ':1., 'KHZ':1e3, 'MHZ':1e6, 'GHZ':1e9}

ID = 0

TYPE_GENERIC     = 1
TYPE_CIRCUIT     = 2
TYPE_TOUCHSTONE  = 3
TYPE_GTL         = 4

TYPES = {
    'GENERIC'    : TYPE_GENERIC,
    'CIRCUIT'    : TYPE_CIRCUIT,
    'TOUCHSTONE' : TYPE_TOUCHSTONE,
    'GTL'        : TYPE_GTL
}

TYPE_STR = {
    TYPE_GENERIC    : 'GENERIC',
    TYPE_CIRCUIT    : 'CIRCUIT',
    TYPE_TOUCHSTONE : 'TOUCHSTONE',
    TYPE_GTL        : 'GTL'
}

#===============================================================================
#
#  r f O b j e c t
#
class rfObject():
    """rfObject is the master class
    
            not sure if this a replacement for TouchStoneClass or already
            includes scatter3a ... or even something else ...
            
    methods:
        getS(fMHz, **kwargs) : kwargs could be parameters determining the object
                for nested objects (circuits) it would be called
                
        getS(frequency, {sub_element_ID:{
                            parameter1: value1,
                            parameter2: value2,
                            sub_sub_element_ID: {
                            
                            }
                         }
                        }
            
    """
    #===========================================================================
    #
    #  _ _ i n i t _ _
    #
    def __init__(self, **kwargs):
        """rfObject
        
        'kwargs'
    
            'ports'      : list of unique strings
            'name'       : string (TBD)
            'Zbase'      : default rcparams['Zbase']
            'funit'      : default rcparmas['funit']
            'interp'     : default rcparams['interp']
            'interpkws'  : default rcparams['interpkws']
            'flim'       : default funit/1000, funit*1000
            'touchstone' : default None -- path to a touchstone file
        """
        
        global ID
             
        self.kwargs = kwargs.copy()

        #TODO: validate allowed port names
        self.Portnames = kwargs.get('ports',[])
        self.alias = {}
        if isinstance(self.Portnames, str):
            self.Portnames = self.Portnames.split()
        self.N = len(self.Portnames)
        
        self.Zbase = kwargs.get('Zbase', rcparams['Zbase'])
        self.funit = kwargs.get('funit', rcparams['funit'])
        self.fscale = FUNITS[self.funit.upper()]
        self.flim = kwargs.get('flim', [0.001, 1000.])
        
        self.interp = kwargs.get('interp',rcparams['interp'])
        self.interpkws = kwargs.get('interpkws',rcparams['interpkws'])
        
        self.touchstone = kwargs.get('touchstone', None)
        if self.touchstone:
            Tkwargs = dict([
                (kw, value) for kw, value in [
                    ('Zbase', self.Zbase),
                    ('funit', self.funit),
                    ('ports', self.Portnames)] if kw in kwargs
            ])
            self.read_tsf(self.touchstone, **Tkwargs)
        
        self.interpOK = False
        
        if 'name' in kwargs:
            self.name = kwargs.get('name')
        else:
            ID += 1
            self.name = '%08d' % ID
        
        self.fs = np.array([],dtype=float).reshape((0,))
        self.sorted = True
        self.Ss = np.array([], dtype=np.complex).reshape((0, self.N, self.N))
        
        self.process_kwargs()
        self.type = TYPE_GENERIC
    
    #===========================================================================
    #
    #  _ _ g e t s t a t e _ _
    #
    def __getstate__(self):
        
        # default
        d = self.__dict__.copy()
        
        # we cannot pickle the interp1d functions
        d.update(interpOK=False, interpA=None, interpM=None)
        
        # get to basic python types int, float, complex, bool, dict, list, tupple
        d['fs'] = self.fs.tolist()
        d['Ss'] = self.Ss.tolist()
        
        return d
    
    #===========================================================================
    #
    #  _ _ s e t s t a t e _ _
    #
    def __setstate__(self, d):
        
        for var, value in d.items():
            if var not in ['fs', 'Ss']:
                self.__dict__[var] = value
        
        # get the arrays back
        self.fs = np.array(d['fs'])
        self.Ss = np.array(d['Ss'])

    #===========================================================================
    #
    #  _ _ r e p r _ _
    #
    def __repr__(self):
        s = '%s( # %r \n' % (type(self).__name__, self.name)
        s += '  Portnames=[   # %d \n' % len(self.Portnames)
        for p in self.Portnames:
            s += '    %r,\n' % p
        s += '  ]'
        for kw, val in self.kwargs.items():
            s += ',\n'
            s += '  %s = %r' % (kw,val)
        s += '\n)'
        return s
    
    #===========================================================================
    #
    #  _ _ s t r _ _
    #
    def __str__(self):
        s =  self.__repr__() + '\n'
        s += '# %d frequencies\n' % len(self.fs)
        return s
    
    #===========================================================================
    #
    #  _ _ l e n _ _
    #
    def __len__(self):
        return len(self.fs)
    
    #===========================================================================
    #
    #  _ _ i t e r _ _
    #
    def __iter__(self):
        for k in range(len(self)):
            yield self.fs[k], self.Ss[k,:,:]
    
    #===========================================================================
    #
    #  _ _ g e t i t e m _ _
    #
    def __getitem__(self, k):
        return self.fs[k], self.Ss[k,:,:]
    
    #===========================================================================
    #
    #  _ _ s e t i t e m _ _
    #
    def __setitem__(self, k, value):
        self.fs[k] = value[0]
        self.Ss[k,:,:] = value[1]
    
    #===========================================================================
    #
    #  _ _ c o p y _ _
    #
    def __copy__(self):
        
        new_self = type(self)(**self.kwargs)
        new_self.sorted = self.sorted
        new_self.fs = self.fs.copy()
        new_self.Ss = self.Ss.copy()
        new_self.interpOK = False    # avoid to copy interpolation functions
        new_self.type = self.type
        
        return new_self
    
    #===========================================================================
    #
    #  _ _ e q _ _
    #
    def __eq__(self, other, verbose=False):
        
        diflist = []
        otherval = [val for val in other.__dict__]
        for val in self.__dict__:
            if val not in otherval:
                diflist.append((val,'<'))
            
            else:
                otherval.pop(otherval.index(val))
                
                if val in ['fs', 'Ss']:
                    if not (len(self.fs) == len(other.fs)
                        and np.allclose(self.__dict__[val], other.__dict__[val])):
                        diflist.append((val,'!='))
                        
                else:
                    if self.__dict__[val] != other.__dict__[val]:
                        diflist.append((val,'!='))
                        
        if otherval:
            diflist += [(val,'>') for val in otherval]
            
        if verbose:
            for val, dif in diflist:
                print('%3s %s' % (dif, val))

        return len(diflist) == 0
    
    #===========================================================================
    #
    #  s o r t p o r t s
    #
    def sortports(self, order=str.__lt__):
        
        if isinstance(order,list):
            idxs = [self.Portnames.index(p1) for p1 in order]
            idxs += [k for k, p1 in enumerate(self.Portnames) if k not in idxs]
            if len(idxs) != len(self.Portnames):
                raise ValueError(
                    'rfObject.sort: either a bug or duplicate ports in order list')
        else:
            idxs = np.array(self.Portnames).argsort()
        
        self.Portnames = [self.Portnames[k] for k in idxs]
        self.Ss = self.Ss[:,:,idxs][:,idxs,:]
    
    #===========================================================================
    #
    #  _ s o r t _ f s _
    #
    def _sort_fs_(self):
        
        if not self.sorted:
            
            idxs = self.fs.argsort()
            self.fs = self.fs[idxs]
            self.Ss = self.Ss[idxs,:,:]
            
            self.sorted = True
        
    #===========================================================================
    #
    #  _ i n t e r p _
    #
    def _interp_(self, fs, insert=False):    #TODO: find a better name
        
        if not (fs is None or hasattr(fs, '__iter__')):
            fs = [fs]
        
        if not self.interpOK:
            
            if not self.sorted:
                self._sort_fs_()
            
            if self.interp == 'MA':
                M = np.abs(self.Ss)
                A = np.unwrap(np.angle(self.Ss),axis=0)
                
            elif self.interp == 'dB':
                L = np.log(self.Ss)
                M = np.real(L)
                A = np.unwrap(np.imag(L), axis=0)
                
            else: # RI
                M = np.real(self.Ss)
                A = np.imag(self.Ss)
                
            self.interpM = interp1d(self.fs, M, axis=0, **self.interpkws)
            self.interpA = interp1d(self.fs, A, axis=0, **self.interpkws)
                            
            self.interpOK = True
            
        if not fs:
            return
        
        M, A = self.interpM(fs), self.interpA(fs)
    
        if self.interp == 'MA':
            Ss = M * np.exp(1j * A)
        
        elif self.interp == 'dB':
            Ss = np.exp(M + 1j * A)
        
        else: #RI
            Ss =  M + 1j * A
            
        if insert:
            self.setS(fs,Ss)
            self._sort_fs_()
            
        return Ss
    
    #===========================================================================
    #
    #  c o n v e r t 2 n e w Z b a s e
    #
    def convert_Zbase(self, newZbase):
        """convert_Zbase
        
            converts the rfObject's Ss and Zbase to newZbase
            
            newZbase: target reference impedance 
        """
        
        self.Ss = convert_general(newZbase, self.Ss, self.Zbase, 'V', 'V')
        self.Zbase = newZbase
    
    #===========================================================================
    #
    #  g e t S
    #
    def getS(self, fs, *kw):
        
        #FIXME: this will not work if fs is not a numpy.array
        
        fs *= FUNITS[kw.pop('funit', self.funit).upper()]/self.fscale
        Zbase = kw.pop('Zbase',self.Zbase)
        
        Ss = []
        for f in fs:
            tS = self._get1S_(f)
            Ss.append(tS)
        return np.array(Ss)
    
    #===========================================================================
    #
    #  s e t S
    #
    def setS(self, f, S, Portnames=None):
        
        #TODO: possibly 
        
        self.interpOK = False
        self.interpA = None
        self.interpM = None
        
        S =  np.array(S)
        
        if S.shape[-1] != S.shape[-2]:
            raise ValueError(
                'rfObject.setS: S does not appear to be a (list of) square matrice(s)')
        
        if  (len(S.shape)==3 
             and S.shape[0] != (len(f) if hasattr(f,'__iter__') else 1)
            ):
            # we have a list of 2D arrays ... and its length is compatible with f
            pass # what did we want to do
        
        if len(self.fs) == 0:
            # the object has no frequency entries yet ...
            self.Ss = self.Ss.reshape((0,)+S.shape[-2])
            if Portnames and len(Portnames) == S.shape[-1]:
                self.Portnames = Portnames[:]
            else:
                self.Portnames = ['%02d' % k for k in range(S.shape[-1])]
            
        if self.Ss.shape[-2:] != S.shape[-2:]:
            raise ValueError('rfOject.setS Shape of S does not match ')
        
        if len(S.shape) < 3:
            S = S.reshape((1, self.N, self.N))
        
        self.fs = np.append(self.fs,[f])
        self.Ss = np.append(self.Ss, S, axis=0)
        
        # do not sort each time
        self.sorted = self.sorted and (
            (len(self.fs) is 1) or (self.fs[-2] < self.fs[-1])
        )
        
    #===========================================================================
    #
    #  e x c i t e
    #
    def excite(self, a0s, f):
        pass
    
    #===========================================================================
    #
    #  p r o c e s s _ k w a r g s
    #
    def process_kwargs(self):
        pass
    
    #===========================================================================
    #
    #  r e a d _ t s f 
    #
    def read_tsf(self, src, **kwargs):
        r = read_tsf(src, **kwargs)
        self.fs = r['fs']
        self.funit = r['funit']
        self.fscale = ...
        self.Ss = r['Ss']
        self.Portnames = r['ports']
        self.Zbase = r['Zbase']
        self.part = r['part']
        self.numbering = r['numbering']
        self.markers = r['markers']
        self.shape = r['shape']
        
    #===========================================================================
    #
    # S _ f r o m _ Z 
    #
    def S_from_Z(self,Z):
        """
        returns the scattering matrix S defined on Zbase [Ohm] from an 
        impedance matrix Z
        """
        return S_from_Z(Z, Zbase=self.Zbase)
    
    #===========================================================================
    #
    # S _ f r o m _ Y 
    #
    def S_from_Y(self,Y):
        """
        returns the scattering matrix S defined on Zbase [Ohm] from an 
        impedance matrix Z
        """
        return S_from_Y(Y, Zbase=self.Zbase)
    

#===============================================================================
#        
#  _ _ m a i n _ _
#

if __name__ == '__main__':
    
    from pyRFtk import TouchStoneClass3a as _ts
    from matplotlib import pyplot as pl
    from pyRFtk2 import tests_rfObject as t
    
    tests = {
        
        '__init__': (
            t.test_create, 
            ('a b c', [20, 30, 40]), 
            {'name':'hello'}
        ),
        
        '__copy__': (
            t.test_copy, 
            (), 
            {}
        ),

#         '__getitem__': (
#             t.test_getitem, 
#             (), 
#             {}
#         ),
        
#         '__iter__': (
#             t.iterator, 
#             (), 
#             {}
#         ),

#         'setS': (
#             t.test_setS, 
#             (), 
#             {}
#         ),

#         'sortports': (
#             t.sortports, 
#             (), 
#             {}
#         ),

#         '__getstate__': (
#             test_pickle, 
#             ('a b', [30, 40, 50, 60], [35, 45, 55]), 
#             {}
#         ),

#         'interpolate': (
#             t.interpolate, 
#             (), 
#             {} 
#         ),
        
#         'test_read_tsf_order': (
#             t.test_read_tsf, 
#             ('../test data/SRI.s6p',), 
#             {} 
#         ),
           
        'test_read_tsf_mixed': (
            t.test_read_tsf, 
            ('../test data/StrapOrientation_v5p0_7_reference_R2_HFSSDesign1.s24p',), 
            {} 
        ),
           
#         'test_read_tsf_fixed': (
#             t.test_read_tsf, 
#             ('../test data/Z_IterCY8vr2R_Low-4cm_RefTOP.s24p',), 
#             {} 
#         ) ,
    }
    
    for test, (f, args, kwargs) in tests.items():
        t0=tic()
        f(*args, **kwargs)
        print('\nElapsed time %.3f ms\n' % (toc(t0)*1000))

    print('legacy TouchstoneClass3a')
    t0 = tic()
    ts = _ts.TouchStone(filepath= '../test data/StrapOrientation_v5p0_7_reference_R2_HFSSDesign1.s24p')
    print('\nElapsed time %.3f ms\n' % (toc(t0)*1000))
    
    b = rfObject()
    b.read_tsf('../test data/StrapOrientation_v5p0_7_reference_R2_HFSSDesign1.s24p')
    print(b)
    
    pl.figure()
    fMHzs = ts.Freqs('MHz')
    i, j = 20, 15
    s11s = ts.Datas(mtype='S',elfmt='RI',elems=[(i,j)],zref=50.0)
    pl.plot(fMHzs, np.abs(s11s),'r')
    pl.plot(b.fs, np.abs(b.Ss[:,i,j]),'b')
    pl.show()
