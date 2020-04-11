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

__updated__ = '2020-04-10 23:48:04'

"""
Created on 8 Apr 2020

@author: frederic

2020-04-08 rewrite of the (in)famous pyRFtk package
"""

import numpy as np
from scipy.interpolate import interp1d

#===============================================================================

params = {
    'Zbase'  : 50,       # Ohm
    'funit'  : 'MHz',    # 'Hz' | 'kHz' | 'MHz' | 'GHz'
    'interp' : 'MA',     # 'MA' | 'RI' | 'dB'
    'interpkws' : {'kind': 3, 'fill_value':'extrapolate'},
}

FUNITS = {'HZ':1., 'KHZ':1e3, 'MHZ':1e6, 'GHZ':1e9}

ID = 0

#===============================================================================
#
#  r f O b j e c t
#
class rfObject():
    """rfObject is the master class
    """
    #===========================================================================
    #
    #  _ _ i n i t _ _
    #
    def __init__(self, Portnames, **kwargs):
        """rfObject
        
            'Portnames' : list of unique strings
        
            'kwargs'
        
                'name'      : string (TBD)
                'Zbase'     : default params['Zbase']
                'funit'     : default parmas['funit']
                'interp'    : default params['interp']
                'interpkws' : default params['interpkws']
        """
        global ID
        
        self.Portnames = Portnames
        self.N = len(self.Portnames)
        
        self.kwargs = kwargs.copy()

        self.Zbase = kwargs.get('Zbase', params['Zbase'])
        self.funit = kwargs.get('funit', params['funit'])
        self.fscale = FUNITS[self.funit.upper()]
        
        self.interp = kwargs.get('interp',params['interp'])
        self.interpkws = kwargs.get('interpkws',params['interpkws'])
        self.interpOK = False
        
        if 'name' in kwargs:
            self.name = kwargs.get('name')
        else:
            ID += 1
            self.name = '%08d' % ID
        
        self.kwargs = kwargs.copy()

        self.fs = np.array([],dtype=float).reshape((0,))
        self.sorted = True
        self.Ss = np.array([], dtype=np.complex).reshape((self.N, self.N, 0))
        
        self.process_kwargs()
        self.type = 'generic'
    
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
            yield self.fs[k], self.Ss[:,:,k]
    
    #===========================================================================
    #
    #  _ _ g e t i t e m _ _
    #
    def __getitem__(self, k):
        return self.fs[k], self.Ss[:,:,k]
    
    #===========================================================================
    #
    #  _ _ c o p y _ _
    #
    def __copy__(self):
        
        new_self = type(self)(self.Portnames, **self.kwargs)
        new_self.sorted = self.sorted
        new_self.fs = self.fs.copy()
        new_self.Ss = self.Ss.copy()
        new_self.interpOK = False    # avoid to copy interpolation functions
        new_self.type = self.type
        
        return new_self
    
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
        self.Ss = self.Ss[:,idxs,:][idxs,:,:]
    
    #===========================================================================
    #
    #  _ s o r t _ f s _
    #
    def _sort_fs_(self):
        
        if not self.sorted:
            
            idxs = self.fs.argsort()
            self.fs = self.fs[idxs]
            self.Ss = self.Ss[:,:,idxs]
            
            self.sorted = True
        
    #===========================================================================
    #
    #  _ i n t e r p _
    #
    def _interp_(self, fs):    #TODO: find a better name
        
        if not self.interpOK:
            
            if self.interp == 'MA':
                M = np.abs(self.Ss)
                A = np.unwrap(np.angle(self.Ss))
                
            elif self.interp == 'dB':
                L = np.log(self.Ss)
                M = np.real(L)
                A = np.unwrap(np.imag(L))
                
            else: # RI
                M = np.real(self.Ss)
                A = np.imag(self.Ss)
                
            self.interpM = interp1d(self.fs, M, **self.interpkws)
            self.interpA = interp1d(self.fs, A, **self.interpkws)
                            
            self.interpOK = True
        
        M, A = self.interpM(fs), self.interpA(fs)
        
        if self.interp == 'MA':
            return M * np.exp(1j * A)
        
        elif self.interp == 'dB':
            return np.exp(M + 1j * A)
        
        else: #RI
            return M + 1j * A
    
    #===========================================================================
    #
    #  c o n v e r t 2 n e w Z b a s e
    #
    def convert2newZbase(self, Ss, newZbase, **kwargs):
        
        return Ss
    
    #===========================================================================
    #
    #  c o n v e r t 2 Z b a s e
    #
    def convert2Zbase(self, Ss, curZbase, **kwargs):
        
        return Ss
    
    #===========================================================================
    #
    #  g e t S
    #
    def getS(self, fs, *kw):
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
    def setS(self, f, S):
        
        self.interpOK = False
        S =  np.array(S)
        
        if self.Ss.shape[:2] != S.shape[:2]:
            raise ValueError('rfOject.setS Shape of S does not match ')
        
        if len(S.shape) < 3:
            S = S.reshape((self.N, self.N, 1))
        
        self.fs = np.append(self.fs,[f])
        self.Ss = np.append(self.Ss, S, axis=2)
        
        self.sorted = (len(self.fs) is 1) or (self.fs[-2] < self.fs[-1])
    
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
    
#===============================================================================
#        
#  _ _ m a i n _ _
#

if __name__ == '__main__':
    
    from Utilities.printMatrices import strM

    tests = [
        '__init__',
        '__copy__',
        '__getitem__',
        '__iter__',
        'setS',
        'sortports',
    ]  
    
    def fill(a, fs):
        S = np.array([[i/10 + j/100 for j in range(1,b.N+1)] 
                      for i in range(1,b.N+1)], dtype=np.complex)
        for k, f in enumerate(fs):
            a.setS(f, S + k)
        
    def show(a):
        for f, S in a:
            print('%7.3f %s    ' % (f, a.funit), end='')
            print(strM(S))
            print()
    
    if '__init__' in tests:
        print('# -----------------------------------------------------------')
        print('# __init__')
        print('# -----------------------------------------------------------')
        
        b = rfObject(Portnames=['d','a'], name='hello')
        fill(b,[20, 
                40, 60])
        print(b)
        print(b.Zbase)
        print(b.funit)
        print(b.Ss.shape)
        
    if 'setS' in tests and False: # if setS does not work then fill does not
        print('# -----------------------------------------------------------')
        print('# setS')
        print('# -----------------------------------------------------------')

        ports = 'a b c'.split()
        b = rfObject(ports)
        fill(b, [30, 35, 40])
        show(b)
        
    if '__iter__' in tests and False: # if __iter__ doesn't work then show fails
        print('# -----------------------------------------------------------')
        print('# __iter__')
        print('# -----------------------------------------------------------')
        ports = 'a b c'.split()
        b = rfObject(ports)
        fill(b, [30, 40, 50])
        show(b)
        
    if 'sortports':
        print('# -----------------------------------------------------------')
        print('# sortports')
        print('# -----------------------------------------------------------')
        
        ports = 'a b c'.split()
        b = rfObject(ports)
        fill(b, [20])
        
        b.sortports(order=['c'])
        print(b.Portnames)
        show(b)

        b.sortports(order='b c a'.split())
        print(b.Portnames)
        show(b)
        
        
