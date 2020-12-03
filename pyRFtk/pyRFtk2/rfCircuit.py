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

__updated__ = '2020-10-19 14:50:07'

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

from pyRFtk2 import rfObject as _RFO
from pyRFtk2 import CommonLib as _RFCL
    
rfObject =  _RFO.rfObject

#===============================================================================
#
# setup logging
#
import logging

logLevel = logging.DEBUG
logslevel = {
    logging.DEBUG    : 'DEBUG',
    logging.INFO     : 'INFO',
    logging.WARNING  : 'WARNING',
    logging.ERROR    : 'ERROR',
    logging.CRITICAL : 'CRITICAL'
}

logger = logging.getLogger('pyRFtk2_rfCircuit.log')
logger.setLevel(logLevel)
logger.info(f'[rfCircuit] -- start logging [{logslevel[logLevel]}]')

#===============================================================================
#
#  r f C i r c u i t
#

class rfCircuit(rfObject):
    
    #===========================================================================
    #
    #  _ _ i n i t _ _
    #
    def __init__(self, **kwargs):
        """rfCircuit
        
            kwargs:
            
            [rfObject]
            
                'ports'      : list of unique strings
                'name'       : string (TBD)
                'Zbase'      : default rcparams['Zbase']
                'funit'      : default rcparmas['funit']
                'interp'     : default rcparams['interp']
                'interpkws'  : default rcparams['interpkws']
                'flim'       : default funit/1000, funit*1000
                'touchstone' : default None -- path to a touchstone file
    
            [rfCircuit]
            
                the touchstone number of ports will need to match the free
                ports of the circuit once defined
                
                TODO: ? circuit validation vs touchstone or silent fail ?
                TODO: ? port renaming between defined circuit and touchstone ? 
        """
        method = '[rfCircuit.__init__]'
        logger.debug(f'{method} init parent object rfObject')
        
        super().__init__(**kwargs)
        # rfObject initializes
        #
        #    self.kwargs
        #    self.Portnames
        #    self.N
        #    self.Zbase
        #    self.funit
        #    self.fscale
        #    self.interp

        self.type = _RFO.TYPE_CIRCUIT
        logger.debug(f'{method} ({self.name}) type set to {_RFO.TYPE_STR[self.type]} ')
        
        # save the creation kwargs ...
        self.kwargs = kwargs.copy()
        
        # initialize the internals
        self.blocks = {
        #   'blockname' : {
        #     'ports'  : [str, ],
        #     'type'   : TYPE_GENERIC | TYPE_GTL | TYPE_TOUCHSTONE | TYPE_CIRCUIT,
        #     'idx'    : (i, j),
        #     'params' : {
        #        # TBD ...
        #     }
        #    }
        }
        
        self.M = np.zeros((0,0), dtype=np.complex128) # eventually a square matrix

        self.C = {}
        self.T = {}
        self.E = []
        
        self.waves = []
        self.unassigned = [] #TODO: interaction with __init__ of rfObject
        self.alias = {}      # {ext_name : int_name}
        self.params = {}
        self.eqn = []
        
        self.invM = None
    
    #===========================================================================
    #
    #  _ _ s t r _ _
    #
    def __str__(self):
        s = f'Circuit object: {self.name} \n\n'
        
        for k, eqn in enumerate(self.eqn):
            if eqn[0] == 'S':
                # S:block.port
                s += f'{k:3d} {eqn:20s} '
                name = eqn[2:].split('.')[0]
                block = self.blocks[name]
                row = k - block['idx'][0]
                col = block['idx'][1]
                nports = len(block['ports'])
                s += f' [{col:3d}]'
                for c in self.M[k,col:col+nports]:
                    s += f' {c.real:+8.5f}{c.imag:+8.5f}j '
                c = self.M[k,col+nports+row]
                s += f' [{col+row+nports:3d}] {c.real:+8.5f}{c.imag:+8.5f}j\n'
           
            elif eqn[0] == 'C':
                # C:A/block.port
                port = eqn.split('/')[1]
                s += f'{k:3d} C:{port:18s} '
                wave = eqn[2:]
                idxA = self.waves.index(wave)
                idxBs = self.C[wave]
                for idx in sorted([idxA] + idxBs):
                    c = self.M[k,idx]
                    s += f' [{idx:3d}] {c.real:+8.5f}{c.imag:+8.5f}j '
                s += '\n'
            
            elif eqn[0] == 'T':
                # T:block.port
                s += f'{k:3d} {eqn:20s} '
                port = eqn[2:]
                for idx in [self.waves.index(w + port) for w in ['A/','B/']]:
                    c = self.M[k,idx]
                    s += f' [{idx:3d}] {c.real:+8.5f}{c.imag:+8.5f}j '
                s += '\n'
                
#             elif eqn[0] == 'E':
#                 # E:block.port
#                 s += f'{k:3d} {eqn:20s} '
#                 port = eqn[2:]
#                 idx = self.waves.index('A/' + port)
#                 c = self.M[k,idx]
#                 s += f' [{idx:3d}] {c.real:+8.5f}{c.imag:+8.5f}j '
#                 c = self.E[k,0]
#                 s += f' [E] {c.real:+8.5f}{c.imag:+8.5f}j\n'
 
            else:
                s += '\n hhhunn ??\n\n'
                
        s += '\n'
        
        for p in self.E:
            wave, c = self.waves[p[0]], p[1]
            s += f'{wave:18s} <- {c.real:+8.5f}{c.imag:+8.5f}j [V]\n'
        
        s += '\n'
        
        if self.unassigned:
            for port in self.unassigned:
                s += f'not assigned {port}\n'
            
            s += '\n'
        
        for name, block in self.blocks.items():
            s += f'  Block {name}:\n\n'
            for s1 in block['obj'].__str__().split('\n'):
                s += f'  {s1}\n' 
            s += '\n'
        
        return s
    
    #===========================================================================
    #
    #  _ _ r e p r _ _
    #
    def __repr__(self):
        s = 'rfCircuit(\n'
        for kw, vl in self.kwargs.items():
            s += f'{kw}={vl},\n'
        s = s[:-1]+')\n'
        return s
    
    #===========================================================================
    #
    #  _ _ g e t s t a te _ _
    #
    def __getstate__(self):
        return {}
    
    #===========================================================================
    #
    #  _ _ s e t s t a te _ _
    #
    def __setstate__(self):
        pass
    
    #===========================================================================
    #
    #  A d d B l o c k
    #
    def AddBlock(self, name, obj, **params):
        method = '[rfCircuit.AddBlock]'
        logger.debug(f'{method} ({self.name}) '
                     f'{name} -- {_RFO.TYPE_STR[obj.type]} -- {params}')
        
        if name in self.blocks:
            msg = f' there is already a block named "{name}"'
            logger.critical(msg)
            raise ValueError(f'{method} {msg}')
        
        self.blocks[name] = {
            'ports' : obj.Portnames.copy(),
            'alias' : obj.alias.copy(),
            'type'  : obj.type,
            'idx'   : self.M.shape,
            'params': params.copy(),
            'obj'   : obj
        }
        
        # expand the matrix
        self.M = \
            np.vstack((
                np.hstack((
                    self.M,
                    np.zeros((self.M.shape[0],2*obj.N) , dtype=np.complex128)
                )),
                np.hstack((
                    np.zeros((obj.N, self.M.shape[1])  , dtype=np.complex128),
                    np.zeros((obj.N,obj.N)             , dtype=np.complex128),
                    np.ones((obj.N,obj.N)              , dtype=np.complex128)
                ))
            ))
        logger.debug(f'   M.shape => {self.M.shape}')
        
        # expand the excitation vector
        # self.E = np.vstack((self.E, np.zeros((obj.N,1), dtype=np.complex128)))
        
        # collect the external portnames and the eqn type
        for objport in obj.Portnames:
            self.unassigned.append(name + '.' + objport)
            self.eqn.append('S:' + self.unassigned[-1])
            
        logger.debug(f'   # eqns => {len(self.eqn)}; # unknowns => {len(self.waves)}')
        logger.debug(f'   PortNames => {self.Portnames}')
        
        # collect the wave variables
        for wavetype in ['A/','B/']:
            for objport in obj.Portnames:
                self.waves.append(wavetype + name + '.' + objport)
        
        # invalidate a possible existing solution
        self.invM = None
    
    #===========================================================================
    #
    #  C o n n e c t 
    #
    def Connect(self, *args, **kwargs):
        method = '[rfCircuit.Connect]'
        logger.debug(f'{method} ({self.name}) -- {args} -- {kwargs}')
        N = len(args)
        S = kwargs.pop('S', 2/N * np.ones((N,N)) - np.eye(N))
        
        if kwargs:
            msg = f'   unexpected keyword argument(s) {[kw for kw in kwargs]}'
            logger.critical(msg)
            raise TypeError(f'{method} {msg}')
        
        if np.array(S).shape != (N,N):
            msg = (f'   mismatch number of ports {N} and shape of S '
                  f'supplied {np.array(S).shape}')
            logger.critical(msg)
            raise ValueError(f'{method} {msg}')
        
        idxAs, idxBs = [], []
        for port in args:
            logger.debug(f'   connecting {port}')
            if port not in self.unassigned:
                msg = f'   port {port} not available'
                logger.critical(msg)
                raise ValueError(f'{method} {msg}')
            
            # remove the connected port from the free portnames
            idx = self.unassigned.index(port)
            self.unassigned.pop(idx)
            
            try:
                idxAs.append(self.waves.index('A/'+port))
                idxBs.append(self.waves.index('B/'+port))
                
            except IndexError:
                msg = f'   port {port} not available'
                logger.critical(msg)
                raise ValueError(f'{method} {msg}')
            
        for k, idxA in enumerate(idxAs):
            MC = np.zeros( (1,self.M.shape[1]) )
            MC[0,idxA] = -1
            for idxB, Skb in zip(idxBs,S[k,:]):
                MC[0,idxB] = Skb
            
            self.M = np.vstack(( self.M, MC ))
            self.eqn.append('C:'+self.waves[idxA])
            self.C[self.waves[idxA]] = idxBs
            
        # self.E = np.vstack(( self.E, np.zeros((N,1)) ))
        
        logger.debug(f'   # eqns => {len(self.eqn)}; # unknowns => {len(self.waves)}')
        logger.debug(f'   Portnames => {self.Portnames}')
        
    #===========================================================================
    #
    #  T e r m i n a t e 
    #
    def Terminate(self, port, *args, **kwargs):
        method = '[rfCircuit.Terminate]'
        logger.debug(f'{method} -- {port} -- {args} -- {kwargs}')
        if args:
            if kwargs:
                msg = (f'   unexpected keyword argument(s) {[kw for kw in kwargs]}'
                       'when a p ositional argument is given')
                logger.critical(msg)
                raise TypeError(f'{method} {msg}')
                
            elif len(args) > 1:
                msg = (f'   unexpected number of positional '
                       f'arguments given ({len(args)+1} > 2) ')
                logger.critical(msg)
                raise TypeError(f'{method} {msg}')
            
            else:
                rho = args[0]
        
        else:
            if len(kwargs) != 1:
                msg = (f'   unexpected number keyword argument(s) ({len(kwargs)}'
                       ' != 1)')
                logger.critical(msg)
                raise TypeError(f'{method} {msg}')
            
            if 'Z' in kwargs:
                Z = kwargs.get('Z')
                rho = (Z - self.Zbase)/(Z + self.Zbase)
                
            elif 'Y' in kwargs:
                Y = kwargs.get('Y')
                rho = (1 - Y * self.Zbase) / (1 + Y * self.Zbase)
                
            elif 'rho' in kwargs:
                rho = kwargs.get('rho')
                
            else:
                msg = (f'   unexpected keyword argument(s) '
                       f'{[kw for kw in kwargs]}')
                logger.critical(msg)
                raise TypeError(f'{method} {msg}')
        
        if port not in self.unassigned:
                msg = (f'   port {port} not available')
                logger.critical(msg)
                raise ValueError(f'{method} {msg}')
            
        # remove the connected port from the free portnames
        idx = self.unassigned.index(port)
        self.unassigned.pop(idx)
        
        idxA = self.waves.index('A/'+port)
        idxB = self.waves.index('B/'+port)
        
        self.M = np.vstack((
            self.M,
            np.zeros((1,self.M.shape[1]))
        ))
        self.M[-1,idxA] = 1
        self.M[-1,idxB] = -rho
        
        self.eqn.append('T:'+port)        
        # self.E = np.vstack(( self.E, np.zeros((1,1)) ))
        
        logger.debug(f'   # eqns => {len(self.eqn)}; # unknowns => {len(self.waves)}')
        logger.debug(f'   Portnames => {self.Portnames}')
        
    #===========================================================================
    #
    #  E x c i t e 
    #
    def Excite(self, port, Urf, newname=None):
        
        method = '[rfCircuit.Excite]'
        logger.debug(f'{method} ({self.name}) -- {port} -- {Urf}')
        
        # the port must be free
        if not port in self.unassigned:
            msg = f'   port {port} not available'
            logger.critical(msg)
            raise ValueError(f'{method} {msg}')
        
        if newname:
            if newname in self.Portnames:
                msg = f'   new port name {newname} is already taken'
                logger.critical(msg)
                raise ValueError(f'{method} {msg}')
                
            self.alias[newname] = port
            port = newname
        
        idx = self.unassigned.index(port)
        self.unassigned.pop(idx)
        
        self.Portnames.append(port)
        # find the associated A/wave
        idx = self.waves.index('A/'+port)
        if idx in self.E:
            logger.warning(f'{method} port {port} re-excited.')
            
        self.E.append((idx,Urf))
         
    #===========================================================================
    #
    #  G e t S 
    #
    def GetS(self, **kwargs):
        method = '[rfCircuit.getS]'
        logger.debug(f'{method} {self.name} -- {kwargs}')
        
        
        f = kwargs.get('f', self.f)
        funit = kwargs.get('funit', self.funit)
        Zbase = kwargs.get('Zbase', self.Zbase)
                
        # need to move the Exciting eqns down (in the order they appear)
        
        # update the self.M matrix
        self.Update(f=f, funit=funit)
        
        
       
    #===========================================================================
    #
    #  U p d a t e
    #
    def Update(self, f, **params):
        method = '[rfCircuit.Update]'
        logger.debug(f'{method} ({self.name}) -- {f} -- {params}')
        
        for name, block in self.blocks.items():
            obj = block['obj'],
            m, n = block['idx']
            self.M[m,n] = obj.GetS(f, self.funit, self.Zbase)
            

#===============================================================================
#        
#  _ _ m a i n _ _
#

if __name__ == '__main__':
    
    obj = rfObject(
        ports = ['a','b', 'c','d'],
        touchstone='../test data/4pj-L4PJ=403mm-lossy_normalized_50ohm_1.s4p'
    )

    a = rfCircuit(name='toplevel')
    a.AddBlock('name1', obj, L=3)
    a.AddBlock('name200', obj, L=5)
    a.Connect('name1.a','name200.b','name1.c')
    a.Terminate('name1.b', -1.0)
    for port in a.unassigned[:]:
        a.Excite(port, 1j)
    
    print(a)
