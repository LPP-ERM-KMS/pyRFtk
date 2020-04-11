################################################################################
#                                                                              #
# Copyright 2015, 2016, 2017, 2018                                             #
#                                                                              #
#                   Laboratory for Plasma Physics                              #
#                   Royal Military Academy                                     #
#                   Brussels, Belgium                                          #
#                                                                              #
#                   Fusion for Energy                                          #
#                                                                              #
#                   EFDA / EUROfusion                                          #
#                                                                              #
#                   ITER Organisation                                          #
#                                                                              #
# Author : frederic.durodie@rma.ac.be                                          #
#                          @gmail.com                                          #
#                          @ccfe.ac.uk                                         #
#                          .lpprma@telenet.be                                  #
#                                                                              #
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by   #
# the European Commission - subsequent versions of the EUPL (the "Licence");   #
#                                                                              #
# You may not use this work except in compliance with the Licence.             #
# You may obtain a copy of the Licence at:                                     #
#                                                                              #
# http://ec.europa.eu/idabc/eupl                                               #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the Licence is distributed on an "AS IS" basis,            #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the Licence for the specific language governing permissions and          #
# limitations under the Licence.                                               #
#                                                                              #
################################################################################


"""
'Fast' circuit solver for ILA simulations

Frederic Durodie 28 May 2013

2017-09-20 : added AddBlockTL and ConnectTL

2015-10-06 : added VSWR in list of value types

2014-02-25 : corrected a nasty little bug in Excite()

2014-07-15 : added normalized impedance and admittance as signal types
             (in method self._value())
             
2015-03-11 : fork to be compatible with the 'array' version of scatter3 (scatter3a)

2015-08-23 : modified terminate to allow options for RC
"""

#------------------------------------------------------ import : the usual stuff
#
# import the usual stuff
#
import numpy as np
import os
import re
from datetime import date
import warnings
import pprint as PP
import pickle

#----------------------------------------------------------------------- version

VERSION = date.fromtimestamp(os.path.getmtime(__file__)).strftime('%Y-%m-%d')

#-------------------------------------------------------- import : our libraries

from pyRFtk import scatter3a as sc
from pyRFtk.General_TL import General_TL

#------------------------------------------------------------- import : specials

from scipy.constants import speed_of_light

#.-----------------------------------------------------------------------------.

any_of = lambda kws, adict : any([kw in adict for kw in kws])
all_of = lambda kws, adict : all([kw in adict for kw in kws])

#===============================================================================
#
# circuit class
#
class circuit():
    """
    ciruit : class to contruct and solve circuits.

    Workflow :
        ct = circuit(fMHz=..., Zbase=...)
        
                # optional fMHz  : not really needed to solve a set of connected
                #                  S, Y, Z matrices ... . However if one adds a
                #                  matrix from a scatter2 object by
                #                  ct.AddBlock("name", SZ=scatter2_object)
                #                  then the frequency will be set and checked when
                #                  adding similar matrices
                #
                # optional Zbase : will default to 50 Ohm. The setting is used to
                #                  covert S matrices and/or scatter2 objects to the
                #                  correct reference base

        ct.AddBlock("this block name",
                    S= ... or Z= ... or Y= .... or SZ= ... ,
                    Portnames= list of names)
                    
                # optional Portnames : if not given default names will be assumed
                #                      for a supplied S matrix the names are S_nn
                #                                     Z                      Z_nn
                #                                     Y                      Y_nn
                #                      for a scatter2 object (SZ) the names will
                #                      will be taken from the object
                #
                # when importing S matrices it is assumed they have the correct
                # reference impedance

        # ... add more matrices ... 

        ct.Connect("blockname/portname","blockname/portname")

        # ... connect more ports ... 

        ct.Terminate("blockname/portname", RC= ... or Z= ... or Y= ...)

                # termination : one of RC (reflectioncoeffient expressed in the
                #               reference impedance base of the circuit),
                #               Z (impedance) or Y (admittance)

        # ... terminate other ports ... 

        ct.Excite("portname/blockname", incident_wave)

                # the incident (forward) wave expressed in the circuit's reference
                # impedance

        sol = ct.Solve(excitations= ..., printit= True | False)
        
                # optional excitations : if given this is a dict object with
                #                        "blockname/portname" : value as items
                #                        if not given it is assumed that the
                #                        free ports (not connected or terminated)
                #                        will all have been given an excitation
                #
                # optional printit : if True with print the solutions on the
                #                    screen
                #
                # the returned value is a dict :
                #
                # { ..., "blockname/A_portname" : value,  # incident voltage wave
                #        "blockname/B_portname" : value,  # reflected voltage wave
                # ... }
                # 
                # for all ports of all blocks. So for connected ports there will
                # be duplicate information as one port's incident wave is the
                # otherone's reflected wave.
                #
                # note calling Solve is optional when subsequently calling e.g.
                # Solution (the latter will detect if there is a solution that
                # was evaluated and is still valid).

        Q = ct.Solution("block/portname", type)

                # get solution at "block/portname" as type :
                #   'V'    : voltage
                #   'I'    : current
                #   'A'    : incident voltage wave
                #   'B'    : reflected voltage wave
                #   'Z'    : impendance
                #   'Y'    : admittance
                #   'P'    : power into the port
                #   'absV' : amplitude of the voltage
                #   'argV' : phase of the voltage (degrees)
                #   'absI' : amplitude of the current
                #   'argI' : phase of the current (degrees)

        # ... get more solutions as desired ...

        # change a ciruit element (matrix) by

        ct.AddBlock('existing_blockname', S= | Y= | Z= | SZ= <a new matrix>)

        # re-excite with other excitations :

        ct.Excite('existing_blockname/portname', <new incident wave>)

        # or

        ct.Solve(excitations={ ...})

    Usefull :

        S = ct.Get_S('blockname',[Z0=reference impedance])
        Z = ct.Get_Z('blockname')
        Y = ct.Get_Y('blockname')
        SZ = ct.Get_SZ('blockname',[Z0=reference impedance])
        
        ct.dump2py('filename',overwrite=False,solvit=False)
        ct.dump2str(solvit=False)

    Experimental :

        ct.Get_Coeffs( ... )

    """

#===============================================================================
#
# d e f i n e   v e r s i o n
#
    __version_info__ = tuple(VERSION.split('-'))
    __version__ = '.'.join(__version_info__)

#===============================================================================
#
#   _ _ i n i t _ _
#
    def __init__(self, **kwargs):
        """
        kwargs:
        
        fMHz : frequency [MHz] (defaults None -> AddBlock will set it from an SZ)
        Zbase : scattering matrices base impedance [Ohm] (default 50.)
        c0 : speed of light [m/s] (default scipy.constants.speed_of_light)
        dx : resolution of VSW curves [m] (default -0.05) (see TLdata)
        """
        
        if 'state' in kwargs:
            d = kwargs.pop('state')
            self.state(d)
            
            if kwargs:
                print('circuit.__init__: state is incompatible with any other kwarg')                
                self.check_kwargs_left('__init__', kwargs)
                
            return
        
        self.fMHz = kwargs.pop('fMHz', None)
        self.Zbase = kwargs.pop('Zbase', 50.)
        self.c0 = kwargs.pop('c0',speed_of_light)
        self.dx = kwargs.pop('dx', -0.05)
        
        self.check_kwargs_left('__init__', kwargs)
        
        self.counter = 0
        
        self.M = []
        self.invM = None
        self.sol = None
        self.names = []

        #--------------------------------------------------------data structures
        
        self.V = []
        self.E = {} # excited ports       blockname/portname : eqn number
        
        # self.internal = {} # internal names (i.e. without the blockname) of the excited ports
        
        self.T = {} # terminated ports    blockname/portname : (eqn number, i1, i2)
        self.C = {} # connected ports
                    #   {... blockname1/port1 : {blockname2/port2:info},
                    #        blockname2/port2 : {blockname1/port1:info}, ...}
                    #
                    #   info : ... , self.TLs key | None
        
        self.TLs = {}   # TL info {'args': TLkwargs,
                        #          'length': lngth,
                        #          'relpos': relpos,
                        #          'ports': [blockname+'/'+p for p in Ports],
                        #          'plotargs': plotargs,
                        #          'isGTL': booleans
                        #         }
                              
        self.S = []
        self.blocks = {} # Key is a block name :
                        #   Value is a dict
                        #    idx0   : start indx
                        #    Nports : number of ports
                        #    names  : [port names]
                        #    TLid   : None | self.TLs key

#===============================================================================
#
#    f i n d _ o r p h a n s
#

    def find_orphans(self):
        
        regex = r'(.*)/A_(.*)'
        orphans = []
        for name in self.names:
            found = re.findall(regex,name)
            if found:
                blk, prt = found[0]
                tname = blk+'/'+prt
                if tname not in self.E:
                    if tname not in self.T:
                        if tname not in self.C:
                            orphans.append(tname)
                                    
        return orphans
            
#===============================================================================
#
#   c h e c k _ k w a r g s _ l e f t
#
    def check_kwargs_left(self,method,kwargs):
        
        if kwargs:
            raise ValueError('circuit.%s: Unrecognised parameter(s):' % method,
                             [k for k in kwargs])

#===============================================================================
#
#   n e w I D
#
    def newID(self):
        self.counter +=1
        return self.counter
    
#===============================================================================
#
# s t a t e
#
    def state(self, d = None):
        if d is None:
            
            sTLs = {}
            for TL, val in self.TLs.items():
                if isinstance(val['isGTL'], General_TL):
                    sTLs[TL] = val.copy()
                    sTLs[TL]['isGTL'] = sTLs[TL]['isGTL'].state
                    

#             # the larger structure is M but it is full of zeros
#             sM = {'shape': np.array(self.M).shape,
#                   'blocks': [],
#                   'terminations': [],
#                   'excitations': [],
#                   'connections': [],
#                  }
#             
#             for blk, b in self.blocks.items():
#                 h0, v0 = b['idx'][0], b['idx'][1]
#                 N = b['Nports']
#                 sM['blocks'].append(
#                                 (h0, v0, N, self.M[h0:h0+N, v0:v0+N].tolist())
#                                    )
#                 
#             for port1, (neq, i1, port2, i2, TLid) in self.C.items() :                ...
#             

            return {'fMHz'    : self.fMHz,
                    'Zbase'   : self.Zbase,
                    'c0'      : self.c0,
                    'dx'      : self.dx,
                    'V'       : self.V.tolist(),
                    'E'       : self.E,
                    ## 'internal': self.internal, ## check comments in self.excite()
                    'T'       : self.T,
                    'C'       : self.C,
                    'TLs'     : sTLs,
                    'S'       : [] if self.S == [] else self.S.tolist(),
                    'blocks'  : self.blocks,
                    'names'   : self.names,
                    'M'       : np.array(self.M).dumps(),
                    # 'sM'       : sM,
                    # we don't store the invM but just the fact that it was there or not
                    'invM'    : None if self.invM is None else True,
                    'sol'     : self.sol,
                    'counter' : self.counter,
                   }
        
        else:

            self.fMHz = d['fMHz']
            self.Zbase = d['Zbase']
            self.c0 = d['c0']
            self.dx = d['dx']
            self.V = np.array(d['V'])
            self.E = d['E']
            ## self.internal = d['internal']  ## check comments in self.excite()
            self.T = d['T']
            self.C = d['C']
            self.TLs = d['TLs']
            self.S = np.array(d['S'])
            self.blocks = d['blocks']
            self.names = d['names']
            # sM = pickle.loads(d['sM'])
            self.M = pickle.loads(d['M'])
            self.invM = d['invM']
            if self.invM is True:
                self.invM = np.linalg.inv(self.M)
            self.sol = d['sol']
            self.counter = d['counter']
            
#             # re-initialize the matrix
#             self.M = 1J * np.zeros(sM['shape'])
#             for h0, v0, N, MM in sM['blocks']:
#                 self.M[h0:h0+N,v0:v0+N] = np.array(MM)
#                 self.M[h0+N:h0+2*N,v0:v0+N] = np.eye(N)
                
            
            # re-initialize the GTLs
            
            for TL in self.TLs:
                if (isinstance(self.TLs[TL]['isGTL'], dict) and 
                    len(self.TLs[TL]['isGTL']) == 2 and 
                    'L' in self.TLs[TL]['isGTL'] and 
                    'kwargs' in self.TLs[TL]['isGTL']):
                
                    self.TLs[TL]['isGTL'] = General_TL(
                        self.TLs[TL]['isGTL']['L'],
                        **self.TLs[TL]['isGTL']['kwargs'])
    
#===============================================================================
#
#   _ _ s t r _ _
#
    def __str__(self):
        s = 'circuit :\n'
        s += '  Zbase : %7.3f Ohm\n' % self.Zbase
        if self.fMHz:
            s += '  frequency : %7.3f MHz\n' % self.fMHz
        else:
            s += '  frequency : undefined \n'
        s += '  %d blocks :\n' % len(self.blocks)
        
        # find the maximum length of the TL labels
        tllen = 0
        if self.TLs:
            tllen = max([len(TLlbl) for TLlbl in self.TLs])
        tlfmt = '%%-%ds' % (tllen + 1)
        
        EQNS = []
        if self.blocks :
            # sort the blocks in order of creation
            blks = sorted([(descr['idx0'][0], 
                            descr['idx0'][1], 
                            block,
                            descr['Nports'], 
                            descr['names'], 
                            descr.get('TLid',{'TL':None})
                           ) for block, descr in self.blocks.items()
                          ])
            # find the maximum string length of block/name
            slen = max([max([len(block+'/'+name) for name in names])
                       for h0, v0, block, Nports, names, isTL in blks])
            fmt = '%%-%ds' % (slen + 1)
            
            # 'print' the info
            for h0, v0, block, Nports, names, TL in blks :
                h1, v1 = h0 + Nports, v0 + Nports
                TLlbl = TL if TL else '-'*tllen
                for h in range(Nports):
                    ss = (('[S] ' + fmt + tlfmt + '[%3d .. %3d]: ') % (
                        block+'/'+names[h], TLlbl, v0, v1-1))
                    for v in range(Nports) :
                        CC = self.M[h+h0,v+v0]
                        ss += '%7.4f%+7.4fj  ' % (np.real(CC),np.imag(CC)) 
                    CC = self.M[h+h0,v0+Nports+h]
                    ss += ('  [%3d] %7.4f%+7.4fj' %
                           (v0+Nports+h, np.real(CC),np.imag(CC)))
                    EQNS += [(h0+h, ss)]
                    
        if self.T :
            for port1, (i1, i2, neq) in self.T.items() :
                CC, CR = self.M[neq,i1], self.M[neq,i2]
                EQNS += [(neq,
                          ('[T] ' + fmt +  (tlfmt%' ')      + '[%3d]: ') %  (port1, i1) +
                          '%7.4f%+7.4fj ' % (np.real(CC), np.imag(CC)) +
                          '[%3d] ' % i2 +
                          '%7.4f%+7.4fj' % (np.real(CR), np.imag(CR)))]
                
        if self.E :
            for port1, (neq, i1) in self.E.items() :
                CC, VR = self.M[neq,i1], self.V[neq]
                EQNS += [(neq,
                          ('[E] ' + fmt +  (tlfmt%' ') +'[%3d]: ') % (port1, i1) +
                          '%7.4f%+7.4fj ' % (np.real(CC), np.imag(CC)) +
                          'V[%3d] ' % neq +
                          '%7.4f%+7.4fj' % (np.real(VR), np.imag(VR)))]

        if self.C :
            for port1, (neq, i1, port2, i2, TLid) in self.C.items() :
                TLlbl = TLid if isinstance(TLid, str) else '-'*tllen
                
                if isinstance(i2, tuple):   # the new version allows to connect
                                            # more than two ports
                    C1 = self.M[neq,i1]
                    tEQ = (('[C] ' + fmt + tlfmt) % (port1,TLlbl) +
                           '[%3d] : ' % i1 +
                           '%7.4f%+7.4fj ' % (C1.real, C1.imag)
                          )
                    for i2k in i2:
                        C2 = self.M[neq,i2k]
                        tEQ += (' [%3d] ' % i2k +
                                '%7.4f%+7.4fj ' % (C2.real, C2.imag)
                               )
                    EQNS += [(neq, tEQ)]
                    
                else: # this is the old connect data
                    C1, C2 = self.M[neq,i1], self.M[neq,i2]
                    EQNS += [(neq,
                              ('*** [C] %-15s -> %-15s ') % (port1, port2) +
                              '[%3d] : ' % i1 +
                              '%7.4f%+7.4fj ' % (np.real(C1), np.imag(C1)) +
                              '[%3d] ' % i2 +
                              '%7.4f%+7.4fj' % (np.real(C2), np.imag(C2)))]

        for eq, ss in sorted(EQNS) :
            s += '%4d : %s\n' % (eq,ss)
        
        s += '\n' if self.TLs else ''
        for TLid, TLdata in sorted(self.TLs.items()):
            # print(TLid, TLdata)
            
            # fmt = '{TLid:<%ds}: {length:6.3f}m,' % tllen
            # tls = fmt.format(TLid=TLid, **TLdata) + (' args: %r ' % TLdata['args'])
            
            fmt = '  %%-%ds: %%r' % tllen
            tls = fmt % (TLid, TLdata)
            
            s += tls + '\n'
            
        return s
            
#===============================================================================
#
#   A d d B l o c k
#
    def AddBlock(self, name, **kwargs):
        """
        add a scattering matrix to the system :
        
        SZ | S | Z | Y : corresponding matrix (strictly one required)
        Portnames      : list of the port names (defaults <type>_## for S, Z, Y
                         
        TLid           : <str> | None (default)
        """
        
        ##------------------------------------------------------------------- SZ
        if 'SZ' in kwargs:
            dtype = 'SZ'
            if any_of(['S','Z','Y'],kwargs):
                raise ValueError(
                    'circuit.AddBlock: only one of SZ, S, Z or Y allowed')
            else:
                SZb = kwargs.pop('SZ')
                if self.fMHz and np.abs(1 - SZb.fMHz/self.fMHz) > 1E-12 :
                    print("circuit_Class.AddBlock() : frequency mismatch")
                    print(self.fMHz, SZb.fMHz, self.fMHz - SZb.fMHz)
                    raise ValueError
                else :
                    self.fMHz = SZb.fMHz
                Sb = SZb.Get_S(Z0=self.Zbase)
                Portnames = kwargs.pop('Portnames', SZb.Portname[:])
        
        ##-------------------------------------------------------------------- S
        elif 'S' in kwargs:
            dtype = 'S'
            if any_of(['Z','Y'],kwargs):
                raise ValueError(
                    'circuit.AddBlock: only one of SZ, S, Z or Y allowed')
            else:
                Sb = np.array(kwargs.pop('S'))
                Portnames = kwargs.pop('Portnames',
                                       ['S_%02d' % k for k in range(Sb.shape[0])])
                
        ##-------------------------------------------------------------------- Z
        elif 'Z' in kwargs:
            dtype = 'Z'
            if 'Y' in kwargs:
                raise ValueError(
                    'circuit.AddBlock: only one of SZ, S, Z or Y allowed')
            else:
                Z = np.array(kwargs.pop('Z'))
                Zeye = self.Zbase * np.eye(Z.shape[0])
                Sb = np.dot(np.linalg.inv(Z - Zeye),(Z + Zeye))
                Portnames = kwargs.pop('Portnames',
                                       ['Z_%02d' % k for k in range(Z.shape[0])])
                
        ##-------------------------------------------------------------------- Y
        elif 'Y' in kwargs:
            dtype = 'Y'
            Y = np.array(kwargs.pop('Y'))
            Yeye = np.eye(Y.shape[0]) / self.Zbase
            Sb = np.dot(np.linalg.inv(Yeye - Y),(Yeye + Y))
            Portnames = kwargs.pop('Portnames',
                                   ['Y_%02d' % k for k in range(Y.shape[0])])
            
        ##----------------------------------------------- none of SZ, S, Z, or Y
        else:
            raise ValueError('circuit.Addblock: one of SZ, S, Z or Y required')
            
        # at this point Portnames, SZ, S, Z, Y have been popped from kwargs
               
        ##----------------------------------------------------------------- TLid
        TLid = kwargs.pop('TLid',None)
        
        ##--------------------------------------------------------- unrecognised
        self.check_kwargs_left('AddBlock', kwargs)

        ##------------------------------------------------------final processing
        Nports = Sb.shape[0]
        if len(Portnames) is not Nports :
            print("circuit_Class.AddBlock : optional Portnames [%d] is not " +
                  "consistent with size of '%s[%d,%d]'" % 
           (len(Portnames),dtype,Nports,Nports))
            raise ValueError
        
        if name in self.blocks :
            if self.blocks[name]['Nports'] == Nports :
                # print "warning : block already exists, so will replace"
                if Portnames != self.blocks[name]['names']:
                    raise ValueError(
                        "circuit_Class.AddBlock : port names don't match " +
                          "(or are in different order)")
                (r0, c0) = self.blocks[name]['idx0']
                Nports = self.blocks[name]['Nports']
                r1, c1 = r0 + Nports, c0 + Nports

                self.M[r0:r1,c0:c1] = Sb
                
                # need to decide how to update the TL information
                # do we need to do this ? MakeTL already has done it
                
                if TLid:
                    self.blocks[name]['TLid'] = TLid
                    
            else :
                print("AddBlock : block size does't match")
                raise ValueError
            
        else : # this is a new block
            r0, c0 = self.M.shape if len(self.M) else (0,0)
            
#            obsolete code
#            TLlbl = info['TL']['label'] if info['TL'] else None

            self.blocks[name] = {'idx0'  : (r0, c0),
                                 'Nports': Nports,
                                 'names' : Portnames,
                                 'TLid'  : TLid}            
            # expand M

#             if False :
#                 print('-'*100)
#                 printM(SZ.S)
#                 printM(Sb)                                     
#                 print('-'*100)
#                 print()
            
            if len(self.M) :
                MV = np.hstack(
                    (np.zeros((Nports,self.M.shape[1])),
                     Sb ,
                     -np.eye(Nports))
                )
                self.M = np.vstack(
                    (np.hstack(
                        (self.M,
                         np.zeros((self.M.shape[0], 2*Nports)))
                     ),
                     MV)
                )
                self.V = np.vstack((self.V, np.zeros((MV.shape[0],1))))
                
            else :
                self.M = np.hstack((Sb,-np.eye(Nports)))
                self.V = np.zeros((self.M.shape[0],1))

            self.names += [name + '/A_' + pname for pname in Portnames]
            self.names += [name + '/B_' + pname for pname in Portnames]
            
        self.invM = None
        self.sol = None
        
#===============================================================================
#
# M a k e T L
#
    def MakeTL(self, lngth, tkwargs):
        """
        returns SZ and info based on the kwargs given in tkwargs
        note : as a side effect the relevant kwargs are popped from tkwargs
        lngth : length [m]
        
        tkwargs is a dict containing :
        
        Portnames : list of 2 strings (default ['1','2'])
        relpos : relative position of the first port [m] (default 0.)
        
        combination AZV (default)
        
        Z  : characteristic impedance (default self.Zbase [Ohm])
        A  : attenuation (default 0 [m-1])
        V  : wave velocity relative to speed of light in vacuum (default 1 ([1])
        
        combination GRLC
        
        G  : conductance (default 0 [Mho/m])
        R  : resistance (default 0 [Ohm/m])
        L  : inductance (default self.Zbase/c0 [nH/m])
        C  capacitance (sdefault 1/(self.Zbase*c0) [pF/m])
        
        TLid : id of the TL (default TL<3 digit counter>)
        
        plotargs : kwargs that can be passed to a plotting routine
        
        dx : resulution for evaluation of the standing waves
             None -> use approx 5 degree i.e. equivalent to -c0/f/72
             0 : return a function f(x) instead of a list of values [default]
             positive int : number of equidistant points (includes end)
             positive float : exact dx (so the end point may not be returned)
             negative float : adjust dx downwards so that the points are 
                              equidistant and the end point in included
        """
        
        TLid = tkwargs.pop('TLid', None)
        
        # get the transmission line parameters -> processed by Scatter.AddTL
        
        TLkwargs = dict([(kw, tkwargs.pop(kw)) 
                         for kw in ['G','R','L','C','A','Z','V'] 
                         if kw in tkwargs])
                            
        # determine if we came through AddBlockTL or ConnectTL
            
        old_TLid = None

        blockname = tkwargs.pop('name','')  # for a connection blockname will be
                                            # an empty string
        
        if blockname:

            if blockname in self.blocks:
                # the block exists and there may already be info on the TL
                if self.blocks[blockname].get('TLid',None):
                    # there appears to be some info there
                    old_TLid = self.blocks[blockname]['TLid']
                    old_data = self.TLs.get(old_TLid,None)

                else:
                    # there was no TL info : possibly the block was originally
                    # created/replaced by AddBlock rather than AddBlockTL:
                    # we will ignore this for the moment
                    
                    #FIXME:
                    # if there was no old_TLid and none is given (TLid is None)
                    # then we need to create a new one
                    pass
                
        if old_TLid:
            # so there was a blockname which has set the TLid
            if old_data['blockname'] != blockname:
                raise ValueError( 'circuit_Class.MakeTL: (internal error):'
                                  ' inconsistent blockname')
                
            if TLid:
                if TLid != old_TLid :
                    raise ValueError('circuit_Class.MakeTL: inconsistent TLid')
                
            else:
                TLid = old_TLid # No TLid was given but it was given before
                                # so we assume it was not changed
                
        else:
            if TLid:
                pass # we got wat we need
            else:
                TLid = '%%TL%03d%%' % self.newID() # create a new TLid
        
        # MakeTL will always be called with Portnames from ConnectTL. When
        # called from AddBlockTL it may not have Portnames
        
        Ports = tkwargs.pop('Portnames', None)
 
        if Ports:
            if (not isinstance(Ports, list) or len(Ports) is not 2 
               or any([not isinstance(p,str) for p in Ports])):
                raise ValueError('circuit_Class.MakeTL: Portnames if given'
                                 ' must be a list of 2 strings')

            if old_TLid: # we assume that old_data will always be a dict
                if Ports != old_data['ports']:
                    raise ValueError('circuit_Class.MakeTL:'
                                     ' inconsistent Portnames')
        else:
            if old_TLid:
                Ports = old_data['ports']
            else:
                Ports = ['1','2'] # defaults
                                
        relpos = tkwargs.pop('relpos', 0.)  # this is the relative position of
                                            # the first port 
                        
        if TLid in self.TLs and blockname != self.TLs[TLid]['blockname']:
            print(self)
            raise ValueError('circuit_Class.MakeTL: TLid %r is already exsists'
                             % TLid)
        
        plotargs = tkwargs.pop('plotargs',{})
        
        dx = tkwargs.pop('dx',self.dx)
        if dx is None:
            dx = -speed_of_light/(self.fMHz*1E6)/72

        self.check_kwargs_left('MakeTL', tkwargs)
        
        SZ = sc.Scatter(fMHz=self.fMHz,Zbase=self.Zbase)
        SZ.AddTL(Ports,lngth,**TLkwargs)
        
        self.TLs[TLid] = {'args'      : TLkwargs,
                          'length'    : lngth,
                          'relpos'    : relpos,
                          'ports'     : Ports,
                          'plotargs'  : plotargs,
                          'blockname' : blockname,
                          'Zc'        : SZ.lastTL['Zc'],
                          'gamma'     : SZ.lastTL['gamma'],
                          'dx'        : dx,
                          'isGTL'     : False,
                         }
        
        return SZ, TLid
   
#===============================================================================
#
# M a k e G T L
#
    def MakeGTL(self, lngth, GTLkwargs, tkwargs):
        """
        returns SZ and info based on the kwargs given in GTLkwargs
        note : as a side effect the relevant kwargs are popped from tkwargs
        
        lngth : length [m]
        
        tkwargs is a dict containing :
        
        Portnames : list of 2 strings (default ['1','2'])
        
        relpos : relative position of the first port [m] (default 0.)
        
        Similar to AddBlock but we know this is a General Transmission Line (GTL)
        handled by pyRFtk.General_TL class.
        
        length : length [m]
        
        kwargs:
        
        Portnames : list of 2 strings (default ['1','2'])
        
        relpos : relative position of the first port [m] (default 0.)
        
        combination of General_TL kwargs:
        
        Geometry:
            'OD': None -> 0.2286m if LTL, CTL, rTL or gTL cannot be resolved,
            'ID': None,
            
        Material properties:
            'rho': None, 'rhoO': None, 'rhoI': None,
            'epsr': None -> 1.00, 'mur': 1.0, ('vr','V'): None, 'etar': None,
            'murI': 1.0, 'murO': 1.0,'sigma': None, 'tand': None,
        
        TL properties:
            'A': None, 'AdB': None,
            ('rTL', 'R'): None, 'rTLO': None, 'rTLI': None,
            ('gTL', 'G'): None, 
            ('LTL', 'L'): None, ('CTL','C'): None, ('Z0TL','Z'): None,
            
        other options:
            fMHz, Zbase : will be copied from circuit_class
            maxdeg, maxdx, xs : 
            
        TLid : id of the TL (default _TL<3 digit counter>_)
        
        plotargs : kwargs that can be passed to a plotting routine
        
        dx : resulution for evaluation of the standing waves
             None -> use approx 5 degree i.e. equivalent to -c0/f/72
             0 : return a function f(x) instead of a list of values [default]
             positive int : number of equidistant points (includes end)
             positive float : exact dx (so the end point may not be returned)
             negative float : adjust dx downwards so that the points are 
                              equidistant and the end point in included
        """
        
        TLid = tkwargs.pop('TLid', None)
                                    
        # determine if we came through AddBlockTL or ConnectTL
            
        old_TLid = None

        blockname = tkwargs.pop('name','')  # for a connection blockname will be
                                            # an empty string
        
        if blockname:

            if blockname in self.blocks:
                # the block exists and there may already be info on the TL
                if self.blocks[blockname].get('TLid',None):
                    # there appears to be some info there
                    old_TLid = self.blocks[blockname]['TLid']
                    old_data = self.TLs.get(old_TLid,None)

                else:
                    # there was no TL info : possibly the block was originally
                    # created/replaced by AddBlock rather than AddBlockTL:
                    # we will ignore this for the moment
                    
                    #FIXME:
                    # if there was no old_TLid and none is given (TLid is None)
                    # then we need to create a new one
                    
                    raise RuntimeError(
                        'MakeGTL: there was no TLid and no TLid info was given:\n'
                        'possibly the block was originally created/replaced by \n'
                        'AddBlock rather than AddBlockGTL (blockname=%r' % blockname)
                
        if old_TLid:
            # so there was a blockname which has set the TLid
            if old_data['blockname'] != blockname:
                raise ValueError( 'circuit_Class.MakeTL: (internal error):'
                                  ' inconsistent blockname:')
                
            if TLid:
                if TLid != old_TLid :
                    raise ValueError('circuit_Class.MakeTL: inconsistent TLid')
                
            else:
                TLid = old_TLid # No TLid was given but it was given before
                                # so we assume it was not changed
                
        else:
            if TLid:
                pass # we got wat we need
            else:
                TLid = '%%TL%03d%%' % self.newID() # create a new TLid
        
        # MakeTL will always be called with Portnames from ConnectTL. When
        # called from AddBlockTL it may not have Portnames
        
        Ports = tkwargs.pop('Portnames', None)
 
        if Ports:
            if (not isinstance(Ports, list) or len(Ports) is not 2 
               or any([not isinstance(p,str) for p in Ports])):
                raise ValueError('circuit_Class.MakeTL: Portnames if given'
                                 ' must be a list of 2 strings')

            if old_TLid: # we assume that old_data will always be a dict
                if Ports != old_data['ports']:
                    raise ValueError('circuit_Class.MakeTL:'
                                     ' inconsistent Portnames')
        else:
            if old_TLid:
                Ports = old_data['ports']
            else:
                Ports = ['1','2'] # defaults
                                
        relpos = tkwargs.pop('relpos', 0.)  # this is the relative position of
                                            # the first port 
                        
        if TLid in self.TLs and blockname != self.TLs[TLid]['blockname']:
            print(self)
            raise ValueError('circuit_Class.MakeTL: TLid %r is already exsists'
                             % TLid)
        
        plotargs = tkwargs.pop('plotkws',{})
        
        maxdx = tkwargs.pop('dx',self.dx)
        if maxdx is None:
            maxdx = -speed_of_light/(self.fMHz*1E6)/72
        GTLkwargs.update({'maxdx': maxdx})

        self.check_kwargs_left('MakeGTL', tkwargs)
        
        tGTL = General_TL(lngth, **GTLkwargs)
        
        SZ = tGTL.get_SZ(Portnames=Ports)
        
        self.TLs[TLid] = {'args'      : GTLkwargs,
                          'length'    : lngth,
                          'relpos'    : relpos,
                          'ports'     : Ports,
                          'plotargs'  : plotargs,
                          'blockname' : blockname,
                          'Zc'        : None,
                          'gamma'     : None,
                          'dx'        : maxdx,
                          'isGTL'     : tGTL,
                         }
        
        return SZ, TLid
   
#===============================================================================
#
# A d d B l o c k T L
#

    def AddBlockTL(self, name, length, **kwargs):
        """
        Similar to AddBlock but we know this is a Transmission Line
        
        length : length [m]
        
        kwargs:
        
        Portnames : list of 2 strings (default ['1','2'])
        
        relpos : relative position of the first port [m] (default 0.)
        
        combination AZV (default)
        
        A  : attenuation (default 0 [m-1])
        Z  : characteristic impedance (default self.Zbase [Ohm])
        V  : wave velocity relative to speed of light in vacuum (default 1 ([1])
        
        combination GRLC
        
        G  : conductance (default 0 [Mho/m])
        R  : resistance (default 0 [Ohm/m])
        L  : inductance (default self.Zbase/c0 [H/m])
        C  capacitance (default 1/(self.Zbase*c0) [F/m])
        
        combination of GRZV
        
        G  : conductance (default 0 [Mho/m])
        R  : resistance (default 0 [Ohm/m])
        Z  : characteristic impedance (default self.Zbase [Ohm])
        V  : wave velocity relative to speed of light in vacuum (default 1 ([1])


        TLid : id of the TL (default _TL<3 digit counter>_)
        
        plotargs : kwargs that can be passed to a plotting routine
        """
        
        ##-------------------------------------------------------- define the TL
        
        kwargs.update({'name':name, 'TLid':name}) # will be consumed by .MakeTL
        SZ, TLid = self.MakeTL(length, kwargs)
        
        ##----------------------------- anything left that should not be there ?
        self.check_kwargs_left('AddBlockTL', kwargs)

        ##-------------------------------------------------------- add the block
        self.AddBlock(name, SZ=SZ,  TLid=TLid)

#===============================================================================
#
# A d d B l o c k G T L
#

    def AddBlockGTL(self, name, length, **kwargs):
        """
        Similar to AddBlock but we know this is a General Transmission Line (GTL)
        handled by pyRFtk.General_TL class.
        
        length : length [m]
        
        kwargs:
        
        Portnames : list of 2 strings (default ['1','2'])
        
        relpos : relative position of the first port [m] (default 0.)
        
        combination of General_TL kwargs:
        
        Geometry:
            'OD': None -> 0.2286m if LTL, CTL, rTL or gTL cannot be resolved,
            'ID': None,
            
        Material properties:
            'rho': None, 'rhoO': None, 'rhoI': None,
            'epsr': None -> 1.00, 'mur': 1.0, ('vr','V'): None, 'etar': None,
            'murI': 1.0, 'murO': 1.0,'sigma': None, 'tand': None,
        
        TL properties:
            'A': None, 'AdB': None,
            ('rTL', 'R'): None, 'rTLO': None, 'rTLI': None,
            ('gTL', 'G'): None, 
            ('LTL', 'L'): None, ('CTL','C'): None, ('Z0TL','Z'): None,
            
        other options:
            fMHz, Zbase : will be copied from circuit_class
            maxdeg, maxdx, xs : 
            
        TLid : id of the TL (default _TL<3 digit counter>_)
        
        plotargs : kwargs that can be passed to a plotting routine
        """
        
        ##-------------------------------------------------------- define the TL
        
        GTLkwargs = dict([(kw, kwargs.pop(kw)) for kw in [
            'OD','ID','rho','rhoI','rhoO','espr','mur','vr','V','etar','murI',
            'murO','A','AdB','rTL','R','rTLI','rTLO','gTL','G','sigma','tand',
            'LTL','L','CTL','C','Z0TL','Z', 'maxdeg', 'maxdx', 'xs', 'qTLI',
            'qTLO','dx']
            if kw in kwargs])

        GTLkwargs.update({'fMHz':self.fMHz, 'Zbase':self.Zbase})
        
        kwargs.update({'name': name,
                       'TLid': name,
                      }) # will be consumed by MakeTL
        
        SZ, TLid = self.MakeGTL(length,GTLkwargs,kwargs)
        # print(SZ, TLid)
        
        ##----------------------------- anything left that should not be there ?
        self.check_kwargs_left('AddBlockGTL', kwargs)

        ##-------------------------------------------------------- add the block
        self.AddBlock(name, SZ=SZ,  TLid=TLid)

#===============================================================================
#
# T e r m i n a t e
#
    def Terminate(self, port1, *args, **kwargs):
        """
        
        port1 : Blockname/Portname
        *args : a single argument which will be the RC to terminate the port
        
        **kwargs : one of RC, Z or Y
        
        RC can be a float/complex (defaults to 0.) or a dict :
        
        'G'     : modulus of the refection coefficient 
                  (default 1. if 'units' is 'MA' or 0. if 'units' is 'DB')
        'phase' : phase of the refelection (default 0.)
        'deg'   : degrees (True, default) or radians (False)
        'units' : 'MA' (default) or 'DB'
        
        
        kwargs : Z, Y, RC
        
        if no kwargs are given it is assumed args is a complex value 
          representing the reflection coefficient in self.Zbase. If args is
          not supplied it is assumed that the RC = 0.
          
        else kwargs contains one of 'Y', 'Z' or 'RC'.
        
        Note that:
           
           circuit.Terminate(port1) terminates port1 on circuit.Zbase
        
        while
        
           circuit.Terminate(port1,RC={}) terminates port1 on an open circuit
           
        """
        ##-------------------------- no kwargs maybe 1 additional positional arg
        if len(kwargs) is 0 :
            if len(args) is 1 :
                RC = args[0]
            elif len(args):
                raise ValueError('circuit.Terminate : there can only be at most'
                                 ' two positional arguments : got %d' % 
                                 (len(args)+1))
            else:
                RC = 0.
            
        ##------------------------------------------------------------------- RC
        elif 'RC' in kwargs:
            if any_of(['Z', 'Y'],kwargs):
                raise ValueError(
                        'circuit.Terminate : only one of RC, Z or Y allowed')
            
            RC = kwargs.pop('RC')
            
            # legacy code hack ... in fact RC should have been a dict.
            
            if not isinstance (RC, dict):
                
                phase = kwargs.pop('phase', 0.)
                deg = kwargs.pop('deg', True)
                units = kwargs.pop('units', 'MA')

                RC = ((RC if units == 'MA' else 10**(RC/20))
                       * np.exp(1j * phase *((np.pi/180) if deg else 1. )))
            
            else:
                
                if any([kw in kwargs for kw in ['phase','deg','units']]):
                    raise ValueError(
                        'circuit.Terminate : RC of type dict does not allow '
                        ' "phase" and/or "deg" kwargs.')

        ##-------------------------------------------------------------------- Z
        elif 'Z' in kwargs :
            if 'Y' in kwargs:
                raise ValueError(
                        'circuit.Terminate : only one of RC, Z or Y allowed')
                
            Z = kwargs.pop('Z') / self.Zbase
            RC = (Z - 1.) /  (Z + 1.)
        
        ##-------------------------------------------------------------------- Y
        elif 'Y' in kwargs :
            Y = self.Zbase * kwargs.pop('Y')
            RC = (1. - Y) /  (1. + Y)
            
        ##---------------------------------------------------- any kwargs left ?
        else :
            # we can only come here if there were kwargs but none were found
            # above: so there is at least an unrecognized kwarg left which
            # will be caught by check_kwargs_left
            pass
        
        ##---------------------------------------------------- any kwargs left ?
        self.check_kwargs_left('Terminate', kwargs)

        ##----------------------------------------------- process RC if required
        if isinstance(RC,dict):
            phase = RC.pop('phase',0.)
            deg = RC.pop('deg', True)
            phasor = np.exp(1j * phase * (np.pi/180 if deg else 1.))
            
            units = RC.pop('units','MA').upper()
            if units == 'MA':
                Gamma = RC.pop('G',1.)
                RC = Gamma * phasor
            elif units == 'DB':
                Gamma = RC.pop('G',0.)
                RC = (10.**(Gamma/20.)) * phasor
            else:
                raise ValueError('circuit.Terminate: units must be one'
                                 ' of "dB" or "MA" : got %r' % units)
                
            self.check_kwargs_left('Terminate.RC', RC)
            
        ##------------------------------------------------------------- finalize

        if port1 in self.T :
            # port1 was already terminated : replace value
            i1, i2, neq = self.T[port1]

        else :
            # this is a new termination request for port1
            if port1 in self.C or port1 in self.E :
                print("Terminate : port '%s' already used" % port1)
                raise ValueError
            
            i1 = self.names.index(port1.replace('/','/A_'))
            i2 = self.names.index(port1.replace('/','/B_'))
            neq = self.M.shape[0]
            self.T[port1] = (i1, i2, neq)
            self.M = np.vstack((self.M, np.zeros((1,self.M.shape[1]))))
            self.V = np.vstack((self.V, np.zeros((1,1))))
            
        self.M[neq,i1], self.M[neq,i2]  = 1., - RC
        self.invM = None
        self.sol = None
        
#===============================================================================
#
#  C o n n e c t
#
    def Connect(self, *ports, **kwargs):
        """
        port : Blockname/Portname
        
        kwargs:
          
          S:
          
          info: 
        
        """
        N = len(ports)

        TLid = kwargs.pop('TLid',None)
        # check that if a TL is specified there are 2 ports
        if TLid and (N != 2):
            raise ValueError('circuit.Connect: A TL cannot connect more than 2 '
                             'ports (%d given)' % N)
        
        S = kwargs.pop('S', None)
        if S is None:
            # if S is not given use ideal junction
            S = np.array([[2/N - (1 if kc is kr else 0) for kc in range(N)]
                              for kr in range(N)])
        else:
            # check that S fits the number of ports given
            if np.array(S).shape != (N,N):
                raise ValueError('circuit.Connect: supplied array S does not'
                                 ' fit number of supplied ports')
        
        self.check_kwargs_left('Connect', kwargs)
            
        used = list(self.T)+list(self.E) # + list(self.C)
        idxA, idxB, old_idxB = [], [], []
        
        for p in ports:
            if p in used:
                raise ValueError("circuit.Connect : port '%s' already used" % p)
            if p in self.C:
                old_idxB.append(self.C[p])
            idxA.append(self.names.index(p.replace('/','/A_'))) # block/port -> block/A_port
            idxB.append(self.names.index(p.replace('/','/B_'))) # block/port -> block/B_port
            
        if old_idxB:
            raise NotImplementedError('circuit.Connect: replacing an existing connection')
            for iBs in old_idxB:
                # apparently some of the ports have been used in a connection
                # so we need to find if that is the connection that is being
                # changed or not
                if len(iBs[3]) is not len(idxB):
                    raise ValueError(
                        'circuit.Connect : modified connection topology')
                
                for old_iB, new_iB in zip(iBs[3], idxB):
                    if old_iB not in idxB:
                        raise ValueError(
                            'circuit.Connect : modified connection topology')
                    elif old_iB is not new_iB:
                        print('circuit.Connect : modified connection order')
                
    #             print('Connect')
    #             print(p)
    #             print(idxA)
    #             print(idxB)
    #             print()
    
        else:
            for kA, (iA, p) in enumerate(zip(idxA,ports)):
                MV = np.zeros((1,self.M.shape[1]))*1j
                MV[0,iA]  = -1.
                for kB, iB in enumerate(idxB):
                    MV[0,iB] = S[kA, kB] # 2/N - (1 if kA is kB else 0)
                self.M = np.vstack((self.M, MV))
                self.V = np.vstack((self.V, np.zeros((1,1))))
                self.C[p] = (self.V.shape[0]-1, iA, '', tuple(idxB), TLid)
            
        self.invM = None
        self.sol = None

#===============================================================================
#
#  C o n n e c t T L
#
    def ConnectTL(self, port1, port2, length, **kwargs):
        """
        port : Blockname/Portname

        kwargs:
        
        relpos : relative position of the first port [m] (default 0.)
        
        combination AZV (default)
        
        Z  : characteristic impedance (default self.Zbase [Ohm])
        A  : attenuation (default 0 [m-1])
        V  : wave velocity relative to speed of light in vacuum (default 1 ([1])
        
        combination GRLC
        
        G  : conductance (default 0 [Mho/m])
        R  : resistance (default 0 [Ohm/m])
        L  : inductance (default self.Zbase/c0 [H/m])
        C  capacitance (sdefault 1/(self.Zbase*c0) [F/m])
        
        TLid : id of the TL (default TL<3 digit counter>)
        
        plotargs : kwargs that can be passed to a plotting routine
        """
        
        ##-------------------------------------------------------- define the TL
        kwargs.update({'Portnames':[port1, port2]})
        SZ, TLid = self.MakeTL(length, kwargs)
        
#         print('SZ     :',SZ)
#         print('TLid   :',TLid)
#         print('kwargs :',kwargs)
        
        ##------------------------------anything left that should not be there ?
        self.check_kwargs_left('ConnectTL', kwargs)
        
        ##-------------------------------------------------- make the connection
        self.Connect(port1,port2,S=SZ.S,TLid =TLid)
    
#===============================================================================
#
#  C o n n e c t
#
    def Connect2(self, port1, port2, **kwargs):
        """
        
        ##
        ## This is the original Connect ...
        ##
        
        port : Blockname/Portname

        kwargs:
            SZ : Scatter structure of the TL 
            info : {'isTL': { ... parameters ...} | False (default)}
        """
        if False:        
            def connect(A, B):
                portA, portB = A.replace('/','/A_'), B.replace('/','/B_')
                i1, i2 = self.names.index(portA), self.names.index(portB)
                MV = np.zeros((1,self.M.shape[1]))
                MV[0,i1], MV[0,i2]  = 1., -1.
                self.M = np.vstack((self.M, MV))
                self.V = np.vstack((self.V, np.zeros((1,1))))
                self.C[A] = (self.V.shape[0]-1, i1, B, i2)
    
            if port1 in self.C or port1 in self.T or port1 in self.E:
                print("Connect : port '%s' already used" % port1)
                raise ValueError
    
            if port2 in self.C or port2 in self.T or port2 in self.E:
                print("Connect : port '%s' already used" % port1)
                raise ValueError
    
            connect(port1,port2)
            connect(port2,port1)
            
            self.invM = None
            self.sol = None
            
        else:
            self.Connect(port1,port2,**kwargs)
        
#===============================================================================
#
#  E x c i t e
#
    def Excite(self, port1, A1):
        """
        port : Blockname/Portname
        """
        
        if port1 in self.E :
            v1 = self.E[port1][0]
            self.V[v1,0:1] = A1
            
        else :
            if port1 in self.T or port1 in self.C :
                print("circuit.Excite : port '%s' already used" % port1)
                raise ValueError
                
            port1_A = port1.replace('/','/A_')
            i1 = self.names.index(port1_A)
            MV = np.zeros((1,self.M.shape[1]))
            MV[0,i1] = 1.
            self.M = np.vstack((self.M, MV))
            self.V = np.vstack((self.V, np.array([[A1+0j]]))) # cast to complex
            self.E[port1] = (self.V.shape[0]-1, i1)
            self.invM = None
            # self.eqns += [(self.E[port1], 'Excite %s' % port1)]
            
            #FIXED: the self.internal construct is somewhat dangerous: there can
            #       conflicts between blocks (even of different composition or
            #       intend). E.g. :
            #
            #       self.excite('Block1/Fdr',complex_value) will set
            #       'Block1/Fdr' --> self.internal['Fdr']
            #
            #       but self.excite('Block2/Fdr', complex_value) will cause
            #       'Block2/Fdr' --> self.internal['Fdr'] thus overwritting
            #       the first excite
            #
            #       this is probably OK if Block1 and Block2 are similar but
            #       might cause issues if Block1 is not at all related to Block2
            #       and both just happen to have a common portname.
            #
            #       it is also not clear what new information is stored as the
            #       data is not used by circuit_class3a and self.E containts the
            #       necessary information:
            #
            #       self.internal = dict((p.split('/')[1],p) for p in self.CT.E)
            
            ## self.internal[port1.split('/')[1]] =  port1
            
        self.sol = None
        
#===============================================================================
#
# S o l v e
#
    def Solve(self, excitations = None, printit = False):
        """
        excitations : if given a { Blockname/Portname : incident wave, ... }
        """
        if excitations :
            for port, wave in excitations.items() :
                self.Excite(port,wave)

        if self.invM is None : # only the excitations changed ?
            Neq, Nunknowns = self.M.shape
            # print(self.M.shape)
            if Neq == Nunknowns: # ??? "is" does not seem to work
                self.invM = np.linalg.inv(self.M) # No
                
            elif Neq < Nunknowns:
                raise ValueError(
                    'circuit_Class.Solve: following ports have not been'
                    ' connected or terminated or excited : %r' % 
                    self.find_orphans())
                
            else:
                raise ValueError('circuit_Class.Solve (internal error) :'
                                 'we have more equations than unknowns ??')
                
        self.S = np.dot(self.invM, self.V)
        
#         print('%r' % self.M)    # matrix
#         print('%r' % self.V)    # array
#         print('%r' % self.invM) # matrix
#         print('%r' % self.S)    # matrix
        
#         # old code
#         self.sol = {}
#         for name, value in zip(self.names, self.S) :
#             self.sol[name] = value[0,0]
#             if printit :
#                 i1 = name.find('/B_')
#                 if i1 >= 0 :
#                     blockname = name.replace('/B_','/')
#                     ss = '%-15s : ' % blockname
#                     A = self.sol[name.replace('/B_','/A_')]
#                     B = self.sol[name]
#                     V = A + B
#                     I = (A - B)/self.Zbase
#                     Z = V/I
#                     P = (np.abs(A)**2 - np.abs(B)**2)/(2*self.Zbase)
#                     ss += " A> %10.3f%+10.3fj" % (np.real(A),np.imag(A))
#                     ss += " B> %10.3f%+10.3fj" % (np.real(B),np.imag(B))
#                     ss += " V> %10.3f%+6.1fdeg" % (np.abs(V),np.angle(V, deg=1))
#                     ss += " I> %10.3f%+6.1fdeg" % (np.abs(I),np.angle(I, deg=1))
#                     ss += " Z> %10.3f%+10.3f" % (np.real(Z),np.imag(Z))
#                     ss += " P> %10.3f" % P
#                     print(ss)
#         if printit : print()

        # new code
        self.sol = dict([(name, self.S[k,0]) for k, name in enumerate(self.names)])
        if printit:
            for name in self.names:
                i1 = name.find('/B_')
                if i1 >= 0 :
                    blockname = name.replace('/B_','/')
                    ss = '%-15s : ' % blockname
                    A = self.sol[name.replace('/B_','/A_')]
                    B = self.sol[name]
                    V = A + B
                    I = (A - B)/self.Zbase
                    try:
                        Z = V/I
                    except FloatingPointError:
                        Z = np.Inf + 1j* np.Inf
                    P = (np.abs(A)**2 - np.abs(B)**2)/(2*self.Zbase)
                    ss += " A> %10.3f%+10.3fj" % (np.real(A),np.imag(A))
                    ss += " B> %10.3f%+10.3fj" % (np.real(B),np.imag(B))
                    ss += " V> %10.3f%+6.1fdeg" % (np.abs(V),np.angle(V, deg=1))
                    ss += " I> %10.3f%+6.1fdeg" % (np.abs(I),np.angle(I, deg=1))
                    ss += " Z> %10.3f%+10.3f" % (np.real(Z),np.imag(Z))
                    ss += " P> %10.3f" % P
                    print(ss)
            print()
        
        return self.sol
            
#===============================================================================
#
# _ v a l u e
#
    def _value(self, a, b, ttype):
        if ttype == 'V' :
            return a + b
        elif ttype == 'I' :
            return (a - b)/self.Zbase
        elif ttype == 'A' :
            return a
        elif ttype == 'B' :
            return b
        elif ttype == 'P' :
            return (np.abs(a)**2 - np.abs(b)**2)/(2*self.Zbase)
        elif ttype == 'P+' :
            return np.abs(a)**2/(2*self.Zbase)
        elif ttype == 'P-' :
            return np.abs(b)**2/(2*self.Zbase)
        elif ttype == 'G' :
            return b/a
        elif ttype == 'Z' :
            return self.Zbase * (a + b)/(a - b)
        elif ttype == 'z' :
            return (a + b)/(a - b)
        elif ttype == 'Y' :
            return (a - b)/(a + b)/self.Zbase
        elif ttype == 'y' :
            return (a - b)/(a + b)
        elif ttype == 'argV' :
            return np.angle(a + b)
        elif ttype == 'absV' :
            return np.abs(a + b)
        elif ttype == 'argI' :
            return np.angle(a - b)
        elif ttype == 'absI' :
            return np.abs(a - b)/self.Zbase
        elif ttype == 'VSWR':
            absG = np.abs(b/a)
            return (1 + absG)/(1 - absG)
        
        else :
            print("circuit_Class.Solution : unknown solution type '%s'" % ttype)
            raise ValueError
            
#===============================================================================
#
# S o l u t i o n
#
    def Solution(self, name, stype):
        """
        name  : Block/Port name
        stype : 'V'|'I'|'A'|'B'|'P'|'G'|'Z'|'Y'|'argV'|'absV'|'argI'|'absI'
                |'VSWR'
        """
        if self.sol is None : 
            self.Solve()
        
        try :
            A = self.sol[name.replace('/','/A_')]
            B = self.sol[name.replace('/','/B_')]
        except KeyError :
            print("circuit_Class.Solution : port '%s' not found" % name)
            raise

        if type(stype) is tuple or type(stype) is list :
            res = [self._value(A, B, ttype) for ttype in stype]
            if type(stype) is tuple :
                res = tuple(res)
        else :
            res = self._value(A, B, stype)

        return res

#===============================================================================
#
#  G e t _ Co e f f s
#
    def Get_Coeffs(self, E=None, D=None, **kwargs):
        """
        construct a dependency matrix between dependent quantities in a list 'D'
        and the exciter quantities in the list 'E'.

        depending item can be a tuple (name,type) or just the name (in which
        case the type is B by default)

        So, for a list of excited port names and the same list for the depending
        quantities the matrix returned will be the scattering matrix of the
        solved system.
        
        returns:
        ( the dependency matrix , list of
        
        #FIXME: does not seem to work ...
        """

        mtype = kwargs.get('mtype', 'S')
        Z0 = kwargs.get('Z0', self.Zbase)
        if 'Z0' in kwargs :
            print("circuit_Class.Get_Coeffs() : Z0 argument is not implemented")
        
        if isinstance(mtype,tuple):
            deftypeE, deftypeD = mtype
            
        else:
            try :
                deftypeE, deftypeD = {'S' : ('A','B'),
                                      'Z' : ('I','V'),
                                      'Y' : ('V','I'),
                                      'VA' : ('A','V'),
                                      'IA' : ('A','I')}[mtype]
            except KeyError :
                print("Unknown matrix type (should be 'S', 'Z' or 'Y' : " +
                      "got '%s')" % mtype)
                raise ValueError
        
        #---------------------------------------------------------------- check1

        def check1(alist, deftype, allowed, ttype):
            """ check requested ports in alist are in allowed; if not tuple set
                type to deftype
            """
            def check2(aname):
                anameA = aname.replace('/','/A_')
                anameB = aname.replace('/','/B_')
                if (anameA not in allowed) and (aname not in allowed) :
                    if ttype == 'E' :
                        print("circuit_Class.Get_Coeffs() : only excited ports",
                              "can be used for the independent coefficients")
                    else :
                        print("circuit_Class.Get_Coeffs() : unknown port '%s'" %
                              aname)
                    raise ValueError
                idxE = self.E[aname][0] if ttype == 'E' else -1
                return self.names.index(anameA), self.names.index(anameB), idxE
            
            tlist = []
            for k, itm in enumerate(alist):
                if type(itm) is tuple :
                    if len(itm) is 2 :
                        idxA, idxB, idxE = check2(itm[0])                            
                        if itm[1] not in ['V','I','A','B']:
                            print("circuit_Class.Get_Coeffs() : only types 'A', ",
                              "'B', 'V' or 'I' allowed (got '%s')" % str(itm[1]))
                            raise ValueError
                    else :
                        print("circuit_Class.Get_Coeffs() : only list of portname",
                          "or tuple(portname,type) allowed (got '%s')" % str(itm))
                        raise ValueError
                    tlist.append(itm + (idxA, idxB, idxE))
                else :
                    idxA, idxB, idxE = check2(itm)
                    tlist.append((itm, deftype, idxA, idxB, idxE))

                tlist[-1] += (
                    {'A':1.,'B':0.,'V':1.,'I': 1./self.Zbase}[tlist[-1][1]],
                    {'A':0.,'B':1.,'V':1.,'I':-1./self.Zbase}[tlist[-1][1]])

            return tlist
                    
        #.......................................................................

        if E is None :
            E = [(port,deftypeE) for port in self.E]
        E = check1(E,deftypeE,self.E,'E')

        if D is None :
            D = [(port,deftypeD)
                 for port, typ, idxA, idxB, idxE, cfA, cfB in E]
        D = check1(D,deftypeD,self.names,'D')

        if self.invM is None : self.invM = np.linalg.inv(self.M)

        M  = 1j * np.zeros((len(D),len(E)))
        S  = 1j * np.zeros((len(E),len(E)))
        ME = 1j * np.zeros((len(E),len(E)))
        for ke, e in enumerate(E) :
            for kd, d in enumerate(D) :
                M[kd,ke] = (d[5] * self.invM[d[2],e[4]] +
                            d[6] * self.invM[d[3],e[4]])
            for ke2, e2 in enumerate(E) :
                S[ke2,ke] = self.invM[e2[3],e[4]]

        for ke, e in enumerate(E):
            ME[ke,:] = e[5] * np.eye(len(E))[ke,:] + e[6] * S[ke,:]

        try :
            invME = np.linalg.inv(ME)
        except : # the only thing here could be a LinAlgError
            print("circuit_Class.Get_Coeffs() :",
                  "Excitations vector basis leads to a singular matrix")
            raise

        return (np.dot(M, invME),
                [(d[0],d[1]) for d in D],
                [(e[0],e[1]) for e in E])

#===============================================================================
#
#  G e t _ Co e f f s 1
#
    def Get_Coeffs1(self, E=None, D=None, **kwargs):
        """
        construct a dependency matrix between dependent quantities in a list 'D'
        and the exciter quantities in the list 'E'.

        depending item can be a tuple (name,type) or just the name (in which
        case the type is B by default)

        So, for a list of excited port names and the same list for the depending
        quantities the matrix returned will be the scattering matrix of the
        solved system.
        
        returns:
        the dependency matrix , list of signals D and E
        
        New version as we have some doubts on fhe original version. We also lift
        the restriction on the exciting ports ...
        
        D and E are lists of tuples [(block/port, sigtype), ...]
        
        sigtype : 'A' | 'B' | 'V' | 'I'
        
        """
        pass
    
        
    
#===============================================================================
#
#  G e t _ Z 
#
    def Get_Z(self, name):
        try :
            (r0, c0) = self.blocks[name]['idx0']
            Nports = self.blocks[name]['Nports']
        except KeyError :
            print("circuit.Get_Z : block '%s' not found" % name)
            raise
        S = self.M[r0:r0+Nports,c0:c0+Nports]
        try:
            Z = self.Zbase * np.dot(np.linalg.inv(np.eye(Nports) - S),
                                    (np.eye(Nports) + S))
        except np.linalg.linalg.LinAlgError:
            Z = np.zeros(S.shape)
        return Z
        
#===============================================================================
#
#  G e t _ Y 
#
    def Get_Y(self, name):
        try :
            (r0, c0) = self.blocks[name]['idx0']
            Nports = self.blocks[name]['Nports']
        except KeyError :
            print("circuit.Get_Z : block '%s' not found" % name)
            raise
        S = self.M[r0:r0+Nports,c0:c0+Nports]
        Y = np.dot(np.linalg.inv(np.eye(Nports) + S),
                   (np.eye(Nports) - S))/self.Zbase
        return Y
        
#===============================================================================
#
#  G e t _ S
# 
    def Get_S(self, name, **kwargs):
        Z0 = kwargs.get('Z0', self.Zbase)
        try :
            (r0, c0) = self.blocks[name]['idx0']
            Nports = self.blocks[name]['Nports']
        except KeyError :
            print("circuit.Get_Z : block '%s' not found" % name)
            raise
        S = self.M[r0:r0+Nports,c0:c0+Nports]
        if Z0 != self.Zbase :
            p = (self.Zbase/Z0 + 1.)/2.
            m = p - 1.
            S = np.dot(np.linalg.inv(p * np.eye(Nports) + m * S),
                       (m * np.eye(Nports) + p * S))

        return S
            
#===============================================================================
#
#  G e t _ S Z 
#
    def Get_SZ(self, name, **kwargs):
        Z0 = kwargs.get('Z0', self.Zbase)
        # Note that for the moment the circuit solver does not care about the
        # frequency
        fMHz = kwargs.get('fMHz', 0.)
        return sc.Scatter(fMHz= fMHz,
                          Zbase= Z0,
                          S= self.Get_S(name, Z0= Z0),
                          Portnames=self.blocks[name]['names'])
    
#===============================================================================
#
#  d u m p 2 p y
#
    def dump2py(self, fname='matrices.py', solvit2=False, overwrite=False):
        """
        dump all the matrices and results in to a python script that can be
        reloaded (imported)
        """
        
        cfname = os.path.realpath(fname +
                                  ('' if fname[-3:] == '.py' else '.py'))
        print(cfname)
        if os.path.exists(cfname) and not overwrite :
            print("%s already exists in working directory :\n%s" %
                  (fname, os.path.dirname(cfname)))
            raise ValueError
        
        fout = open(cfname,'w')
        fout.write(self.dump2str(solvit2=False))
        fout.close()

#===============================================================================
#
#  d u m p 2 s t r
#
    def dump2str(self, solvit2=False):
        """
        dump all the matrices and results in to a python script that can be
        reloaded (imported)
        """
        s = ('# [circuit.dump2py()]\n'
             '# to use the data in this file add following line without the'
             ' leading # to your python script :\n'
             '# from <this file name> import Zref, M, C, T, E, S\n\n'
             '# dict object "M" contains all matrices :\n'
             '# M["matrix-name"]["Z"] : the Z-matrix coefficients\n'
             '# M["matrix-name"]["S"] : the S-matrix coefficients\n'
             '# M["matrix-name"]["P"] : the port names\n\n'
             'M = {}\n\n'
             'Zref = %0.4e' % self.Zbase +
             '# [Ohm] reference impedance for the S matrices\n\n')

        for bname, block in self.blocks.items():
            # print(bname, block)
            s += '# Matrix : %s\n' % bname
            s += 'M["%s"] = {}\n' % bname
            s += 'M["%s"]["P"] = %s \n' % (bname, block["names"])
            s += 'M["%s"]["Z"] = [\n' % bname
            Z = self.Get_Z(bname)
            for k1 in range(block['Nports']):
                s += '    ['
                for k2 in range(block['Nports']):
                    s += '%+0.4e%+0.4ej,'% (np.real(Z[k1,k2]),np.imag(Z[k1,k2]))
                s = s[:-1] + '],\n' # remove the trailing ,
            s += '   ]\n'

            s += 'M["%s"]["S"] = [\n' % bname
            S = self.Get_S(bname)
            for k1 in range(block['Nports']):
                s += '    ['
                for k2 in range(block['Nports']):
                    s += '%+0.4e%+0.4ej,'% (np.real(S[k1,k2]),np.imag(S[k1,k2]))
                s = s[:-1] + '],\n' # remove the trailing ,
            s += '   ]\n'
            s += '\n'

        s += ('# the dict "C" contains the connection list for the ports of'
               ' the matrices above\n'
              '# C["matrix1/port1"] = "matrix2/port2"\n'
              '# \n'
              'C = {}\n')
        for c1, c2 in self.C.items() :
            if c2[2]: # TODO: need to coorect for the new way of connecting ports
                pass
            s += 'C["%s"] = "%s"\n' % (c1, c2[2])
        s += '\n'
                       
        s += ('# the dict "T" contains the list of the terminated ports of the'
               ' matrices above\n'
              '# T["matrix1/port1"] = (Z, gamma) (gamma = reflection coeff when'
               ' using S-matrices)\n'
              '# \n'
              'T = {}\n')
        for t1, t2 in self.T.items() :
            RC = - self.M[t2[2],t2[1]]
            Z = self.Zbase * (1 + RC) / (1  -RC) if RC != 1. else np.Infinity
            s += ('T["%s"] = (%+0.4e%+0.4ej,%+0.4e%+0.4ej)\n' %
                       (t1, np.real(Z), np.imag(Z), np.real(RC), np.imag(RC)))
        s += '\n'
                       
        s += ('# the dict "E" contains the list of the excited ports of the'
               ' matrices above\n'
              '# E["matrix1/port1"] = (V, I, Vfwd) (Vfwd = port forward voltage'
               ' when using S-matrices)\n'
              '# \n'
              'E = {}\n')
        for e1, e2 in self.E.items() :
            Vfwd = self.V[e2[0]]
            Vp, Ip, Ap = self.Solution(e1,('V','I','A'))
            s += ('E["%s"] = (%+0.4e%+0.4ej,%+0.4e%+0.4ej,%+0.4e%+0.4ej)\n' %
                  (e1,np.real(Vp),np.imag(Vp),
                      np.real(Ip),np.imag(Ip),
                      np.real(Ap),np.imag(Ap)))
        s += '\n'

        if solvit2 and not self.sol :
            self.Solve()

        if self.sol :
            s += ('# the dict "S" contains the V and I at the ports of the'
                   ' matrices above\n'
                  '# S["matrix1/port1"] = (V, I) \n'
                  '# \n'
                  'S = {}\n')
            for name, sol in self.sol.items() :
                i1 = name.find('/B_')
                if i1 >= 0 :
                    blockname = name.replace('/B_','/')
                    s += '\n# %-15s : ' % blockname
                    A = self.sol[name.replace('/B_','/A_')]
                    B = self.sol[name]
                    V = A + B
                    I = (A - B)/self.Zbase
                    Z = V/I
                    P = (np.abs(A)**2 - np.abs(B)**2)/(2*self.Zbase)
                    s += " A> %12.6f%+12.6fj" % (np.real(A),np.imag(A))
                    s += " B> %12.6f%+12.6fj" % (np.real(B),np.imag(B))
                    s += " V> %12.4f%+6.1fdeg" % (np.abs(V),np.angle(V, deg=1))
                    s += " I> %12.6f%+6.1fdeg" % (np.abs(I),np.angle(I, deg=1))
                    s += " Z> %12.6g%+12.6g" % (np.real(Z),np.imag(Z))
                    s += " P> %12.6f" % P
                    s += '\n'
                    s += ('S["%s"] = (%+0.4e%+0.4ej,%+0.4e%+0.4ej)\n' %
                          (blockname, np.real(V), np.imag(V),
                                      np.real(I), np.imag(I)))

        return s

#===============================================================================
#
# G T L d a t a
#
    def GTLdata(self):
        """see TLdata but knows how to handle also GTL blocks (TLids)
        """
        result = {}
        for TLid, TLinfo in self.TLs.items():
            if isinstance(TLinfo['isGTL'], General_TL):
                blockname = TLinfo['blockname']
                tport = ((blockname + '/') if blockname else '') + TLinfo['ports'][0]
                VI = self.Solution(tport,['V','I'])
                result[TLid] = TLinfo['isGTL'].fullsolution(V0=VI[0],I0=-VI[1])
#                 if TLid == 'atlis':
#                     PP.pprint(result[TLid])
                
        return result
        
#===============================================================================
#
# T L d a t a
#
    def TLdata(self, dtype, **kwargs):
        """
            returns a dict of the defined TL segments, e.g. :
            
            {'TL001': (x, result), ... }
            
            x is a list of positions and result mimics dtype
            
            dtype: str | [str, ... ] | (str, ... ) | {str:None, ... }
                   where str is one of 'V[SW]', 'I[SW]', 'P[WR]', 'L[OSS]'
            
            TLdata('VSW') 
            -> {'TL001': ([x0, ... , xn], [V0, ... , Vn]), ... }
            
            TLdata(['V','I']) 
            -> {'TL001': ([x0, ... , xn], [[V0, ... Vn], [I0, ... In]]), ... }
            
            TLdata({'V':None, 'P':None}] 
            -> {'TL001': ([x0, ... , xn], {'V':[V0, ... Vn], 'P':[P0, ... Pn]})}
            
            kwargs
            
            dx : see AddBlockTL, MakeTL - this will override what was set
        """
        
        def SWfunctions(tTLinfo):
                        
            # get the TL properties
            Zc, gamma = tTLinfo['Zc'], tTLinfo['gamma']
            x0, L = tTLinfo['relpos'], tTLinfo['length']
            dx = dx1 if dx1 is not None else  tTLinfo['dx']
            gmx0 = gamma * x0
            
            # get the 1st port's voltage and current
            blockname = tTLinfo['blockname']
            tport = ((blockname + '/') if blockname else '') + tTLinfo['ports'][0]
            VI = self.Solution(tport,['V','I'])
            
            # find the coefficients for the propagation of the waves
            #   in x = refpos
            #
            #  V(x) = A exp(gamma x) + B exp(- gamma x)
            #  Zc I(x) = A exp(gamma x) - B exp(-gamma x)
            
            sI = 1. if blockname else -1. # connections use the block ports
                        
            AexpGxr = (VI[0] + sI * Zc * VI[1]) / np.exp(-gmx0) / 2
            BexpGxr = (VI[0] - sI * Zc * VI[1]) / np.exp(+gmx0) / 2

            if dx is 0:
                xs = x0, x0+L # the interval on which the function is defined
                
            elif isinstance(dx, int):
                if dx > 0:
                    xs = np.linspace(x0, x0+L, num=dx)
                else: 
                    raise ValueError('circuit.TLfata.TLdata1: dx is int < 0 !')
                
            elif isinstance(dx, float):
                N = int(L // np.abs(dx))
                if dx > 0:
                    xs = np.linspace(x0, x0 + N * dx, num=N+1)
                else:
                    xs = np.linspace(x0, x0 + L, num=N+1)
                    
            # define the functions
            
            V = lambda x, A=AexpGxr, B=BexpGxr, g=gamma: (
                    A*np.exp(-g*x) + B*np.exp(+g*x)
                )
            I = lambda x, A=AexpGxr, B=BexpGxr, g=gamma, z=Zc: (
                    (A*np.exp(-g*x) - B*np.exp(+g*x))/z
                )
            P = lambda x, u=V, i=I: 0.5 * np.real(u(x)*np.conj(i(x)))
            
            dP = lambda x, u=V, i=I, z=Zc:(
                    0.5 * np.real(gamma*(np.abs(i(x))**2*z + np.abs(u(x))**2/z))
                )
            
            return xs, dx, V, I, P, dP
        
        #- ------------------------------------------------------------------- -
        
        if 'dx' in kwargs:
            dx1 = kwargs.pop('dx',-300/self.fMHz/72)
        else:
            dx1 = None
            
        # check for unknown kwargs    
        self.check_kwargs_left('TLdata', kwargs)
        
        # solve if not done so already
        if not self.sol:
            self.Solve()
            
        result = {}

        for TLid, TLinfo in self.TLs.items():
            if TLinfo['isGTL'] is False:
    #             xs, dx, V, I, P, dP = SWfunctions(TLinfo)
                # get the TL properties
                Zc, gamma = TLinfo['Zc'], TLinfo['gamma']
                x0, L = TLinfo['relpos'], TLinfo['length']
                dx = dx1 if dx1 is not None else  TLinfo['dx']
                gmx0 = gamma * x0
                 
                # get the 1st port's voltage and current
                blockname = TLinfo['blockname']
                tport = ((blockname + '/') if blockname else '') + TLinfo['ports'][0]
                VI = self.Solution(tport,['V','I'])
                 
                # find the coefficients for the propagation of the waves
                #   in x = refpos
                #
                #  V(x) = A exp(gamma x) + B exp(- gamma x)
                #  Zc I(x) = A exp(gamma x) - B exp(-gamma x)
                 
                sI = 1. if blockname else -1. # connections use the block ports
                 
                # find the power flow direction
                sP = 1. if 0.5 * np.real(VI[0] * np.conj(sI*VI[1])) > 0. else -1.
                 
                AexpGxr = (VI[0] + sI * Zc * VI[1]) / np.exp(-gmx0) / 2
                BexpGxr = (VI[0] - sI * Zc * VI[1]) / np.exp(+gmx0) / 2
     
                if dx is 0:
                    xs = x0, x0+L # the interval on which the function is defined
                     
                elif isinstance(dx, int):
                    if dx > 0:
                        xs = np.linspace(x0, x0+L, num=dx)
                    else: 
                        raise ValueError('circuit.TLfata.TLdata1: dx is int < 0 !')
                     
                elif isinstance(dx, float):
                    N = int(L // np.abs(dx))
                    if dx > 0:
                        xs = np.linspace(x0, x0 + N * dx, num=N+1)
                    else:
                        xs = np.linspace(x0, x0 + L, num=N+1)
                         
                # define the functions
                 
                V = lambda x, A=AexpGxr, B=BexpGxr, g=gamma: (
                        A*np.exp(-g*x) + B*np.exp(+g*x)
                    )
                I = lambda x, A=AexpGxr, B=BexpGxr, g=gamma, z=Zc: (
                        (A*np.exp(-g*x) - B*np.exp(+g*x))/z
                    )
                P = lambda x, u=V, i=I: 0.5 * np.real(u(x)*np.conj(i(x)))
                 
                dP = lambda x, u=V, i=I, z=Zc:(
                        0.5 * np.real(gamma*(np.abs(i(x))**2*z + np.abs(u(x))**2/z))
                    )
             
                if isinstance(dtype,str):
                    DTYPE = dtype.upper()
                    if dx is 0:
                        if DTYPE in ['V', 'VSW']:
                            result[TLid] = (xs, V)
                        elif DTYPE in ['I','ISW']:
                            result[TLid] = (xs, I)
                        elif DTYPE in ['P', 'PWR']:
                            result[TLid] = (xs, P)
                        elif DTYPE in ['DP', 'PLOSS']:
                            result[TLid] = (xs,dP)
                    else:
                        if DTYPE in ['V', 'VSW']:
                            result[TLid] = (xs, V(xs))
                        elif DTYPE in ['I','ISW']:
                            result[TLid] = (xs, I(xs))
                        elif DTYPE in ['P', 'PWR']:
                            result[TLid] = (xs, P(xs))
                        elif DTYPE in ['DP', 'PLOSS']:
                            result[TLid] = (xs,dP(xs))
                        
                elif isinstance(dtype,(list,tuple,dict)):
                    pass
                
                else:
                    raise ValueError('circuit.TLdata: dtype must be str, list, tuple or'
                                     ' dict')
        
        return result

#===============================================================================
#
# p r o c e s s G T L   
#
def processGTL1(fMHz, Zbase, GTLdata):
    
    #---------------------------------------------------- r e s o l v e V a  r s   
    #
    def resolveVars(kwargs, variables):
        ekwargs = {}
        for kw, val in kwargs.items():
            if isinstance(val, str):
                ekwargs[kw] = eval(val, globals(), variables)
            else:
                ekwargs[kw] = val
        return ekwargs
    #
    #- ----------------------------------------------------------------------- -
    
    CT = circuit(fMHz=fMHz, Zbase=Zbase)
    variables = GTLdata.get('variables',{})
    Portnames = GTLdata.get('Portnames', [])
    
    #--------------------------------------------------------- resolve variables
    found = True # prime the while loop
    while found:
        found = {}
        for var, val in variables.items():
            if isinstance(val, str):
                try:
                    found[var] = eval(val, globals(), variables)
                except (NameError, SyntaxError, TypeError):
                    pass # probably could not resolve yet
        if found:
            variables.update(found)
    
    #----------------------------------------------------- iterate over GTL list
    for TL, GTLkwargs in GTLdata['GTL'].items():                
        CT.AddBlockGTL(TL, **resolveVars(GTLkwargs, variables))
        
    #TODO: auto process terminations when a portname appears to be a dict 
    #TODO: auto process rel positions from pornames and lengths

    #---------------------------------------------------- find all the portnames
    portnames = {}
    for TLid, TLkws in CT.TLs.items():
        for port in TLkws['ports']:
            if port not in portnames:
                portnames[port] = []
            portnames[port].append(TLid)
    
    internal, external, orphans = {}, {}, []
    for port, TLids in portnames.items():
        if port == '[sc]':
            # terminate the short circuit(s)
            for TLid in TLids:
                CT.Terminate(TLid+'/[sc]',Z=0)
        
        elif port == '[oc]':
            for TLid in TLids:
                CT.Terminate(TLid+'/[oc]',Y=0)
           
        elif len(TLids) > 1:
            # connect multiples
            CT.Connect(*(TLid+'/'+port for TLid in TLids))
                            
        else:
            # excite the remaining orphan ports  (needed to extract the 
            # S-matrix) and keep track of the internal/external portnames
            orphans.append(TLids[0]+'/'+port)
            CT.Excite(orphans[-1], 0.)
            internal[port] = orphans[-1]
            external[orphans[-1]] = port
                    
    #----------------------------------------------------- extract the SZ matrix
    S = CT.Get_Coeffs(orphans, orphans)
    SZ =  sc.Scatter(fMHz=fMHz, Zbase=Zbase, 
                      Portnames=[external[p] for p in orphans], 
                      S=S[0])
    
    #------------------------------- check Portnames match and sort if available
    if Portnames:
        # verify
        if len(Portnames) != len(orphans) or any(
                     [internal[p] not in orphans for p in Portnames]):
            warnings.warn('Portnames do not match')
        else:
            SZ.sortports(Portnames)
            
    return CT, SZ

#===============================================================================
#
# p r o c e s s G T L   
#
def processGTL(fMHz, Zbase, GTL_data, **kwargs):
    """
    a GTL data structure is as follows:
    
    {'Portnames': ordered list of port names (if a sNp touchstone file  is 
                  provided the order must match),
                  
     'sNp'      : optional touchstone file, # *** future extension ***
     
     'relpos'   : {'p1':position, ... } where the p1 are some or all of the 
                  portnames,
     
     'variables': { dict of substitutions },
     
     'defaults' : { dict of default AddBlockGTL kwargs }, # *** future extension ***
     
     'GTL       : {
         'TLid1': { # AddBlockGTL kwargs General TL section 1
         
             length: float
             
             Portnames: [ list of 2 portnames ], # portname = 
                                                 #     str | (str, dict) | dict
                                                 # dict = CT.Terminate kwargs
                                                 
             relpos: float # optional: position of 1st port
             
             ... General_TL kwargs of the TL properties ...
         
         }, # TLid1
         
         ...
         
         'TLidN': { AddBlockGTL kwargs General TL section N 
         
             ... TLid 1 ...
         }, # TLidN
         
     }, # GTL
    }
    """
    
    #---------------------------------------------------- r e s o l v e V a  r s   
    #
    def resolveVars(kwargs, variables):
        ekwargs = {}
        for kw, val in kwargs.items():
            if isinstance(val, str):
                ekwargs[kw] = eval(val, globals(), variables)
            
            else:
                ekwargs[kw] = val

        return ekwargs
    #
    #- ----------------------------------------------------------------------- -
    
    CT = circuit(fMHz=fMHz, Zbase=Zbase)
    variables = {**GTL_data.get('variables',{}), **kwargs.get('variables',{})}
    Portnames = GTL_data.get('Portnames', [])
    relpos = GTL_data.get('relpos', dict([(p, 0.) for p in Portnames[:-1]]))
    
    #--------------------------------------------------------- resolve variables
    found = True # prime the while loop
    while found:
        found = {}
        for var, val in variables.items():
            if isinstance(val, str):
                try:
                    found[var] = eval(val, globals(), variables)
                except (NameError, SyntaxError, TypeError):
                    pass # probably could not resolve yet
        if found:
            variables.update(found)    
            
    #------------------------------------------- check for expressions in relpos
    
    for p, v in relpos.items():
        if isinstance(v,str):
            relpos[p] = eval(v, globals(), variables)
    
    #----------------------------------------------------- iterate over GTL list
    GTLs = {}
    for TL, GTLkwargs in GTL_data['GTL'].items():
        
        GTLkws =    resolveVars(GTLkwargs, variables)
        terminations = []
        for k, port in enumerate(GTLkws['Portnames']):
            
            # detect if a port is terminated or not
            if isinstance(port, dict):
                # auto generated portname (we not need it later on)
                GTLkws['Portnames'][k] = '<T%05d>' % CT.newID()
                terminations.append((GTLkws['Portnames'][k], port))
                
            elif isinstance(port, tuple):
                # we are interested to keep the portname
                GTLkws['Portnames'][k] = port[0]
                terminations.append(port)
            
            # substitute variables/expressions
            for k, (port, Tkws) in enumerate(terminations):
                for kw, val in Tkws.items():
                    if isinstance(val, str):
                        Tkws[kw] = eval(val, globals(), variables)
                                
        GTLs[TL] = [GTLkws, terminations]
    
    #------------------------------------ try and resolve the relative positions
        
    found = True
    while found:
        found = False
        for TL in GTLs:
            for k, port in enumerate(GTLs[TL][0]['Portnames']):
                if 'relpos' not in GTLs[TL][0]:
                    if port in relpos:
                        if k: # the 2nd port: thus the relpos is moved back
                            GTLs[TL][0]['relpos'] = relpos[port] - GTLs[TL][0]['length']
                        else:
                            GTLs[TL][0]['relpos'] = relpos[port]
                        # print('%s : %.4f m' % (TL, GTLs[TL][0]['relpos']))
                        found = True
                    
                    else:
                        # we don't have a port position yet 
                        otherport = GTLs[TL][0]['Portnames'][1-k]
                        if otherport in relpos:
                            # but we got the other port
                            if k: # the 2nd port: thus otherport is port 1
                                relpos[port] = relpos[otherport] + GTLs[TL][0]['length']
                                GTLs[TL][0]['relpos'] = relpos[otherport]
                            else: # the 1st port thus other port is port 2
                                relpos[port] = relpos[otherport] - GTLs[TL][0]['length']
                                GTLs[TL][0]['relpos'] = relpos[port]
                            # print('port %s : %.4f m' % (port, relpos[port]))
                            # print('TL %s : %.4f m' % (TL, GTLs[TL][0]['relpos']))
                            found = True
                        # else: cannot resolve further
                else:
                    # check to see if this resolves furtur segments
                    if port not in relpos:
                        if k: # 2nd port
                            relpos[port] = GTLs[TL][0]['relpos'] + GTLs[TL][0]['length']
                        else:
                            relpos[port] = GTLs[TL][0]['relpos']
                        found = True
                        # print('port %s : %.4f m' % (port, relpos[port]))
                        
        # PP.pprint(relpos)
        
    #------------------------------ iterate over GTL again with the final GTLkws
    for TL, (GTLkws, terminations) in GTLs.items():
        
        CT.AddBlockGTL(TL, **GTLkws)
        
        for port, termination in terminations:
            CT.Terminate(TL + '/' + port, **termination)
                    
    #TODO: auto process rel positions from portnames and lengths

    #---------------------------------------------------- find all the portnames
    portnames = {}
    for TLid, TLkws in CT.TLs.items():
        for port in TLkws['ports']:
            if port not in portnames:
                portnames[port] = []
            portnames[port].append(TLid)
    
    internal, external, orphans = {}, {}, []
    for port, TLids in portnames.items():
                    
        if any([(TLid+'/'+port in CT.T) for TLid in TLids]):
            pass # already terminated
        
        elif port == '[sc]':
            # terminate the short circuit(s)
            for TLid in TLids:
                CT.Terminate(TLid+'/[sc]',Z=0)
        
        elif port == '[oc]':
            for TLid in TLids:
                CT.Terminate(TLid+'/[oc]',Y=0)
           
        elif len(TLids) > 1:
            # connect multiples
            CT.Connect(*(TLid+'/'+port for TLid in TLids))
                            
        else:
            # excite the remaining orphan ports  (needed to extract the 
            # S-matrix) and keep track of the internal/external portnames
            orphans.append(TLids[0]+'/'+port)
            CT.Excite(orphans[-1], 0.)
            internal[port] = orphans[-1]
            external[orphans[-1]] = port
                    
    #----------------------------------------------------- extract the SZ matrix
    S = CT.Get_Coeffs(orphans, orphans)
    SZ =  sc.Scatter(fMHz=fMHz, Zbase=Zbase, 
                      Portnames=[external[p] for p in orphans], 
                      S=S[0])
    
    #------------------------------- check Portnames match and sort if available
    if Portnames:
        # verify
        if len(Portnames) != len(orphans) or any(
                     [internal[p] not in orphans for p in Portnames]):
            warnings.warn('Portnames do not match')
        else:
            SZ.sortports(Portnames)
            
    return CT, SZ

#===============================================================================
#
# self testing code
#

if __name__ == "__main__" :
    
    from pyRFtk import scatter3a as _sc

    if True:
        print('test processGTL')
        
        GTL = {'variables': {'rho' : 0., # 0.02214,
                             'L1' : 1.0,
                             'L2' : 'L1 + 2*L3',
                             'L3' : '7',
                             'Zsc'  : -1,
                            },
               'Portnames': ['in', 'out'],
               'relpos' : {
                   'T': 1.00
               },
               'GTL': {
                   'TL1': {'Portnames': ['in','T'],
                           'length': 1.00,
                           'rho': 'rho',
                           'OD': 185.,
                           'Z0TL': 20.,
                           'plotkws': {'color':'r'},
                   },
                   'TL2': {'Portnames': ['T','out'],
                           'length': 1.40,
                           'rho': 'rho',
                           'OD': 185.,
                           'Z0TL': 20.,
                           'plotkws': {'color':'b'},
                   },
                   'TL3': {'Portnames': ['T',('sc',{'Z':'Zsc + 1'})],
                           'length': 1.60,
                           'rho': 'rho',
                           'OD': 185.,
                           'Z0TL': 20.,
                           'plotkws': {'color':'g'},
                   },
               },
        }
        
        GTL = {# 'sNp' : 'some touchstone file',
               'Portnames' : ['ss', 'fdr'], # used for the sNp and should match the GTL
               
                'variables' : {
                    'RHO'   : 0., # 0.02214,          # uOhm.m (Cu at 100 degC)
                    'DX'    : -0.01,            # m
                    'SSPOS' : 1.326,            # m
                    'SSFDR' : 0.200,            # m
                    'L1'    : '0.4082 - 1.226 - SSPOS', # m compensate for SSPOS
                    'ZTL1'  : 22.,              # Ohm
                    'L2'    : 0.2568,           # m
                    'ID2'   : [265,140],        # mm
                    'ZTL2'  : 18.,              # Ohm
                    'L3'    : 0.4123,           # m
                    'ID3'   : 140.,             # mm
                    'ZTL3'  : 20.,              # Ohm
                    'VCWPS' : 'SSPOS+L1+L2+L3', # m
                    'LMTL'  : 2.0               # m
               },
               
#               'defaults': {
#                  'rho':'RHO', 
#                  'vr':1.0,
#                  'plotkws':{'color':'b', 'ls':'-', 'lw':1}
#              }, # defaults *** future extension

               'relpos': {
                   'ss':'SSPOS + SSFDR',
               },
               
               'GTL' : {
                   'V3': {
                           'Portnames': ['ss','cone']  , 
                           'length': 'L1 - SSFDR', 
                         # 'OD': 185.0,
                        'Z0TL': 'ZTL1',  # Ohm   note: qTL's only exact for 21 Ohm
                        'qTLI': 1.489,   # /m    note: the vtl outer is not round -> qTLc = rTLc/Rsc
                        'qTLO': 1.227,   # /m          takes into account the current distribution
                           'vr': 1.0000, 
                           'rho': 'RHO', 
                           'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 
                           'dx':'DX'
                   },
                   'V4': {
                           'Portnames': ['cone','v5']  , 
                           'length': 'L2', 
                         'ID': 'ID2',
                        'Z0TL': 'ZTL2',
                           'vr': 1.0000, 
                           'rho': 'RHO', 
                           'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 
                           'dx':'DX'
                   },
                   'V5': {
                           'Portnames': ['v5','vcw1']  , 
                           'length': 'L3', 
                        'ID': 'ID3',
                        'Z0TL': 'ZTL3',
                           'vr': 1.0000, 
                           'rho': 'RHO', 
                           'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 
                           'dx':'DX'
                   },
                   
                   'vcw2a' : {'Portnames': ['vcw1','vcw2'], 'length': 0.0310, 'OD': 185.0, 'Z0TL': 20.5032, 'vr': 0.7838, 'rho': 'RHO', 'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 'dx':'DX'},      
                   'vcw2b' : {'Portnames': ['vcw2','vcw3'], 'length': 0.0310, 'OD': 185.0, 'Z0TL': 21.1408, 'vr': 0.7731, 'rho': 'RHO', 'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 'dx':'DX'},      
                   'vcw2c' : {'Portnames': ['vcw3','vcw4'], 'length': 0.0310, 'OD': 185.0, 'Z0TL': 25.0218, 'vr': 0.7869, 'rho': 'RHO', 'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 'dx':'DX'},
                   'vcw2d' : {'Portnames': ['vcw4','vcw5'], 'length': 0.0350, 'OD': 185.0, 'Z0TL': 20.00  , 'vr': 1.0000, 'rho': 'RHO', 'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 'dx':'DX'},
                   'vcw2e' : {'Portnames': ['vcw5','vcw6'], 'length': 0.0310, 'OD': 185.0, 'Z0TL': 25.0218, 'vr': 0.7869, 'rho': 'RHO', 'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 'dx':'DX'},
                   'vcw2f' : {'Portnames': ['vcw6','vcw7'], 'length': 0.0310, 'OD': 185.0, 'Z0TL': 21.1408, 'vr': 0.7731, 'rho': 'RHO', 'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 'dx':'DX'},      
                   'vcw2g' : {'Portnames': ['vcw7','v6']  , 'length': 0.0310, 'OD': 185.0, 'Z0TL': 20.5032, 'vr': 0.7838, 'rho': 'RHO', 'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 'dx':'DX'},      
                   
                   'V6'    : {'Portnames': ['v6','fdr']   , 'length': 0.1600, 'OD': 185.0, 'Z0TL': 20.00  , 'vr': 1.0000, 'rho': 'RHO', 'plotkws': {'color':'b', 'ls':'-', 'lw':1}, 'dx':'DX'},
               }, #GTL
        } 
              
        CT, SZ = processGTL(55., 30., GTL)
        print(CT)
        print(SZ)
        
        # try and extract internal SZ object like ['V4/cone','V5/vcw']
        rA = CT.Get_Coeffs(E=['V6/fdr','V3/ss'], D=['V6/fdr','V3/ss'], mtype=('A','A'))
        rB = CT.Get_Coeffs(E=['V6/fdr','V3/ss'], D=['V6/fdr','V3/ss'], mtype=('A','B'))
        SZ = _sc.Scatter(fMHz=55, Zbase=CT.Zbase,
                         S=np.dot(np.linalg.inv(rA[0]), rB[0]), 
                         Portnames=['V6/fdr','V3/ss'])
        
        SZ4 = _sc.Scatter(fMHz=55, Zbase=CT.Zbase)
        SZ4.trlAZV(['V4/cone','V4/v5'], 0.2568, Z=18.)
#         SZ5 = _sc.Scatter(fMHz=55, Zbase=CT.Zbase)
#         SZ5.trlAZV(['V5','V5/vcw1'], 0.4123, Z=20.)
#         SZ45 = SZ4 + SZ5
        print(rA)
        print(rB)
        print('\nSZ from get_coeffs\n')
        print(SZ)
#        print(SZ45)

        print('\nSZ4 from Scatter\n')
        print(SZ4)
#        print(SZ5)
        
#         astate = CT.state()
#         PP.pprint(CT.state())
#         
#         CT2 = circuit(state=astate)
#         print(CT2)

    if False :
        print("Simple test : add fixed length TL to an N port ideal junction,\n"
              "              terminate N-1 ports and excite the remaining port.")

        SZ = sc.Scatter(fMHz=42.,Zbase=20.)
        SZ.trlAZV(['port1','port2'],1.2) # a TL with A=0, Zc=Zbase, v=c0 and length 1.2m
        
        # S-matrix of an ideal N-port junction
        N = 3
        S = np.array([[(2.-N)/N if rw is cl else 2./N for rw in range(N)]
                      for cl in range(N)])

        # build the circuit
        ct = circuit(Zbase=20.)   # default Zbase=50 Ohm
        ct.AddBlock("junction",
                    S=S,
                    Portnames=['J%d' % k for k in range(N)])

        for k in range(N):
            ct.AddBlock("TL%d" % k, SZ=SZ)
            ct.Connect(("TL%d" % k)+"/port1","junction/J%d" %k)
            if k :
                ct.Terminate(("TL%d" % k)+"/port2" )# , RC=-20, phase=1.57, units='dB', deg=0) # RC=0 on ct.Zbase Ohm !
        #        ct.Terminate(("TL%d" % k)+"/port2", RC=0.1, phase=0, units='MA') # RC=0 on ct.Zbase Ohm !
            else :
                ct.Excite(("TL%d" % k)+"/port2",1.)

        ct.Solve(printit=True)
        print(ct)
        print(ct.Zbase)
        
    
        
        print("Simple test : add fixed length TL to an N port ideal junction,\n"
              "              terminate N-1 ports and excite the remaining port.\n"
              "              Use TL elements")
        
        ct = circuit(Zbase=20.,fMHz=42.)
        ct.AddBlock("junction",S=S,Portnames=['Jport%d' % k for k in range(N)])
        for k in range(N):
            ct.AddBlockTL("TL%d" % k, 1.2, Portnames=['port1','port2'])
            ct.Connect(("TL%d" % k)+"/port1","junction/Jport%d" %k)
            if k :
                ct.Terminate(("TL%d" % k)+"/port2" )# , RC=-20, phase=1.57, units='dB', deg=0) # RC=0 on ct.Zbase Ohm !
        #        ct.Terminate(("TL%d" % k)+"/port2", RC=0.1, phase=0, units='MA') # RC=0 on ct.Zbase Ohm !
            else :
                ct.Excite(("TL%d" % k)+"/port2",1.)

        ct.Solve(printit=True)
        print(ct)
        print(ct.Zbase)
            
    if False :
        print("Test of Get_Coeffs\n")
        print("1- Simple TL of 1m composed of 2 sections of 0.5m\n")
        
        SZ = sc.Scatter(fMHz= 42.,  Zbase= 20.)
        SZ.trlAZV(["1","2"],0.5)

        SZ1 = sc.Scatter(fMHz= 42.,  Zbase= 20.)
        SZ1.trlAZV(["1","2"],0.5, 0., 50., 1.)

        ct = circuit(Zbase=20.)
        ct.AddBlock("TL_0.5m_1", SZ=SZ)
        ct.AddBlock("TL_0.5m_2", SZ=SZ1)
        ct.Connect("TL_0.5m_1/2","TL_0.5m_2/1")
        ct.Excite("TL_0.5m_1/1",1.)
        ct.Excite("TL_0.5m_2/2",1.)
        
        print(ct)
        print("Get the resulting S matrix between the 2 ports :\n")
        M, D, E = ct.Get_Coeffs(E=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                D=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                mtype='S')
        print("M=\n",M)
        print("\nD=\n",D)
        print("\nE=\n",E)

        print("\nCheck with a full section of 1m\n")
        SZ2 = sc.Scatter(fMHz= 42., Zbase= 20.)
        SZ2.trlAZV(["1","2"],0.5)
        SZ2.trlAZV("2",0.5, 0., 50., 1.)
        
        print(SZ2.Get_S())

        print("Get the resulting Z matrix between the 2 ports :\n")
        M, D, E = ct.Get_Coeffs(E=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                D=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                mtype='Z')
        
        print("M=\n",M)
        print("\nD=\n",D)
        print("\nE=\n",E)

        print("\nCheck with a full section of 1m\n")
        print(SZ2.Get_Z())
        
        print("Get the resulting VA and IA matrices between the 2 ports :\n")
        
        MV, D, E = ct.Get_Coeffs(E=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                 D=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                 mtype='VA')
        
        MI, D, E = ct.Get_Coeffs(E=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                 D=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                 mtype='IA')

        ZVI = np.dot(MV, np.linalg.inv(MI))

        print("Z from VA and IA =\n",ZVI)

        YVI = np.dot(MI, np.linalg.inv(MV))
        
        print("Y from VA and IA =\n",YVI)
      
        print("From Scatter \n",SZ2.Get_Y())

        Y, D, E = ct.Get_Coeffs(E=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                 D=["TL_0.5m_1/1","TL_0.5m_2/2"],
                                 mtype='Y')
        print("Y=\n",Y)
