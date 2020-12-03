################################################################################
#                                                                              #
# Copyright 2015, 2016, 2017, 2018, 2019, 2020                                 #
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
#                          .lpptrma@tlenet.be                                  #
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

__updated__ = '2020-10-30 14:47:00'

"""
Implementation of the Scatter toolbox in Python
Frederic Durodie, 16 November 2005

+-----------+------------------------------------------------------------------+
| date      | Modification                                                     |
+===========+==================================================================+
| 02 Oct 19 | corrected bug in __iadd__ when the 1st operant has no signals    |
|           | but the 2nd one does.                                            |
+-----------+------------------------------------------------------------------+
| 13 Oct 18 | added method convert_to_Zbase_from_mixed(Smixed,Zmixed)          |
+-----------+------------------------------------------------------------------+
| 19 Sep 17 | with EXACT=True (see import section) the exact speed of light in |
|           | vacuum is used as found in scipy.constants                       |
+-----------+------------------------------------------------------------------+
| 14 Sep 15 | added the __sub__ method for de-embedding ports                  |
+-----------+------------------------------------------------------------------+
| 14 Apr 15 | corrected a little bug in 'termport' (see 04 Apr 2015)           |
+-----------+------------------------------------------------------------------+
| 04 Apr 15 | corrected a nasty little bug in 'extendport' to do with having   |
|           | used arrays : S[:,k] is not really a column vector, one must use |
|           | S[:,[k]]|. It is surprising this has not bitten us before        |                      |
+-----------+------------------------------------------------------------------+
| 10 Mar 15 | fork from scatter3 : use array instead of matrix in view of      |
|           | performance [ongoing]                                            |
+-----------+------------------------------------------------------------------+
| 31 Jan 14 | python3 version                                                  | 
+-----------+------------------------------------------------------------------+
| 11 Dec 13 | cleanup and make python3 compatible                              | 
+-----------+------------------------------------------------------------------+
| 02 Oct 13 | cleanup of some errors (thanks to Canopy (pyflake))              | 
+-----------+------------------------------------------------------------------+
| 09 Jul 13 | added HybridCoupler4Port                                         | 
+-----------+------------------------------------------------------------------+
| 07 Mar 13 | added trlCoax(**kwargs)                                          | 
+-----------+------------------------------------------------------------------+
| 02 May 12 | added helper routines to return the SZ data as S, Z of Y matrices| 
+-----------+------------------------------------------------------------------+
| 18 Apr 12 | added equivallent L1, L2, X12 parameters extract from a given    | 
|           | S2P (scattering matrix of lossless 2 port) :                     | 
|           | Equivalent2PortParameters(S2P,Z0,signX12,fMHz=None,Vc=1)         |  
|           | -> (beta)L1, (beta)L2, X12                                       |
+-----------+------------------------------------------------------------------+
|  6 Jan 12 | added sortports method                                           |
+-----------+------------------------------------------------------------------+
| 15 Mar 11 | added the "tolerant" boolean parameter (default False) to        |
|           | calcsignals : if True its effect is to ingore signals not        |
|           | found instead of raising an exception.                           |
+-----------+------------------------------------------------------------------+
| 21 Sep 10 | modified convert_Zbase(newZbase) so that it does not involve     |
|           | evaluating Z (which may not always exist)                        |
+-----------+------------------------------------------------------------------+
| 20 Sep 10 | added renameports({'oldname':'newname',...})                     |
+-----------+------------------------------------------------------------------+
| 30 Apr 09 | added calcsignals ... under debugging                            |
+-----------+------------------------------------------------------------------+
| 16 Mar 09 | * adapt for pylab imports instead of scipy                       |
|           | * corrected an error in swapports                                |
+-----------+------------------------------------------------------------------+
|  7 Dec 05 | got more or less all key routines in place. But need testing.    |
+-----------+------------------------------------------------------------------+

..
    __init__(self,fMHz,Portnames=[],Smatrix=None,Zbase=1,iwt=1) -> Scatter object
    __repr__(self)                                              -> void
    flip_iwt(self)                                              -> Scatter object
    convert_Zbase(self)                                         -> Scatter object
    __add__(self,other)                                         -> Scatter object
    renameports(self,{'oldname':'newname', ...})                -> void
    renameport(self,oldname,newname)                            -> void
    findport(self)                                              -> int
    findsignal(self)                                            -> int
    addsignal(self,portname,signalname,type="V")                -> void
    swapports(self)                                             -> void
    sortports(self)                                             -> void
    termport(self)                                              -> void
    trlAZV(self,port1,lngth,A=0,Z=None,V=1,newname=None)        -> void
    series_impedance(self,Zs)                                   -> void
    parallel_impedance(self,Zp)                                 -> void
    extendport(self,port1,S=mat([[0,1],[1,0]]),newname=None)    -> void
    joinports
    connectports
    read_touchstone
"""

import os
from datetime import date

#---------------------------------------------------------------------- versions
#
# d e f i n e   v e r s i o n s
#

__version__ = date.fromtimestamp(os.path.getmtime(__file__)).strftime('%Y.%m.%d')
__version_info__ = tuple(__version__.split('.'))

#----------------------------------------------------------------------- imports
#
# i m p o r t   s e c t i o n
#
from copy import deepcopy
import numpy as np
from numpy.linalg import inv

from Utilities.printMatrices import printMA

from scipy.constants import speed_of_light
from scipy.constants import epsilon_0
from scipy.constants import mu_0

from scipy.optimize import fmin
from scipy.interpolate.fitpack2 import UnivariateSpline

#.-----------------------------------------------------------------------------.

any_of = lambda kws, adict : any([kw in adict for kw in kws])
all_of = lambda kws, adict : all([kw in adict for kw in kws])

#===============================================================================
#
# d e f i n e   e r r o r   c l a s s e s
#
class ScatterError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# not sure  why I kept those separated (FDe 19 Mar 2011)
# OK I remember vaguely : this is the easily differentiate between
# exactly this type of ScatterError and others ...

class ScatterError_port_not_found(ScatterError): pass   
    # used in calcsignals, raised nowhere ??
    
class ScatterError_signal_not_found(ScatterError): pass 
    # used in calcsignals and addsignal, raised in findsignal

#===============================================================================
#
#  S c a t t e r
#
class Scatter:
    """
    fMHz : the frequency in MHz
    Zbase : must be given except when a touchstone file is given as source and
            must be >0 if given
    iwt : time dependance convention (defaults to e^+jwt)
    
    Portnames : optional, if not given a default name will be used depending on
                the source that was supplied (n is port number:
                    Z_n for an impedance matrix (Z = ...)
                    Y_n for an admittance matrix (Y = ...)
                    S_n for a scattering matrix (S = ...)
                    T_n for touchstone file (TSF = ...)
                    M_n for a MWS directory (MWS = ...

    kwargs :
    
    one of the sources maybe specified to populate the Scatter class
    (otherwise an empty Scatter claas is returned)

    TSF : source is a path to a touchstone file (if Zbase is not given, the one
          of the file will be used)
    S : source is a scattering matrix (type pylab.mat or a [[]])
    Y : source is an admitance matrix (type pylab.mat or a [[]])
    Z : source is an impedance matrix (type pylab.mat or a [[]])
    MWS : source is a MWS directory

    experimental :
    V    : wave velocity relative to the speed of light in vacuum
    mur  : medium relative permeability
    epsr : medium retative permittivity
    
    c0   : speed of light [m/s] (defaults to scipy.constants.speed_of_light)
    """


    #===========================================================================
    #  _ _ i n i t _ _
    #
    def __init__(self, **kwargs):
        """
        kwargs :
        
        fMHz : the frequency in MHz
        
        Zbase : defaults to 50. Ohm or to the value in the Touchstone file if
                this is specified as a source
                
        iwt : time dependance convention (defaults to 1. for e^+jwt)
        
        Portnames : optional, if not given a default name will be used depending
                    on the source that was supplied (n is port number:
                    
          Z_n for an impedance matrix (Z = ...)
          Y_n for an admittance matrix (Y = ...)
          S_n for a scattering matrix (S = ...)
          T_n for touchstone file (TSF = ...)
          M_n for a MWS directory (MWS = ...
        
        one of the sources maybe specified to populate the Scatter class
        (otherwise an empty Scatter claas is returned)
        
          TSF : source is a path to a touchstone file (if Zbase is not given,
                the one of the file will be used)
          S   : source is a scattering matrix (type pylab.mat or a [[]])
          Y   : source is an admitance matrix (type pylab.mat or a [[]])
          Z   : source is an impedance matrix (type pylab.mat or a [[]])
          MWS : source is a MWS directory
          
        pfmt : print format for complex numbers. It is a tupple
                the first is 'RI' or 'MA' or 'DB'
                the second is a format string basically allowing
                two real arguments to be converted : e.g. '%+7.4f%+7.4fj '
               defaults to :  ('RI','%+7.4f%+7.4fj')

        these have been added to make the __repr__ somewhat meaningfull :

        Signals : optional list of signal names
        
        SM : optional signal matrix (has twice the number of signal names as rows
                                     and the same number of ports as columns)
             each pair row (python numbering 2*k) represents the voltage at the node
             where the signal was defined and each unpair row (2*k+1) is the current
             into the node as a function of the forward wave port excitations.

        experimental : these will affect default values in Scatter.trlCoax

        V    : wave velocity relative to the speed of light
        mur  : medium relative permeability
        epsr : medium retative permittivity
        
        c0 : speed pf light [m/s] (default use scipy.constants.speed_of_light)
        """

        if 'state' in kwargs:
            d = kwargs.pop('state')
            self.state(d)
            
            if kwargs:
                raise ValueError(
                    'Scatter.__init__: state is incompatible with any other kwarg')

            return

        self.c0 = kwargs.get('c0',speed_of_light)
        
        #------------------------------------------------------------------toMat

        def toArray(M):
            """return a numpy.array of M"""
            try:
                return np.array(M,dtype=np.complex128)
            except:
                raise ValueError("Scatter.__init__ : supplied matrix could not "
                                 "be converted to numpy.array")
        #.......................................................................
            
        # defaults

        self.S        = [] # note tbc we might use None
        self.Ssigs    = np.zeros((0,0), dtype=np.complex128) # np.mat([[]])


        # analyse supplied parameters (kwargs)
        
        #------------------------------------------------------------------ fMHz
        self.fMHz = kwargs.pop('fMHz', None)
        if self.fMHz <= 0. :
            raise ValueError("Scatter.__init__ : fMHz must be positive")
        self.w = 2E6*np.pi*self.fMHz
        self.fHz = self.fMHz*1E6

        #----------------------------------------------------------------- Zbase
        self.Zbase = kwargs.pop('Zbase', 50.)
        if self.Zbase <= 0. :
            raise ValueError("Scatter.__init__ : Zbase must be positive")
                
        #------------------------------------------------------------------- iwt
        self.iwt = kwargs.pop('iwt',+1.)
        if np.abs(self.iwt) != 1.:
            raise ValueError("Scatter.__init__ : |iwt| must be 1")
        
        #------------------------------------------------------------- Portnames
        self.Portname = kwargs.pop('Portnames',[])

        #------------------------------------------------------------------ pfmt
        self.pfmt = kwargs.pop('pfmt',('RI','%+10.6f%+10.6fj'))
        
        #---------------------------------------------------------- V, epsr, mur
        if 'V' in kwargs:
            self.V = kwargs.pop('V')
            if self.V > 1:
                raise ValueError('Scatter.__init__ : V > 1')
            
            if 'epsr' in kwargs:
                self.epsr = kwargs.pop('epsr')
                if self.epsr < 1:
                    raise ValueError('Scatter.__init__ : epsr < 1')
               
                self.mur = 1 / (self.V**2 * self.epsr)
                
                if self.mur < 1:
                    raise ValueError('Scatter.__init__ : V, epsr -> mur < 1')
                
                if 'mur' in kwargs:
                    mur = kwargs.pop('mur')
                    if mur < 1 :
                        raise ValueError('Scatter.__init__ : mur < 1')
                    if np.abs(mur/self.mur - 1) > 1E-5:
                        raise ValueError(
                                'Scatter.__init__ : V <> 1/sqrt(mur.epsr)')
            else:
                self.mur = kwargs.pop('mur', 1.)
                self.epsr = 1 / (self.V**2 * self.mur)
                if self.epsr < 1:
                    raise ValueError('Scatter.__init__ : V, mur -> epsr < 1')
                                            
        else:
            self.epsr = kwargs.pop('epsr', 1.)
            if self.epsr < 1:
                raise ValueError('Scatter.__init__ : epsr < 1')
            
            self.mur = kwargs.pop('mur', 1.)
            if self.mur < 1:
                raise ValueError('Scatter.__init__ : mur < 1')
            
            self.V = 1 / np.sqrt(self.epsr * self.mur)
            if self.V > 1:
                raise ValueError('Scatter.__init__ : epsr, mur -> V > 1')
            
        #---------------------------------------------------------------- source
        
        got_source = False
        err = False
        if 'TSF' in kwargs:
            got_source = 'TSF'
            err =  any_of(['MWS', 'S', 'Z', 'Y'], kwargs)
        
        elif 'MWS' in kwargs:
            got_source = 'MWS'
            err =  any_of(['S', 'Z', 'Y'], kwargs)
            
        elif 'S' in kwargs:
            got_source = 'S'
            err = any_of(['Z', 'Y'], kwargs)
            
        elif 'Z' in kwargs:
            got_source = 'Z'
            err = 'Y' in kwargs
        
        elif 'Y' in kwargs:
            got_source = 'Y'
        
        if err:
            raise ValueError("Scatter.__init__ : only one of (TSF = ..., "
                             "S = ..., Y = ...,  = ..., MWS = ...) allowed")
        
        if got_source:
            defaultname = got_source + '_%d'
        
        #--------------------------------------------------------------- Signals
        
        self.Signal = kwargs.pop('Signals', [])
        SM = kwargs.pop('SM', np.zeros((0,0), dtype=np.complex128))
        self.Ssigs = toArray(SM) if SM else np.zeros((0,0), dtype=np.complex128)
         
        #-------------------------------------------------------- process source
        
        # need to interprete the source if given when Zbase and fMHz are set
        
        if got_source :
            value = kwargs.pop(got_source)
            if got_source == 'TSF':
                # must put here to avoid circular dependencies
                from pyRFtk import TouchStoneClass3a as ts
                # ! TouchStone.Datas returns a numpy.array
                tsf = ts.TouchStone(filepath=value)
                M = tsf.Datas(freqs=self.fMHz,
                              unit='MHz',
                              mtype='S',
                              elfmt='RI',
                              zref=self.Zbase)
                self.S = toArray(M)
                
            elif got_source == 'S':
                self.S = toArray(value)
                
            elif got_source == 'Y':
                self.S = S_from_Z(inv(toArray(value)),self.Zbase)
                
            elif got_source == 'Z':
                self.S = S_from_Z(toArray(value),self.Zbase)
                
            else: # if kwarg == 'MWS'
                raise ValueError("Scatter.__init__ : MWS method not "
                                 "implemented yet")

        #--------------------------------------------------- unrecognized kwargs

        if kwargs:
            raise ValueError("Scatter.__init__ unrecognized argument : %s" %
                                 kwargs)

        #--------------------------------------------- set portnames as required
        
        if self.Portname:
            if len(self.S) is not len(self.Portname):
                raise ValueError("Scatter.__init__ : mismatched number of "
                                 "ports and port names")
        else:
            if len(self.S):
                self.Portname = [defaultname % (k+1) for k in range(len(self.S))]
        #endif

        #------------------------------------------------------ check on signals

        if 2 * len(self.Signal) is not len(self.Ssigs):
            raise ValueError("Scatter.__init__ : mismatched number of " +
                             "of signal names (%d) and " % len(self.Signal) +
                             "rows of the signal matrix (%d)" % len(self.Ssigs))
        
        elif self.Signal:
            try:
                Q = len(self.Portname) is not self.Ssigs.shape[1]
            except AttributeError: # when self.Ssigs is [] (nothing defined : empty)
                Q = True
            if Q:
                raise ValueError("Scatter.__init__ : mismatched number of " +
                                  "of ports (%d)" % len(self.Portname) +
                                  "and coulumns of the signal matrix (%d)" %
                                  self.Ssigs.shape[1])
        
        self.TLproperties() # sets self.lastTL
        
    #===========================================================================
    #
    #  s t a  t e
    #
    def state(self, d=None):
        # print('going through state')
        if d is None:
            return {
                'fMHz'    : self.fMHz,
                'Zbase'   : self.Zbase,
                'c0'      : self.c0,
                'Portname': self.Portname,
                'S'       : [] if self.S == [] else self.S.tolist(),
                'iwt'     : self.iwt,
                'Ssigs'   : [] if self.Ssigs == [] else self.Ssigs.tolist(),
                'w'       : self.w,
                'fHz'     : self.fHz,
                'pfmt'    : self.pfmt,
                'V'       : self.V,
                'epsr'    : self.epsr,
                'mur'     : self.mur,
                'Signal'  : self.Signal,
                'lastTL'  : self.lastTL
            }
            
        else:
            self.fMHz = d['fMHz']
            self.Zbase = d['Zbase']
            self.c0 = d['c0']
            self.Portname = d['Portname']
            self.S = [] if d['S'] == [] else np.array(d['S'])
            self.iwt = d['iwt']
            self.Ssigs = [] if d['Ssigs'] == [] else np.array(d['Ssigs'])
            self.w = d['w']
            self.fHz = d['fHz']
            self.pfmt = d['pfmt']
            self.V = d['V']
            self.epsr = d['epsr']
            self.mur = d['mur']
            self.Signal = d['Signal']
            self.lastTL = d['lastTL']

    #===========================================================================
    #
    #  c l o n e _ e x c e p t  
    #
    def clone_except(self, **kwargs):
        """
        return a clone of self except for those attributes given in kwargs
        """
        other = self.copy()
        other.fMHz = kwargs.pop('fMHz', self.fMHz)
        other.Zbase = kwargs.pop('Zbase', self.Zbase)
        other.iwt = kwargs.pop('iwt', self.iwt)
        other.Portname = kwargs.pop('Portname', deepcopy(self.Portname))
        other.S = kwargs.pop('S', deepcopy(self.S))
        other.Ssigs = kwargs.pop('Ssigs',deepcopy(self.Ssigs))
        other.Signal = kwargs.pop('Signal',deepcopy(self.Signal))
        other.pfmt = kwargs.pop('pfmt', self.pfmt)
        other.V = kwargs.pop('V',self.V)
        other.epsr = kwargs.pop('epsr', self.epsr)
        other.mur = kwargs.pop('mur',self.mur)
        
        if kwargs:
            raise ValueError('Scatter.copy_to: unknown attributes %r'
                             % [kw for kw in kwargs])
        return other
    
    #===========================================================================
    #
    #  c o p y _ f r o m 
    #
    def copy_from(self, other, **kwargs):
        """
        copy other's attributes to self except those given in kwargs
        """
        if not isinstance(other,Scatter):
            raise ValueError("Scatter.copy_from: type mismatch")
            
        self.fMHz = kwargs.pop('fMHz', other.fMHz)
        self.Zbase = kwargs.pop('Zbase', other.Zbase)
        self.iwt = kwargs.pop('iwt', other.iwt)
        self.Portname = kwargs.pop('Portname', deepcopy(other.Portname))
        self.S = kwargs.pop('S', deepcopy(other.S))
        self.Ssigs = kwargs.pop('Ssigs',deepcopy(other.Ssigs))
        self.Signal = kwargs.pop('Signal',deepcopy(other.Signal))
        self.pfmt = kwargs.pop('pfmt', other.pfmt)
        self.V = kwargs.pop('V',other.V)
        self.epsr = kwargs.pop('epsr', other.epsr)
        self.mur = kwargs.pop('mur',other.mur)
        
        if kwargs:
            raise ValueError('Scatter.copy_from: unknown attributes %r'
                             % [kw for kw in kwargs])
        
    #===========================================================================
    #
    #  c o p y _ t o
    #
    def copy_to(self, other, **kwargs):
        """
        copy self's attributes to other except those given in kwargs
        """
        
        if not isinstance(other,Scatter):
            raise ValueError("Scatter.copy_to: type mismatch")

        other.fMHz = kwargs.pop('fMHz', self.fMHz)
        other.Zbase = kwargs.pop('Zbase', self.Zbase)
        other.iwt = kwargs.pop('iwt', self.iwt)
        other.Portname = kwargs.pop('Portname', deepcopy(self.Portname))
        other.S = kwargs.pop('S', deepcopy(self.S))
        other.Ssigs = kwargs.pop('Ssigs',deepcopy(self.Ssigs))
        other.Signal = kwargs.pop('Signal',deepcopy(self.Signal))
        other.pfmt = kwargs.pop('pfmt', self.pfmt)
        other.V = kwargs.pop('V',self.V)
        other.epsr = kwargs.pop('epsr', self.epsr)
        other.mur = kwargs.pop('mur',self.mur)
        
        if kwargs:
            raise ValueError('Scatter.copy_to: unknown attributes %r'
                             % [kw for kw in kwargs])
    
        
    #===========================================================================
    #  _ _ e q _ _  ,  _ _ l t _ _
    #
    # (not very usefull)
    #
    def __eq__(self, other):
        return repr(self) == repr(other)

    def __lt__(self, other): # not very usefull
        return self.fMHz < other.fMHz
    
    #===========================================================================
    #  t e s t e q
    #
    def testeq(self,other,**kwargs):
        """
        test if 2 Scatter object could represent the same circuit :

        portnames / numbers must be the same (but not in same order)
        signalnames / numbers must be the same (but not in the same order)

        if abstol (or none given : default 1e-6) :

        and/or if reltol given :
        
        kwargs:

            abstol : absolute tolerance for the S matrix
            reltol : relative tolerance for the S and Ssigs matrix
        """

        def convert2MA(X,
                       fmt='%0.9g<%10.4deg',
                       fun=lambda x : (np.abs(x), np.angle(x,deg=True))):
            return fmt % fun(X)

        abstol = kwargs.pop('abstol',1E-6)
        reltol = kwargs.pop('reltol', None)
        info = kwargs.pop('info',False)

        if kwargs:
            raise TypeError("Scatter.test : unexpected argument(s) '%r'"
                            % kwargs)
                        
        # test if size wize it is the same thing
        if len(self.Portname) is not len(other.Portname) :
            if info:
                print("Scatter.testeq : different number of ports")
            return False

        # ok do we have the same ports (names)
        for p1 in self.Portname:
            if p1 not in other.Portname :
                if info:
                    print("Scatter.testeq : different port names")
                return False

        # ok so now check the S-matrices
        abserr, relerr = 0., 0.
        for k1, p1 in enumerate(self.Portname):
            q1 = other.Portname.index(p1)
            for k2, p2 in enumerate(self.Portname):
                q2 = other.Portname.index(p2)
                if abstol:
                    abs_err = self.S[k1,k2] - other.S[q1,q2]
                    if np.abs(abs_err) > abserr:
                        k1max, k2max, abserr = k1, k2, abs_err
                        
                if reltol:
                    avg = (self.S[k1,k2] + other.S[q1,q2])/2
                    dif = self.S[k1,k2] - other.S[q1,q2]
                    if np.abs(avg) :
                        rel_err = dif / np.abs(avg)
                    elif np.abs(dif):
                        rel_err = 1e+20 # something ridiculously big
                    if np.abs(rel_err) > relerr:
                        q1max, q2max, relerr = k1, k2, rel_err

        fail = False
        
        if abstol:
            if np.abs(abserr) > abstol:
                if info:
                    print('Scatter.testeq: S[%s,%s] = %s : delta = ' %
                          (self.Portname[k1max],self.Portname[k2max],
                           convert2MA(self.S[k1max,k2max]),
                           convert2MA(abserr)))
                fail = True
                    
        if reltol:
            if np.abs(relerr) > reltol:
                if info:
                    print('Scatter.testeq: S[%s,%s] = %s : delta/avg = ' %
                          (self.Portname[k1max],self.Portname[k2max],
                           convert2MA(self.S[q1max,q2max]),
                           convert2MA(relerr)))
                fail = True
                    
        if fail:
            return False

        # ok now check the signals
        
        for ks, s in enumerate(other.Signal):
            if s not in self.Signal :
                if info:
                    print('Scatter.testeq : different signals')
                return False

        abs_err, rel_err = 0., 0.
        
        for ks, s in enumerate(self.Signal):
            if s not in other.Signal :
                if info:
                    print('Scatter.testeq : different signals')
                return False
            else:
                qs = other.Signal.index(s)
                for kp, p in enumerate(self.Portname):
                    qp = other.Portname.index(p)
                    for k, q in zip([2*ks, 2*ks+1],[2*qs, 2*qs+1]):
                        if abstol:
                            abs_err = self.S[k,kp] - other.S[q,qp]
                            if np.abs(abs_err) > abserr:
                                k1max, k2max, abserr = k, kp, abs_err
                                
                        if reltol:
                            avg = (self.S[k,kp] + other.S[q,qp])/2
                            dif = self.S[k,kp] - other.S[q,qp]
                            if np.abs(avg) :
                                rel_err = dif / np.abs(avg)
                            elif np.abs(dif):
                                rel_err = 1e+20 # something ridiculously big
                            if np.abs(rel_err) > relerr:
                                q1max, q2max, relerr = k, kp, rel_err
                        
        fail = False
        
        if abstol:
            if np.abs(abserr) > abstol:
                if info:
                    print('Scatter.testeq: Ssig[%s,%s] = %s : delta = ' %
                          (self.Portname[k1max],self.Portname[k2max],
                           convert2MA(self.Ssigs[k1max,k2max]),
                           convert2MA(abserr)))
                fail = True
                    
        if reltol:
            if np.abs(relerr) > reltol:
                if info:
                    print('Scatter.testeq: Ssig[%s,%s] = %s : delta/avg = ' %
                          (self.Portname[k1max],self.Portname[k2max],
                           convert2MA(self.Ssigs[q1max,q2max]),
                           convert2MA(relerr)))
                fail = True
                    
        if fail:
            return False

        return True
    
    #===========================================================================
    #  _ _ r e p r _ _
    #
    def __repr__(self):
        """
        return a string unambiguously representing the Scatter class object
        """
        s = 'Scatter(\n'
        s += ' fMHz=%r,\n' % self.fMHz
        s += ' Zbase=%r,\n' % self.Zbase
        s += ' iwt=%r,\n' % self.iwt
        s += ' Portnames=%r,\n' % self.Portname
        if len(self.S) :
            s += ' S=%r,\n' % self.S.tolist()
        else:
            s += ' S=%r,\n' % self.S
        s += ' Signals=%r,\n' % self.Signal
        try:
            s += ' SM=%r,\n'  % self.Ssigs.tolist()
        except AttributeError:
            # this is because we have a list rather than a numpy array
            # it is proabably a remanent from earlier versions which were
            # pickled
            s += ' SM=%r,\n'  % self.Ssigs
        s += ' pfmt=(%r,%r),\n' % self.pfmt
        s += ' V=%r,\n' % self.V
        s += ' mur=%r,\n' % self.mur
        s += ' epsr=%r\n' % self.epsr
        s += ')'

        return s
    
    #===========================================================================
    #  _ _ s t r _ _
    #
    def __str__(self):
        """
        pretty print a Scatter class as ascii strings
        """
        s = ('frequency = %10.3f [MHz], Zbase = %5.1f [Ohm], iwt = %2i \n' %
                 (self.fMHz, self.Zbase, self.iwt))
        s += (' V = %10.4f, mur = %10.4f, epsr = %10.4f\n\n' %
                 (self.V, self.mur, self.epsr))
        s += '%5i ports, %6i signals \n\n' % (len(self.Portname),len(self.Signal))
        s += 'Ports(' + str(len(self.Portname)) + '):\n'

        pfun = {'RI': lambda x : (np.real(x),np.imag(x)),
                'MA': lambda x : (np.abs(x),np.angle(x,deg=1)),
                'DB': lambda x : (20*np.log10(x),np.angle(x,deg=1))
               }[self.pfmt[0]]
        pfmt = ' '+self.pfmt[1]+' '
        
        for k in range(len(self.Portname)):
            s += '%10s: ' % self.Portname[k]
            for l in range(len(self.S)):
                s += pfmt % pfun(self.S[k,l])
            s += '\n'
            
        s += 'Signals(' + str(len(self.Signal)) + '):\n'
        if self.Signal:
            for k in range(len(self.Signal)):
                s += '%10s_v: ' % self.Signal[k]
                for l in range(len(self.S)):
                    s += pfmt % pfun(self.Ssigs[2*k,l])
                s += '\n'
                s += '          _i: '
                for l in range(len(self.S)):
                    s += pfmt % pfun(self.Ssigs[2*k+1,l])
                s += '\n'
        return s + '\n'
    
    #===========================================================================
    #  c o p y 
    #            
    def copy(self):
        """
        copy : returns a copy of the scatter object (forces a deepcopy)
        """
        return deepcopy(self) # make a copy 

    #===========================================================================
    #  f l i p _ i w t
    #
    def flip_iwt(self, inplace=False):
        """
        flips the time dependence "preference" from +iwt to -iwt or vice versa
        """
        if inplace:
            self.iwt = - self.iwt
            self.S = np.conjugate(self.S)
            if len(self.Ssigs):
                self.Ssigs = np.conjugate(self.Ssigs)
        else:
            SZ = self.copy()
            SZ.flip_iwt(inplace=True)
            return SZ

    #===========================================================================
    #  c o n v e r t _ t o _ m i x e d
    #
    def convert_to_mixed(self, Zmixed):
        """
        return the Smixed with mixed port impedances Zmixed from self
        """
        
        Smixed = convert_general(Zmixed, self.S, self.Zbase)
        
        return Smixed
        
    #===========================================================================
    #  c o n v e r t _ t o _ Z b a s e _ f r o m _ m i x e d
    #
    def convert_to_Zbase_from_mixed(self, Smixed, Zmixed):
        """
        return the scatter3 of a Smixed with mixed port impedances Zmixed
        """
        
        SZbase = convert_general(self.Zbase, Smixed, Zmixed)
        
        return Scatter(fMHz=self.fMHz, Zbase=self.Zbase, S=SZbase, 
                       iwt=self.iwt, Portnames=self.Portname)
        
    #===========================================================================
    #  c o n v e r t _ Z b a s e
    #
    def convert_Zbase(self, newZbase, inplace=False):
        """
        return a Scatter change the reference impedance base of the Scatter object
        """
        if float(newZbase) <= 0.:
            raise ValueError("Scatter.convert_Zbase: "
                             "new Zbase must be strictly positive")
        if inplace :
            if len(self.S):
                Id = np.eye(len(self.S))
                # 2010-09-21 FDe : change of base not involving Z
                # (which may not always exists)
                p = (self.Zbase/newZbase + 1.)/2
                m = p - 1.
                # convert signals first as the old self.S is needed
                if self.Signal:
                    self.Ssigs = np.dot(self.Ssigs, (p * Id - m * self.S))
                self.S = np.dot(inv(p * Id + m * self.S), (m * Id + p * self.S))
                
                self.Zbase = float(newZbase)
        else:
            SZ = self.copy()
            SZ.convert_Zbase(newZbase, inplace=True)
            return SZ

    #===========================================================================
    #  f i n d p o r t 
    #
    def findport(self,portname):
        """
        find the index for a given portname (if an int was supplied this int is
        returned except if the int is not a valid index).
        """
        if type(portname) is int:
            if (portname < 0) or (portname >= len(self.S)) :
                raise IndexError("Scatter.findport: "
                                 "port number (%d) out of range" % portname)
            return portname
                                       
        else:
            try:
                return self.Portname.index(portname)
            except ValueError:
                # should that be ScatterError_port_not_found
                raise ValueError("Scatter.findport: Port '%s' not found " %
                                 str(portname))
                                       
    #===========================================================================
    #  r e n a m e p o r t s
    #
    def renameports(self, namedict, inplace=False):
        """
        renameports({'oldname':'newname','oldname2':'newname2', ... })
        Note : this version returns a Scatter object unless inplace is True
        """
        
        if inplace:
            for name in namedict.keys():                
                if namedict[name] in self.Portname :
                    raise ScatterError("Scatter.renameports: " +
                                       "port name '%s' exists " % str(name) +
                                       " --> '%s'" % str(namedict[name]))
                else :
                    self.Portname[self.findport(name)] = namedict[name]
        else:
            SZ = self.copy()
            SZ.renameports(namedict,inplace=True)
            return SZ
        
    
    #===========================================================================
    #  r e n a m e s i g n a l s
    #
    def renamesignals(self, namedict, inplace=False):
        """
        renamesignals({'oldname':'newname','oldname2':'newname2', ... })
        Note : this version returns a Scatter object
        """
        if inplace :
            for name in namedict.keys():
                if namedict[name] in self.Signal :
                    raise ScatterError("Scatter.renamesignals: " +
                                       "signal name '%s' exists " % str(name) +
                                       " --> '%s'" % str(namedict[name]))
                else :
                    self.Signal[self.findsignal(name)] = namedict[name]
        else:
            SZ =  self.copy()
            SZ.renamesignals(namedict,inplace=True)
            return SZ
                    
    #===========================================================================
    #  r e n a m e p o r t
    #
    def renameport(self, portname, newname):
        """
        renames a port except if the newname already exsists.
        """
        if newname in self.Portname :
            raise ScatterError("Scatter.renameport: "
                               "port name '%s' exists" % str(newname))
        else:
            self.Portname[self.findport(portname)] = newname

    #===========================================================================
    #  f i n d s i g n a l
    #
    def findsignal(self, signalname):
        """
        find the index to a signal name.
        """
        try:
            k = self.Signal.index(signalname)
        except ValueError:
            raise IndexError("signal '%s' not found " % str(signalname))
        return k
    
    #===========================================================================
    #  a d d s i g n a l
    #
    def addsignal(self,portname,signalname=None):
        """
        add a signalname to a port (it does not matter if the port is resolved
         later).
        """
        if not signalname:
            signalname = portname
        k = self.findport(portname)
        a = np.zeros(len(self.S))
        # print('%r' % a)
        a[k] = 1.
        b = self.S[k]
        if self.Signal:       # this could be rewritten not to use exceptions ...
            try:
                s = self.findsignal(signalname)
                self.Ssigs[2*s] = a + b
                self.Ssigs[2*s+1] = (a - b)/self.Zbase
            except IndexError: # because signal does not exist
                self.Signal.append(signalname)
                self.Ssigs = np.vstack((self.Ssigs,a + b,(a - b)/self.Zbase))
        else:
            self.Signal.append(signalname)
            self.Ssigs = np.vstack((a + b,(a - b)/self.Zbase))
        
        # print('%r' % self.Ssigs)

    #===========================================================================
    #  c a l c s i g n a l s 
    #
    def calcsignals(self, signals, fwdwaves, tolerant=False):
        """
        computes the requested signals (dict) for the port excitation given in
        fwdwaves (dict) e.g.

            signalvalues = scatter.calcsignals({'M1':['P+','Z'],
                                                'P2':('Vmax',l0,Z0,lwave),
                                                ...
                                               },
                                               {'P1':(1-1j),'P2':0})

            where 'M1', ... are defined signal names and 'P1', 'P2' are all
            the port names of the Scatter object.
            
            the value returned is a dict :
            {'M1' : [value of P+, value of Z],
             'P2' : (value of Vmax, location of Vmax),
             ...
            }
            
        note 1 - that the fwdwaves dict must contain all ports currently
        defined for the Scatter object

        note 2 - the value in the signal, value pair in signals does not need to
        be a list if only a single value type is requested. The resulting sigal
        values dictionary will reflect this as well.
        
        accepted signal (single value) types are :
        
        'A'  : forward voltage wave 
        'B'  : reflected voltage
        'P+' : forward power
        'P-' : reflected power
        'V'  : voltage
        'I'  : current into the port
        'Z'  : impedance looking into the port
        'Y'  : admittance looking into the port
        'G'  : reflection coefficient into the port

        notes :

        1) forward, reflected, into : refer to the port where the signal was 
           added; that port could be eliminated when the circuit is further
           build up.
                                           
        2) where applicable the reference impedance is Scatter.Zbase

        3) if a signalname is not found port names are searched

        following composed signal types are also possible (they are requested
        as tuples rather than just a string) :

        ('Vmax' or 'Imax',[length, [Z0, [Lwave | {'v0' : 1.0}]]]):
        
            maximum voltage on a TL of specified length (optional default infinite)
            and characteristic impedance Z0 (optional default Scatter.Zbase) for
            a wave length of eiter Lwave (optinional default c0/Scatter.fMHz) or
            given from phase velocity relative to c0 'v0' (default 1.)
            
            if length is positive : outwards from the port position
            else : inwards from the port position
            
            -> returns (Vmax, xmax) or (Imax, xmax)
            
        ('VSW'|'ISW'|'VISW',[length,[Z0,[Lwave | {'v0' : 1.0},[dx]]]])

            optional parameters are as above with the addition of distance interval,
            dx, which defaults to a 72nd (5 electric degrees) of the wave length.

            -> returns ([x[k]], [V(x[k]) or I(x[k])])  for 'VSW' or 'ISW'
                       ([x[k]], [V(x[k])], [I(x[k])])  for 'VISW'

        However this form is now considered deprecated (and a warnign will be issued)
        in view of what follows :

        We should now consider : a dict as a general signal type :

        { 'general-signal-type' : { 'parameter1' : p1, ... , 'parameterN' : pN} }

        e.g.
        
        { 'VSW' : {'L' : 0.,
                   'Z0' : self.Zbase,
                   'Lwave' : speed_of_light/self.fMHz or 'v0' : 1.0,
                   'dx' : Lwave/72} }

        { 'A' : {'Z0' : Zbase} } would produce the forward voltage wave
                                 on eg. a different reference impedance

        also

        { 'sig1' : {},
          'sig2' : { ... 'sigtype_k' : sk ... } would return the results as a dict
                                                avoiding the need to remember the
                                                order of the sigtypes                                                             
        }
        """

        if sorted(fwdwaves.keys()) != sorted(self.Portname):
            raise ValueError("Scatter.calcsignals: "
                             "excitation not compatible with ports")

        # construct the column vector of the excitations
        Afwds = np.array([fwdwaves[port] for port in self.Portname])
        
        # sort out the signalnames now
        # sigslist = signals.keys()
        tresult = {}
        for sig, sigtypes in signals.items():
            docalc = True
            try:
                ksig = self.findsignal(sig)
                # OK found the signal; now get the V and I "vectors"
                Vsig = np.dot(self.Ssigs[2*ksig], Afwds)
                Isig = np.dot(self.Ssigs[2*ksig+1], Afwds)
                
            except IndexError:
                try:
                    ksig = self.findport(sig)
                    # OK found a port with the same name;
                    # now construct the V and I "vectors" from self.S
                    a = np.zeros(len(self.S),dtype=complex)
                    a[ksig] = 1.+0.j
                    b = self.S[ksig]
                    Vsig = np.dot(a + b, Afwds)
                    Isig = np.dot((a - b)/self.Zbase,  Afwds)
                    
                except IndexError:
                    # it's not a signal name nor a portname
                    if tolerant:
                        docalc = False # the user chooses to ignore the signal quietly
                    else:
                        raise ValueError("Scatter.calcsignals: "
                                         "signal (or port)'s' not found :" %
                                         str(sig))

            # produce the required signal types
            if docalc:
                tresult[sig] = self.process_sig(sigtypes, Vsig, Isig)
            else:
                tresult[sig] = None # TBC how was this handled before ? Does it matter
              
        return tresult

    #===========================================================================
    #  p r o c e s s _ s i g 
    #
    def process_sig(self, sigtype, Vsig, Isig):
        """
        process (possibly) multiple signal types :

        sitype :
        
            str    - string (single key) type
            
                     eg. 'V', 'I', ...

                    these will be transformed to (single key) dict type w.o.
                    parameters  -> {'V':{}}, {'I':{}}, ...
                    and then further processed by Scatter.base_signal() which
                    will return a float
                    
            tuple  - (single key) type
            
                     eg. ('VSW', L, Z0, ... )
                     
                     these will be transformed to (atomic) dict type with parameters
                     
                     -> {'VSW':{'L': ..., 'Z0': ..., ...}}

                     and then further processed by Scatter.process_base()
                     which will return a dict :

                       'Vmax' | 'Imax'   ->   {'max': max V | max I, 'x' : x of max}
                       'VSW' | 'ISW'     ->   ('V': [Vk] | 'I': [Ik], 'x': [xk])
                       'VISW'            ->   ('V': [Vk], 'I': [Ik], 'x': [xk])

                     these will be unpacked to return a tuple :
                     
                       'Vmax' | 'Imax'   ->   (max V | max I, x of max)
                       'VSW' | 'ISW'     ->   ([Vk] | [Ik], [xk])
                       'VISW'            ->   ([Vk], [Ik], [xk])

            dict   - single key or multiple keys
            
                     eg. {'V': {} }                          # single w.o parameter
                         {'VSW': {'L': ..., ...}}            # single w. parameter
                         {'A':{'Z0': ...}}                   # single w. paramater
                         {'V':{}, 'ISW':{'L': ...,}, ...}    # multiple w./w.o. params
                         
                     further processed by Scatter.base_signal() and return a dict
                     that have the same keys as the passed sigtype.
                     
            list   - multiple -> process each item
                     eg. ['V','Z', ...,                 # string atomic types
                          ('ISW', ... ),                # tuple atomic types
                          {'A': {}},                    # atomic dict types wo params
                          {'VSW' : {'L': L, ...}},      # atomic dict type w params
                         ]
                         
        """
        if type(sigtype) is list:
            tresult = [self.process_sig(sig, Vsig, Isig) for sig in sigtype]

        elif type(sigtype) is str:
                    tresult = self.base_signal(sigtype, {}, Vsig, Isig)
            
        elif type(sigtype) is dict:
            tresult = {}
            for sigid, params in sigtype.items():
                tresult[sigid] = self.base_signal(sigid, params, Vsig, Isig)

        elif type(sigtype) is tuple:
            # prepare the defaults
            # TBC warnings.warn('Use of tuples as signal type is depreciated')
            qwargs = {'L':0.,
                      'Z0':self.Zbase,
                      'v0':1.,  # equivalent to Lwave = v0*speed_of_light/self.fMHz
                      'dx':speed_of_light/self.fHz/72}
            
            # replace the defaults as data is provided
            try:
                qwargs['L'] = sigtype[1]
                qwargs['Z0'] = sigtype[2]
                if type(sigtype[3]) == dict:
                    qwargs['v0'] = sigtype[3]['v0']
                else:
                    qwargs.pop('v0') # use supplied Lwave rather than v0
                    qwargs['Lwave'] = sigtype[3]
                qwargs['dx'] = sigtype[4]
            except IndexError:
                pass
            
            # now process it as you would process a dict
            tresult = self.base_signal(sigtype[0], qwargs, Vsig, Isig)

            # unpack the results into a tuple
            if sigtype[0].upper() in ['VMAX', 'IMAX'] :
                tresult = (tresult['max'],tresult['x'])
            elif sigtype[0].upper() == 'VSW' :
                tresult = (tresult['V'],tresult['x'])
            elif sigtype[0].upper() == 'ISW' :
                tresult = (tresult['I'],tresult['x'])
            else: # sigtype[0] == 'VISW'
                tresult = (tresult['V'], tresult['I'], tresult['x'])

        else:
            raise TypeError("Scatter.calcsignals: incompatible signal type")
            
        return tresult

    #===========================================================================
    #  b a s e _ s i g n a l
    #
    def base_signal(self, sigid, params, Vsig, Isig):
        """
        process a base type signal :

        sigid :                                       params.keys():
        
        'A'      : forward voltage wave                 'Z0'
        'B'      : reflected voltage                    "
        'G','G+' : reflection coeff. into the port      "
        'G-'     : reflection coeff. out of the port    "
        'P+'     : forward power                        "
        'P-'     : reflected power                      "
        'V'      : voltage                              None
        'I'      : current into the port                "
        'Z','Z+' : impedance looking into the port      "
        'Z-'     : impedance looking out of the port    "
        'Y'      : admittance looking into the port     "
        'P','PWR','PWR+' : net power into the port      "
        'PWR-'   : net power out of the port (= -'P')   "
        'VSW'    : voltage standing wave                'L', 'Z0', 'Lwave'|'v0', 'dx'
        'ISW'    : current standing wave                "
        'VISW'   : V and I standing waves               "
        'VMAX'   :  Vmax and location of Vmax           'L', 'Z0', 'Lwave'|'v0'
        'IMAX'   : Imax and location of Imax            "

        defaults :

        'Z0'    = Scatter.Zbase
        'L'     = 0. if 'L' is positive the VSW, ... is on a TL extending the port
                     else it on a TL inwards of the port. if L is 0. the TL is infinte
        'Lwave' = 300./Scatter.fMHz but if 'v0' is given : v0 * 300./Scatter.fMHz
        'v0'    = 1. but 'Lwave' overrides 'v0'
        'dx'    = 'Lwave'/72

        Vsig, Isig : port/node's voltage and current        
        """

        sigid = sigid.upper()
        
        if sigid in ['A','B','G','G+','G-','P+','P-']:
            Z0 = params.get('Z0', self.Zbase)
            if sigid == 'A':
                return (Vsig + Z0 * Isig)/2
            elif sigid == 'B':
                return (Vsig - Z0 * Isig)/2
            elif sigid == 'P+':
                return (np.abs(Vsig + Z0 * Isig)/2)**2/(2*Z0)
            elif sigid == 'P-':
                return (np.abs(Vsig - Z0 * Isig)/2)**2/(2*Z0)                
            elif sigid in ['G','G+']:
                return (Vsig - Z0 * Isig)/(Vsig + Z0 * Isig)
            else: # sigid == 'G-'
                return (Vsig + Z0 * Isig)/(Vsig - Z0 * Isig)
            # endif
            
        elif sigid in ['V','I','Z','Z+','Z-','Y']:
            if sigid == 'V':
                return Vsig
            elif sigid == 'I':
                return Isig
            elif sigid in ['Z', 'Z+','Z-']:
                try :
                    Z = Vsig/Isig
                    if sigid == 'Z-':
                        return -Z
                    else: # sigid in ['Z','Z+']
                        return Z
                    # endif
                except ZeroDivisionError:
                    return np.inf
                # endtry
            else: # sigid == 'Y'
                try :
                    return Isig/Vsig
                except ZeroDivisionError:
                    return np.inf
                # endtry
            # endif
            
        elif sigid[0:3] in ['P','PWR']:
            P = 0.5 * np.real(Vsig * np.conjugate(Isig))
            if sigid[-1] == '-':
                return -P
            else:
                return P
            # endif
            
        elif sigid in ['VSW','ISW','VISW','VMAX','IMAX']:
            L = params.get('L', 0.)
            if L < 0.:
                L = -L
                Isig = -Isig # reverse the direction
            Z0 = params.get('Z0',self.Zbase)
            Lwave = params.get('Lwave', 
                               params.get('v0',1.) * speed_of_light/self.fHz)
            v0 = params.get('v0',Lwave/(speed_of_light/self.fHz))
            dx = params.get('dx',Lwave/72)
            if sigid[-2:] == 'SW':
                VISW = TL_Vmax(L,Z0,Lwave,V=Vsig,I=Isig,dx=dx)[2]
                if sigid == 'VISW':
                    return {'V':VISW[1], 'I':VISW[2], 'x':VISW[0]}
                elif sigid == 'VSW':
                    return {'V':VISW[1], 'x':VISW[0]}
                else : # sigid == 'ISW'
                    return {'I':VISW[2], 'x':VISW[0]}
                # endif
            else:
                if sigid == 'VMAX':
                    VISW = TL_Vmax(L,Z0,Lwave,V=Vsig,I=Isig)
                else: # ssigif == 'IMAX'
                    VISW = TL_Imax(L,Z0,Lwave,V=Vsig,I=Isig)
                #endif
                return {'max':VISW[0],'x':VISW[1]}
            # endif

        else:
            raise ValueError("calcsignals : signal type " +
                             sigid + " not implemented")
        # endif
        
    #===========================================================================
    #  _ _ i a d d _ _
    #
    def __iadd__(self, other):
        """
        merges two Scatter classes into one except if the frequencies differ.
        If Zbase and/or iwt differ the second operand is converted so that the
        result has the Zbase and iwt of the first operand.
        Ports with equal names are resolved (i.e. concected and eliminated)
        """
        
        ## use overloaded "+=" to combine Scatter objects
        ## note the first operand determines the Zbase and iwt of the result

        ## do some basic checking we do not mix apples with tomatoes
        if not isinstance(other,Scatter):
            raise ValueError("Scatter.__iadd__ : type mismatch")

        if self.fMHz != other.fMHz:
            raise ValueError("Scatter.__iadd__ : frequency mismatch")
        
        if self.epsr != other.epsr or self.mur != other.mur or self.V != other.V:
            raise ValueError("Scatter.__iadd__ : medium mismatch")
        

        if self.Zbase != other.Zbase:
            self += other.convert_Zbase(self.Zbase)
            return self

        if self.iwt != other.iwt:
            self += other.flip_iwt()
            return self

        ## take care of trivial cases
        if not len(self.S):
            self = deepcopy(other)
            return self
        
        elif not len(other.S):
            return self
        
        else: # non-trivial case
            oldself = deepcopy(self)
            
            ## we need to expand the S and Ssigs matrices as well as list of
            # port names and signal names

            Ns, No = len(oldself.S), len(other.S)
            r1 = np.hstack((  oldself.S  ,  np.zeros((Ns,No))  ))
            r2 = np.hstack((  np.zeros((No,Ns))  ,  other.S  ))
            self.S = np.vstack((r1,r2))
            self.Portname = oldself.Portname + other.Portname
            

            Ss, So = len(oldself.Ssigs), len(other.Ssigs)
            self.Signal = oldself.Signal + other.Signal
            if Ss:
                if So:
                    self.Ssigs = np.vstack(
                                     (np.hstack((oldself.Ssigs, np.zeros((Ss,No)))),
                                      np.hstack((np.zeros((So,Ns)),other.Ssigs)))
                                   )
                else:
                    self.Ssigs = np.hstack((oldself.Ssigs, np.zeros((Ss,No))))
            elif So:
                self.Ssigs =  np.hstack((np.zeros((So,Ns)), other.Ssigs))
            else:
                self.Ssigs = np.zeros((0,0), dtype=np.complex128)
            # endif
            
            ### Join ports with the same names now

            kp = 0
            while kp < len(self.Portname):
                # find all ports with same name
                ids = [k for k, nk in enumerate(self.Portname)
                         if nk == self.Portname[kp]]
                if len(ids) > 1:
                    kp = 0  ## reset the port counter
                    if len(ids) > 2:
                        ## keep the port for subsequent connections
                        self.joinports(ids[0],ids[1],newname=self.Portname[kp])
                    else:
                        ## don't need to keep the port
                        self.connectports(ids[0],ids[1])
                    # endif
                else:
                    kp += 1 ## go to next name in the list of ports
                # endif
            # endwhile

        return self
    
    #===========================================================================
    #  _ _ a d d _ _
    #
    def __add__(self,other):
        """
        merges two Scatter classes into one except if the frequencies differ.
        If Zbase and/or iwt differ the second operand is converted so that the
        result has the Zbase and iwt of the first operand.
        Ports with equal names are resolved (i.e. concected and eliminated)
        """

        ## use overloaded "+" to combine Scatter objects
        ## note the first operand determines the Zbase and iwt of the result
        
        result = deepcopy(self)
        result += other
        return result
    
#         #FIXME: new properties are not handled correctly !! (v, mur, epsr)
# 
#         ## do some basic checking we do not mix apples with tomatoes
#         if not isinstance(other,Scatter):
#             raise ValueError("Scatter.__add__ : type mismatch")
# 
#         if self.fMHz != other.fMHz:
#             raise ValueError("Scatter.__add__ : frequency mismatch")
#         
#         if self.epsr != other.epsr or self.mur != other.mur or self.V != other.V:
#             raise ValueError("Scatter.__add__ : medium mismatch")
#         
# 
#         if self.Zbase != other.Zbase:
#             return self + other.convert_Zbase(self.Zbase)
# 
#         if self.iwt != other.iwt:
#             return self + other.flip_iwt()
# 
# 
#         ## take care of trivial cases
#         if not len(self.S):
#             return deepcopy(other)
#         
#         elif not len(other.S):
#             return deepcopy(self)
#         
#         else: # non-trivial case
#             result = deepcopy(self)
# 
#             ## we need to expand the S and Ssigs matrices as well as list of
#             # port names and signal names
# 
#             Ns, No = len(self.S), len(other.S)
#             r1 = np.hstack((  self.S  ,  np.zeros((Ns,No))  ))
#             r2 = np.hstack((  np.zeros((No,Ns))  ,  other.S  ))
#             result.S = np.vstack((r1,r2))
#             result.Portname = self.Portname + other.Portname
# 
#             Ss, So = len(self.Ssigs), len(other.Ssigs)
#             result.Signal = self.Signal + other.Signal
#             if self.Signal:
#                 if other.Signal:
#                     result.Ssigs = np.vstack(
#                                      (np.hstack((self.Ssigs, np.zeros((Ss,No)))),
#                                       np.hstack((np.zeros((So,Ns)),other.Ssigs)))
#                                    )
#                 else:
#                     result.Ssigs = np.hstack((self.Ssigs, np.zeros((Ss,No))))
#             elif other.Signal:
#                 result.Ssigs =  np.hstack((np.zeros((So,Ns)), other.Ssigs))
#             else:
#                 result.Ssigs = []
#             # endif
#             
#             ### Join ports with the same names now
# 
#             kp = 0
#             while kp < len(result.Portname):
#                 # find all ports with same name
#                 ids = [k for k, nk in enumerate(result.Portname)
#                          if nk == result.Portname[kp]]
#                 if len(ids) > 1:
#                     kp = 0  ## reset the port counter
#                     if len(ids) > 2:
#                         ## keep the port for subsequent connections
#                         result.joinports(ids[0],ids[1],result.Portname[kp])
#                     else:
#                         ## don't need to keep the port
#                         result.connectports(ids[0],ids[1])
#                     # endif
#                 else:
#                     kp += 1 ## go to next name in the list of ports
#                 # endif
#             # endwhile
#      
#             
#             return result

    #===========================================================================
    #
    # _ _ sub _ _
    #
    def __sub__(self, other, maxerr=1E-12):
        """substract two circuits == de-embed other from self
        
           self(c1, c2, ..., cN, e1..., eQ) - other(i1, ... iP, e1, ..., eQ)
           
           where ck, ik and ek are ports
           
           ==> SZ(c1, ..., cN, i1, ..., iP) 
           
           
                       +-------------------------------------+
                       |                                     |
                       +----------+               +----------+
                       |          |               |          |
                      <c1>      <i1> ----------- <i1>      <e1>
                       :          :               :          :
           com_ports   :          :   int_ports   :          :   ext_ports
                       :          :               :          :
                      <cN>      <iP> ----------- <iP>      <eQ>
                       |          |               |          |
                       +----------+               +----------+
                       | internal                    deembed |
                       |  object == result   other == object |        
                       +-------------------------------------+
                            external object == self
        
           Q must be larger or equal than P
        """
        # need to do some basic checks on the compatibility of the 2 objects
        
        if self.fMHz != other.fMHz :
            raise ValueError(
                'Scatter.__sub__ (de-embed) : frequency mismatch (%f != %f)' %
                (self.fMHz,other.fMHz))
        
        if self.Zbase != other.Zbase:
            print('changing Zbase')
            other = other.convert_Zbase(self.Zbase)
            
        if self.iwt != other.iwt:
            print('changing iwt')
            other = other.flip_iwt()
            
        # need to find the ports common to the internal and external object,
        # (com_ports) the ports of the external object that will be de-embedded
        # (ext_ports) into the ports of the internal object (int_ports)
        
        ext_ports = [p for p in other.Portname if p in self.Portname]
        com_ports = [p for p in self.Portname if p not in ext_ports]
        int_ports = [p for p in other.Portname if p not in ext_ports]
        
        N, P, Q = len(com_ports), len(int_ports), len(ext_ports)
        # print('exp=\n',ext_ports)
        # print('com=\n',com_ports)
        # print('int=\n',int_ports)
        
        if P > Q:
            raise ValueError(
                'Scatter.__sub__ (de-embed) : #internal ports (%d) '
                'must be less or equal to the #external ports (%d)' %(P, Q))
            
        # sort the ports
        self.sortports(com_ports+ext_ports)
        # print('self=\n',self)
        other.sortports(int_ports+ext_ports)
        # print('other=\n',other)
        
        # extract the sub-matrices
        Dii, Die = other.S[:P, :P], other.S[:P, P:]
        Dei, Dee = other.S[P:, :P], other.S[P:, P:]
        
        # print('Dii=\n',Dii)
        # print('Die=\n',Die)
        # print('Dei=\n',Dei)
        # print('Dee=\n',Dee)
        
        Eee, Eed = self.S[:N, :N], self.S[:N, N:]
        Ede, Edd = self.S[N:, :N], self.S[N:, N:]
        
        # print('Eee=\n',Eee)
        # print('Eed=\n',Eed)
        # print('Ede=\n',Ede)
        # print('Edd=\n',Edd)
        
        
        IDie = np.linalg.pinv(Die) # Q x P
        IDei = np.linalg.pinv(Dei) # P x Q
        
        # print('IDie=\n',IDie)
        # print('IDei=\n',IDei)
        
        # check that the inverses make sense
        
        IPP = np.max(np.abs(np.dot(Die, IDie) - np.eye(P)))
        # print('IPP=',IPP)
        if IPP > maxerr:
            raise ValueError(
                'Scatter.__sub__ (de-embed) : pseudo inverse Die error (%g) '
                'larger than allowable maximum error (%g)' % (IPP,maxerr))
            
        IPP = np.max(np.abs(np.dot(IDei, Dei) - np.eye(P)))
        # print('IPP=',IPP)
        if IPP > maxerr:
            raise ValueError(
                'Scatter.__sub__ (de-embed) : pseudo inverse Dei error (%g) '
                'larger than allowable maximum error (%g)' % (IPP,maxerr))
        
        M = np.dot(IDei, np.dot(Edd - Dee, IDie))
        # print('M=\n',M)
        
        Kii = np.dot(M,np.linalg.inv(np.eye(P)+ np.dot(Dii,M)))
        # print('Kii=\n',Kii)
        IKiiDii = np.eye(P)-np.dot(Kii,Dii)
        IDiiKii = np.eye(P)-np.dot(Dii,Kii)
        Kie = np.dot(IKiiDii,np.dot(IDei,Ede))
        # print('Kie=\n',Kie)
        Kei = np.dot(np.dot(Eed,IDie),IDiiKii)
        # print('Kei=\n',Kei)
        invIDiiKii = np.linalg.inv(IDiiKii)
        Kee = Eee - np.dot(np.dot(np.dot(Kei,invIDiiKii),Dii),Kie)
        # print('Kee=\n',Kee)
        
        # print('stacked Kxx\n')
        # print(np.vstack((np.hstack((Kee,Kei)),
        #                  np.hstack((Kie,Kii)))))
        
        SZ = Scatter(fMHz = self.fMHz, 
                     Zbase = self.Zbase,
                     S = np.vstack((np.hstack((Kee,Kei)),
                                    np.hstack((Kie,Kii)))),
                     Portnames = com_ports + int_ports)
        
        # FIXME: now we have to work out the signal transformations
        
        return SZ
        
    #===========================================================================
    #  s o r t p o r t s 
    #
    def sortports(self, order=str.__lt__):
        """
        order :
            None : do nothing

            [ portnames ] : rearrange as per given list

            a function : default : str.__lt__ (ascending order)
            
                         as we are comparing strings the function should be a
                         function of 2 string returning a boolean :

                         if order(Scatter.Portname[k2], Scatter.Portname[k1]) :
                            # where k1 < k2
                            Scatter.swapports(k1,k2)
        """

        if order :
            if type(order) == list:
                for k1, name in enumerate(order) :
                    if self.Portname[k1] != name :
                        self.swapports(k1, self.findport(name))
                    # endif
                #endfor
        
            else: # we assume order is a boolean function of 2 strings
                try:
                    if type(order(self.Portname[0],self.Portname[0])) is bool:
                        for k1 in range(len(self.Portname)-1):
                            for k2 in range(k1,len(self.Portname)):
                                if order(self.Portname[k2], self.Portname[k1]):
                                    self.swapports(k1,k2)
                                # endif
                            # endfor
                        # endfor
                    else:
                        raise TypeError('Scatter.sortports : order does not appear '
                                        'to return a boolean')
                    # endif
                except TypeError:
                    raise TypeError('Scatter.sortports : order does not appear '
                                    ' to be a function of 2 strings')

                except:
                    print('Scatter.sortports : unidentified error')
                    raise
                # endtry
            #endif
        #endif
                
        return self # e.g. so you can do : SZ = (SZA + SZB).sortports()
    
    #===========================================================================
    #  s w a p p o r t s
    #
    def swapports(self,port1,port2):
        """
        ports can be their index number as well as a name
        """
        ## replace ports passed as named by their index

        p1, p2 = self.findport(port1), self.findport(port2)
        
        if p1 != p2 :
            ## swap the names 
            self.Portname[p1], self.Portname[p2] = (self.Portname[p2],
                                                    self.Portname[p1])
            
            ## swap the S-matrix coefficients (this is done in place but force
            ## copies)

            #swap column
            self.S[:,p1], self.S[:,p2] = self.S[:,p2].copy(), self.S[:,p1].copy()
            # swap row
            self.S[p1,:], self.S[p2,:] = self.S[p2,:].copy(), self.S[p1,:].copy()
            
            ## swap the ports in the Signal definitions (this is done in place)
            ## force copies
            if self.Signal:
                self.Ssigs[:,p1], self.Ssigs[:,p2] = (self.Ssigs[:,p2].copy(),
                                                      self.Ssigs[:,p1].copy())
        # endif
        
    #===========================================================================
    #  t e r m p o r t
    #
    def termport(self,port1,rho=(1+0j)):
        """
        Terminates a port (given by index number or name) on a supplied reflection
        coefficient (defaults to an open circuit)
        """
        p1 = self.findport(port1)
        
        nP1 = len(self.S)-1
        if p1 != nP1:
            self.swapports(p1,nP1)
        # endif
        S11 = self.S[:nP1,:nP1]
        S1n = self.S[nP1,:nP1].reshape((1,nP1))
        # Sn1 = self.S[:nP1,nP1]
        k1S1n = rho/(1-self.S[-1,-1]*rho)*S1n
        self.S = S11 + np.dot(self.S[:nP1,nP1].reshape((nP1,1)), k1S1n)
        if len(self.Ssigs):
            # print(len(self.Ssigs))
            # print('%r' % self.Ssigs[:,:-1])
            
            self.Ssigs = self.Ssigs[:,:-1] + np.dot(self.Ssigs[:,[-1]],k1S1n)
        # endif
        self.Portname.pop(-1)

    #==========================================================================
    #  N p o r t I d e a l N o d e
    #
    def NportIdealNode(self,N, Portnames = 'IdealNode'):
        """
        return ideal N-port node :
        """
        if not (type(Portnames) is list and len(Portnames) is N):
            if type (Portnames) is not str:
                raise ValueError('Scatter.NportIdealNode : Portname must be'
                                 'of type str or list of str.')
            pnames = [Portnames + '-%d' % (k+1) for k in range(N)]
        else :
            pnames = Portnames
        
        a, b = - (N-2)/N+0j, 2/N+0j # make complex
        S = np.array([[a if i is j else b for j in range(N)] for i in range(N)])
            
        return Scatter(fMHz=self.fMHz, Zbase=self.Zbase, iwt=self.iwt,
                       V=self.V, epsr=self.epsr, mur=self.mur,
                       S=S, Portnames=pnames)
        
    #==========================================================================
    #  TL_VImatrix
    #
        
    def TL_VImatrix(self, lngth, **TLkwargs):
        """
        returns the VI transform matrix of a TL with length lngth and properties
        given in TLkwargs
        
        """
                
        return VI_from_S(self.TL_Smatrix(lngth, **TLkwargs), self.Zbase)

    #==========================================================================
    #  TL_Smatrix
    #
    
    def TL_Smatrix(self, lngth, **TLkwargs):
        """
        returns the scattering matrix of a TL with length lngth and properties
        given in TLkwargs
        
        """
        
        gamma, Zc = self.TLproperties(**TLkwargs)
        
        gml = gamma * lngth
        sinhgml = np.sinh(gml)
        coshgml = np.cosh(gml)
            
        N = (Zc**2+self.Zbase**2)*sinhgml + 2*Zc*self.Zbase*coshgml
        s = (Zc**2 - self.Zbase**2)*sinhgml / N
        t = 2*Zc*self.Zbase / N
        
        return np.array([[s,t],[t,s]])
    
    #==========================================================================
    #  A d d T L
    #
    
    def AddTL(self, port_s, lngth, **kwargs):
        """ AddTL
        
            wrapper for trlAZV and trlGRLC
        
            lngth : length of TL [m]
            port_s : str | [str, str]
            
            kwargs:
                        
            (AZV type (default))
            
            A: attenuation [m-1] (default 0.)
            Z: characteristic impedance [Ohm] (default self.Zbase)
            V: wave velocity relative to self.c0 (default 1.)
            
            ((G)RLC type)
            
            G: specific conductance [Mho/m] (default 0.)
            R: specific resistance [Ohm/m] (default 0.)
            L: specific inductance [nH/m] (default 1E9 * self.Zbase/self.c0)
            C: specific capcitance [pF/m] (default 1E12 /(self.Zbase*self.c0))
            
            strict_kwargs : if False ingnore unknown kwargs (default True)
            
        """
        
        if not (isinstance(port_s,str) or 
                (isinstance(port_s, list) and len(port_s) is 2 
                 and all([isinstance(p,str) for p in port_s]))
               ):
            raise ValueError(
                'Scatter.AddTL: ports must be a string or a list of 2 strings')        
        
        self.extendport(port_s, self.TL_Smatrix(lngth, **kwargs))

        
    #==========================================================================
    #  T L p r o p e r t i e s
    #
    def TLproperties(self,**TLkwargs):
        """
        returns the complex propagation (gamma) and characteristic impedance (Zc)
        of a TL with given TLkwargs properties:
        
        AZV (default):
        
        A: attenuation [1/m] (default 0.)
        Z: (lossless thus real) characteristic impedance [Ohm] default self.Zbase)
        V: wave velocity relative to the speed of light (self.c0) (default 1.)
        
        GRLC (defaults equivallent to default AZV):
        
        G: specific conductance [Mho/m] (default 0.)
        R: specific resistance [Ohm/m] (default 0.)
        L: specific inductance [nH/m] (default 1E9 * self.c0/self.Zbase)
        C: specific capacitance [pF/m] (default 1E12 / (self.c0 * self.Zbase))
        
        lossless : also return the lossless beta and Z0
        
        strictly_TLdata : if True (default) don't allow unknown kwargs
        
        This is not implemented yet:
        coaxial geometry:
        
        OD : TL outer diameter [m]
        ID : TL inner diameter [m]
        OR : TL outer radius [m]
        IR : TL inner radius [m]
        
        R : specific resistance [Ohm/m] = Ro + Ri
        R_o : TL outer specific resistance [Ohm/m]
        R_i : TL inner specific resistance [Ohm/m]
        
        rho : TL material resistivity [Ohm.m]
        rho_o : TL outer material resitivity [Ohm.m]
        rho_i : TL inner material resitivity [Ohm.m]
        
        eps_r : relative permittivity [1] (sefault self.epsr)
        mu_r : relative permeability [1] (default self.mur)
        sigma : medium conductivity [Mho.m] (default 0)
        tand : medium tangent delta [1] (default 0)
        
        """
#         #----------------------------------------------------- medium properties
#         epsr = TLkwargs.pop('eps_r', self.epsr)
#         mur = TLkwargs.pop('mu_r', self.mur)
#         sigma = TLkwargs.pop('sigma', 0.)
#         tand = TLkwargs.pop('tand', 0)
#         
#         #------------------------------------------------------ coaxial geometry
#         OD = TLkwargs.pop('OD', None)
#         if not OD and 'OR' in TLkwargs:
#             OD = 2 * TLkwargs.pop('OR')
#         
#         ID = TLkwargs.pop('ID', None)
#         if not ID and 'IR' in TLkwargs:
#             ID = 2 * TLkwargs.pop('IR')
#             
#         #------------------------------------------------- conductors properties
#         
#         #---- first try to find the fiundamental inputs like G, R, L, C
#         # 
#         #---- then lower level inputs like rho, ID, OD, sigma, Z, V, A, which will
#         #       allow to find G, R, L, C
#         
#         G = TLkwargs.pop('G', None)
#         R = TLkwargs.pop('R', None)
#         L = TLkwargs.pop('L', None)
#         C = TLkwargs.pop('C', None)
#         
#         if L is None and C is None:
#             # do we have Z and V then
#             Z = TLkwargs.pop('Z', None)
#             V = TLkwargs.pop('V', None)
#             if V is None:
#                 # we do have epsr and mur then
#                 V = 1 / np.sqrt(epsr * mur)
#             if Z is None:
#                 # do we have OD(R) and ID(R) then
#                 if ID is None or OD is None:
#                     raise ValueError('Scatter.TLproperties: cannot determine Z')
#                 Z = np.sqrt(mur/epsr*mu_0/epsilon_0) / (2*np.pi) * np.log(OD/ID)
#             L = Z / (V * speed_of_light) * 1E9      # nH/m
#             C = 1 / (Z * V * speed_of_light) * 1E12 # pF/m
#             
#         elif L is None:
#             # we have C : do we have Z or V
#             pass
#             
#             
#         if 'rho' in TLkwargs:
#             rho = TLkwargs.pop('rho', 0.)
#             rho_i = TLkwargs.pop('rho_i', rho)
#             rho_o = TLkwargs.pop('rho_o', rho)
#             
#         elif 'rho_i' in TLkwargs:
#             rho_i = TLkwargs.pop('rho_i')
#             rho_o = TLkwargs.pop('rho_o', 0.)
#         
#         else:
#             rho_o = TLkwargs.pop('rho_o', 0.)
#             rho_i = 0.
#         
#         
#          
#         #--------------------------------------------------------- TL properties
#         if OD and ID:
#             # geometry overrides Z
#             Z = np.sqrt(mur/epsr*mu_0/epsilon_0) / (2*np.pi) * np.log(OD/ID)
#             
#         
        if any_of(['R','G'], TLkwargs) and any_of(['Z','V'],TLkwargs):
            # (special case) 'G', 'R', 'Z', 'V' -> 'G', 'R', 'L', 'C'
            G = TLkwargs.pop('G', 0.)
            R = TLkwargs.pop('R', 0.)
            Z = TLkwargs.pop('Z',self.Zbase)
            V = TLkwargs.pop('V', self.V)
            if TLkwargs:
                raise ValueError('Scatter/TLproperties: incompatible mix of '
                                 'parameters: R, G, Z, V excl. or A, L, C')
            L = Z / (self.c0 * V) * 1E9
            C = 1 / (Z * self.c0 * V) * 1E12
            
            jwL = self.iwt * 1j * self.w * L * 1e-9
            jwC = self.iwt * 1j * self.w * C * 1e-12
           
            gamma = np.sqrt((G + jwC) * (R + jwL)) 
            Zc = (R + jwL) / gamma
            
            beta = np.imag(np.sqrt(jwL * jwC))
            Z0 = np.sqrt(jwL / jwC)
            
        elif any_of(['G','R','L','C'], TLkwargs):
            if any_of(['A','Z','V'], TLkwargs):
                raise ValueError('Scatter.TLproperties: incompatible mix of '
                                 'parameters: A, Z, V excl. or G, R, L, C')
            
            G = TLkwargs.pop('G', 0.)
            R = TLkwargs.pop('R', 0.)
            C = TLkwargs.pop('C', 1E12 / (self.c0 * self.V * self.Zbase)) # pF/m
            L = TLkwargs.pop('L', 1E9 * self.Zbase / (self.c0 * self.V))  # nH/m
            
            jwL = self.iwt * 1j * self.w * L * 1e-9
            jwC = self.iwt * 1j * self.w * C * 1e-12
           
            gamma = np.sqrt((G + jwC) * (R + jwL)) 
            Zc = (R + jwL) / gamma
            
            beta = np.imag(np.sqrt(jwL * jwC))
            Z0 = np.sqrt(jwL / jwC)

        else:
            A = TLkwargs.pop('A', 0.)
            Z = TLkwargs.pop('Z',self.Zbase)
            V = TLkwargs.pop('V', self.V)

            k0 = self.iwt * self.w / (self.c0 * V)
            c1 = A / k0
            cc = np.sqrt(1. + c1**2) - 1j*c1

            gamma = 1j * k0 * cc
            Zc = Z * cc
            
            beta = k0
            Z0 = Z
            
        lossless = TLkwargs.pop('lossless', False)
            
        strict_kwargs = TLkwargs.pop('strict_kwargs', True)
        
        if strict_kwargs and TLkwargs:
            raise ValueError('Scatter.TLproperties: unknown arguments: ' + 
                             str([k for k in kwargs]))
            
#         print('TLproperties: gamma= %+10.6f%+10.6fj, Zc= %+10.6f%+10.6fj' %
#               (gamma.real, gamma.imag, Zc.real, Zc.imag))
        
        self.lastTL = {'gamma':gamma,'Zc':Zc,'beta':beta,'Z0':Z0}
        
        if lossless:
            return gamma, Zc, beta, Z0
        else:
            return gamma, Zc

    #==========================================================================
    #
    #  t r l R L C
    #
    def trlRLC(self, port1, lngth, R=0, L=None, C=None):
        """
        trlRLC : inserts the S-matrix of a uniform transmission line with
                 specific resistance R [Ohm/m], inductance L [nH/m], capacitance
                 C [pF/m] and length lngth [m].

        port1 is either a single port (name or index number or a list of 2 ports
        (names or index numbers). In the case a single port is supplied that port
        'extended' with the specified transmission line.
        
        Defaults :
        
         if neither L and C are given the specific inductance and capacitance of
         a TL with characteristic impedance Scatter.Zbase and wave velocity equal
         to the speed of light in vacuum is used;

         if either C or L are given the other one is such that the wave velocity
         equals the speed of light in vacuum

         the specific resistance R defaults to 0.
        """
        
        self.trlGRLC(port1, lngth, G=0, R=R, L=L, C=C)

    #===========================================================================
    #
    #  t r l G R L C 
    #
    def trlGRLC(self, port1, lngth, G=0, R=0, L=None, C=None):
        """
        trlGRLC : inserts the S-matrix of a uniform transmission line with
                  specific resistance R [Ohm/m], G [Mho/m], inductance L [nH/m],
                  capacitance C [pF/m]mand length lngth [m].
             
        port1 is either a single port (name or index number or a list of 2 ports
        (names or index numbers). In the case a single port is supplied that port
        'extended' with the specified transmission line.
        
        Defaults :
        
         if neither L and C are given the specific inductance and capacitance of
         a TL with characteristic impedance Scatter.Zbase and wave velocity equal
         to the speed of light in vacuum is used;

         if either C or L are given the other one is such that the wave velocity
         equals the speed of light in vacuum

         the specific resistance R and conductance G default to 0.
        """

        if L == None:
            if C == None:
                # if L nor C are supplied, they default
                jwL = self.iwt * 1j * self.w * (self.Zbase / (self.V * self.c0)) 
                jwC = self.iwt * 1j * self.w / (self.Zbase * (self.V * self.c0))
            else:
                jwL = self.iwt * 1j * self.w / (self.c0**2 * C * 1e-12)
                jwC = self.iwt * 1j * self.w * C * 1e-12
        elif C == None :
            jwC = self.iwt * 1j * self.w / (self.c0**2 * L * 1e-9)
            jwL = self.iwt * 1j * self.w * L * 1e-9
        else:
            jwL = self.iwt * 1j * self.w * L * 1e-9
            jwC = self.iwt * 1j * self.w * C * 1e-12
            
        gml = np.sqrt((G + jwC) * (R + jwL))
        Zc = (R + jwL) / gml
        gml *= lngth
 
        sinhgml = np.sinh(gml)
        coshgml = np.cosh(gml)

        N = (Zc**2+self.Zbase**2)*sinhgml + 2*Zc*self.Zbase*coshgml
        s = (Zc**2 - self.Zbase**2)*sinhgml / N
        t = 2*Zc*self.Zbase / N
        
        self.extendport(port1,np.array([[s,t],[t,s]]))

    #===========================================================================
    #  t r l A Z V
    #
    def trlAZV(self,ports, L=0., A=0., Z=None, V=None):
        """
        trlAZV : inserts the S-matrix of a uniform transmission line with
             attenuation A [1/m], characteristic impedance Z [Ohm], propagation
             velocity V [relative to c0, the speed of light in vacuum] and
             length lngth [m].

        ports is either a single port (name or index number or a list of 2 ports
        (names or index numbers). In the case a single port is supplied that port
        'extended' with the specified transmission line.
        
        Defaults :
        
         the characteristic mpedance Z defaults to Scatter.Zbase
         the attenuation defaults to 0. (note that the formulas assume that
           A is only due to conductor resistive losses)
         the wave velocity relative the speed of light in vacuum V defaults to
           the one set when creating the Scatter object (default 1. if not given)
        """
        
        if Z == None:
            Z = self.Zbase
            
        if V == None:
            V = self.V
            
        k0 = self.w / (self.c0 * V)
        
        if (A == 0) and (Z == self.Zbase):
            s = 0j                         ### can use simpler formulas
            t = np.exp(-self.iwt*1j* k0 *L)
            Zc, gamma = self.Zbase, self.iwt * 1j * k0
            
        else:
            
            c1 = A / k0
            cc = np.sqrt(1. + c1**2) - self.iwt*1j*c1
            gamma = self.iwt*1j*cc*k0
            gml = gamma*L
            sinhgml = np.sinh(gml)
            coshgml = np.cosh(gml)
            Zc = Z * cc
            
            N = (Zc**2+self.Zbase**2)*sinhgml + 2*Zc*self.Zbase*coshgml
            s = (Zc**2 - self.Zbase**2)*sinhgml / N
            t = 2*Zc*self.Zbase / N
        
#         print('trlAZV: gamma= %+10.6f%+10.6fj, Zc= %+10.6f%+10.6fj' %
#               (gamma.real, gamma.imag, Zc.real, Zc.imag))
        
        self.extendport(ports,np.array([[s,t],[t,s]]))
        
        # as a side effect save the Zc and gamma used
        self.Zc = Zc
        self.gamma = gamma
        
##    #===========================================================================
##    #  t r l C o a x
##    #
##    def trlCoax(self, ports, length, **kwargs):
##        """
##        general coax parameters are combined :
##        
##        epsr, mur          : relative permitivity and permeability of the medium
##        rhoi, rhoo (, rho) : conductor resistivities (ohm.m) for inner and outer
##                             (or both)
##        Ri, Ro (, R)       : specific resistance (ohm/m) for inner and outer
##                             (or total)
##        ID, OD             : inner and outer diameters [m] (unless there are no
##                             losses : rhoi, rhoo, rho, Ri, Ro, R are all 0 : in
##                             this case ID and OD can be in AU)
##        L, C               : specific inductance (nH/m) and capacitance (pF/m)
##        Z, V               : characteristic impedance (ohm) and wave propagation
##                             (relative to c0 = <speed_of_light> m/s)
##        G                  : specific conductivity (mho/m)
##
##        e.g. SZ.trlCoax(['p1','p2'],1.0,ID=100.,OD=230.,rho=2e-8)
##
##        Hierarchy :
##
##        the algorithm will try and find values for V, Z and A first
##
##        for V if not explicitly given, it will first try to obtain V from
##        L and C if given
##        
##        """
##        
##        tol = kwargs.get('tol',1e-6)
##
##        C0 = self.c0
##        MU0 = 4e-7*np.pi
##        EPS0 = 1/(MU0 * C0**2)
##        ZVAC = MU0 * C0 / (2 * np.pi)
##        PF = 1e-12
##        NH = 1e-9
##        
##        # parameter defining a lossless TL
##        P = ['Z','V','L','C','mur','epsr','lnDs']
##
##        #.......................................................................|.....
##
##        def solvit(**quargs):
##            
##            params = quargs.keys()
##            paramset = set(params)
##
##            # list the variables we need
##            needP = [v for v in P if v not in params]
##
##            # the equation matrix becomes
##            M, RHS = [], []
##            for eq in E:
##                meq, req = [], eq.get('K')
##                for v in P:
##                    if v in needP:
##                        meq.append(eq.get(v,0))
##                    else: # so we have v
##                        req -= eq.get(v,0)*np.log(quargs.get(v))
##                M.append(meq)
##                RHS.append([req])
##
##            rankM = np.linalg.matrix_rank(M)
##
##            if quargs.get('info'):
##                print('-'*40)
##                print([v for v in P if v in params])
##                print(needP)
##                for meq, rh in zip(M,RHS):
##                    for mc in meq:
##                        print('%5.1f' % mc, end='')
##                    print(' = %7.4f' % rh[0])
##                print(rankM)
##                print()
##                
##            if rankM >= len(needP):
##                # note for rankM > len(needP) we may be overconstrained
##                sols = np.linalg.lstsq(M,RHS)
##                sol = dict([(p, np.exp(v[0])) for p, v in zip(needP,sols[0])])
##
##                # check the solution to see if it is overconstrained or not
##                RHS = []
##                for eq in E:
##                    RHS.append(eq.get('K'))
##                    for v in P :
##                        if v in quargs.keys():
##                            RHS[-1] -= eq.get(v,0)*np.log(quargs.get(v))
##                        else:
##                            RHS[-1] -= eq.get(v,0)*np.log(sol.get(v))
##                maxErr = np.abs(np.max(RHS))
##
##                # check if the solution is physical
##                eq_ = lambda x, v: np.abs(sol.get(x,quargs.get(x)) - v) < tol
##                le_ = lambda x, v: sol.get(x,quargs.get(x)) <= v+tol
##                ge_ = lambda x, v: sol.get(x,quargs.get(x)) >= v-tol
##
##                physical = ((eq_('V',1.) and eq_('mur',1.) and eq_('epsr',1.)) or
##                            (le_('V',1.) and ge_('mur',1.) and ge_('epsr',1.)))
##            else:
##                sols = np.linalg.lstsq(M,RHS)
##                sol = {}
##                maxErr, RHS = 0, [0 for eq in E]
##                physical = False
##            
##            if quargs.get('info'):
##                print(sol)
##                print(sols)
##                
##            return sol, RHS, maxErr, physical
##
##        #.......................................................................|.....
##
##        def solvit2(**quargs):
##            haveP = [v for v in P if v in quargs.keys()]
##            needP = [v for v in P if v not in haveP]
##
##            if len(haveP) < 1:
##                # everything is default
##                quargs.update({'Z'    : self.Zbase,
##                               'V'    : self.V,
##                               'mur'  : self.mur,
##                               'epsr' : self.epsr,
##                               'L'    : self.Zbase / (C0 * self.V) / NH,
##                               'C'    : 1 / (self.Zbase * C0 * self.V) / PF,
##                               'lnDs' : (np.sqrt(self.mur/self.epsr) *
##                                         self.Zbase / ZVAC)})
##                
##            elif len(haveP) < 2:
##                if 'Z' in haveP or 'lnDs' in haveP:
##                    # if we only have these assume default V and mur
##                    quargs.update({'V'    : self.V,
##                                   'mur'  : self.mur,
##                                   'epsr' : self.epsr})
##                    
##                    Zr = np.sqrt(self.mur/self.epsr)
##                    
##                    if 'Z' in haveP:
##                        Z = quargs.get('Z')
##                        lnDs = Z / (Zr * ZVAC)
##                        quargs.update({'lnDs' : lnDs})
##                        
##                    elif 'lnDs' in haveP:
##                        lnDs = quargs.get('lnDs')
##                        Z = Zr * ZVAC * lnDs 
##                        quargs.update({'Z' : Z})
##
##                    quargs.update({'L': Z / (self.V * C0) / NH,
##                                   'C': 1 / (Z * self.V * C0) /PF})
##                        
##                elif 'V' in haveP or 'epsr' in haveP or 'C' in haveP :
##                    # assume V or C (or epsr) is due to epsr and most materials
##                    # have mur=1, however we will use the Scatter.mur as default
##                    # (likely to be 1)
##                    
##                    quargs.update({'mur'  : self.mur})
##
##                    if 'V' in haveP:
##                        quargs.update({
##                            'epsr': 1 / quargs.get('V')**2 * self.mur})
##                    else:
##                        if 'C' in haveP:
##                            # epsr is the cause of the change of C
##                            Cdef = 1 / (self.Zbase * self.V * C0) / PF
##                            quargs.update({
##                                'epsr': self.epsr * quargs.get('C') / Cdef})
##                            
##                        quargs.update({
##                            'V': 1/np.sqrt(quargs.get('epsr')* self.mur)})
##                        
##                    # this is however not sufficient so we need to assume futher
##                    # that Z is related to the Scatter.Zbase : we will assume that
##                    # the geometry has not changed so that the change of V or C is
##                    # solely due to a change of epsr from the default medium
##                    # we had Zbase with the default mur and epsr : so this defines
##                    # lnDs :
##                    
##                    lnDs = self.Zbase/(ZVAC * np.sqrt(self.mur/self.epsr))
##                    quargs.update({'lnDs': lnDs})
##                    
##                    Zr = np.sqrt(quargs.get('mur')/quargs.get('epsr'))
##                    Z = Zr * ZVAC * lnDs
##                    
##                    quargs.update({
##                        'Z': Z,
##                        'L': Z/(C0 * quargs.get('V')) / NH,
##                        'C': 1 / (Z * C0 * quargs.get('V')) / PF})
##                    
##                else: # 'mur' in haveP or 'L' in haveP:
##                    # this is similar as above : the change in L (or mur) is
##                    # due to a change in mur. It is however doubtfull that
##                    # an unchanged epsr is a valid assumption.
##                    pass
##                
##            elif len(haveP) < 3:
##
##                if 'Z' in haveP:
##                    if 'V' in haveP:
##                        pass
##                    elif 'mur' in haveP:
##                        pass
##                    elif 'epsr' in haveP:
##                        pass
##                    elif 'L' in haveP or 'C' in haveP:
##                        pass
##                    else: # 'lnDs' in haveP:
##                        pass
##                    
##                elif 'V' in haveP:
##                    if 'mur' in haveP:
##                        pass
##                    elif 'epsr' in haveP:
##                        pass
##                    elif 'L' in haveP or 'C' in haveP:
##                        pass
##                    else: # 'lnDs' in haveP:
##                        pass
##                    
##                elif 'mur' in haveP:
##                    if 'epsr' in haveP:
##                        pass
##                    elif 'L' in haveP or 'C' in haveP:
##                        pass
##                    else: # 'lnDs' in haveP:
##                        pass
##                    
##                elif 'epsr' in haveP:
##                    if 'L' in haveP or 'C' in haveP:
##                        pass
##                    else: # 'lnDs' in haveP:
##                        pass
##                    
##                else: # 'L' in haveP:
##                    if 'C' in haveP:
##                        pass
##                    else: # if 'lnDs' in haveP
##                        pass
##                # we should have found 'lnDs' with another parameter
##
##            elif len(haveP) < 4:
##                # these could yield a fully determined system except in those
##                # cases below
##                pass
##                
##        #.......................................................................|.....
##
##        # independent relations between parameters
##        E = [# Z = (L NH / (C PF))**0.5
##             # -> ln(Z) - 0.5 ln(L) + 0.5 lnC = 0.5 ln(NH/PF)
##             {'Z': 1, 'L': -0.5, 'C': 0.5, 'K': 0.5*np.log(NH/PF)},             # eq1
##             # Z = (mur / epsr)**0.5 ZVAC ln(OD/ID)
##             # -> ln(Z) - 0.5 ln(mur) + 0.5*ln(epsr) - ln(lnDs) = ln(ZVAC)
##             {'Z': 1, 'mur': -0.5, 'epsr': 0.5, 'lnDs': -1, 'K': np.log(ZVAC)}, # eq2
##             # C0 * V = (mur MU0 * eprs EPS0)**-0.5
##             # -> ln(V) + 0.5 ln(mur) + 0.5 ln(eprs) = 0
##             {'V': 1, 'mur': +0.5, 'epsr': 0.5, 'K': 0.},                       # eq3
##             # C0 V = (L NH C PF)**-0.5
##             # -> ln(V) + 0.5 ln(L) + 0.5 ln(C) = -ln(C0) -0.5 ln(NH PF)
##             {'V': 1, 'L': 0.5, 'C': 0.5, 'K': -np.log(C0)-0.5*np.log(NH*PF)}]  # eq4
##
##        # check for lnDs :
##        try:
##            kwargs.update({'lnDs': np.log(kwargs.get('OD')/kwargs.get('ID'))})
##        except TypeError:
##            pass # no such luck
##
##        sol = dict([(v, None) for v in P if v not in kwargs.keys()])
##
##        
##        rankM = 0
##        itercount = 0
##        defaults = []
##        sol = {}
##
####        lambda have x: x in kwargs.keys()
####        lambda need x: not have(x)
##        
##        while not sol and itercount < 5 :
##            itercount += 1
##            sol, RHS, maxErr, physical = solvit(**kwargs)
##            if maxErr > tol and itercount is 1:
##                raise ValueError("Scatter.trlCoax : values provided do not lead to"
##                                 " possible solution for the TL parameters")
##            if not sol: # rankM < len(needP):
##                needP = set([v for v in P if v not in kwargs.keys()])
##                haveP = set([v for v in P if v in kwargs.keys()])
##                
##                if (need('mur') and not(set(['V','epsr']) < haveP)
##                                and not(set(['Z','V']) < haveP)
##                                and not(set(['L','C']) < haveP)):
##                    kwargs.update({'mur': 1})
##                    defaults.append('mur')
##                    
##                elif ('epsr' in needP
##                      and not('V' in haveP and 'mur' in haveP)
##                      and ('mur' in haveP and kwargs.get('mur') == 1.)):
##                    defaults.append('epsr')
##                    kwargs.update({'epsr': 1.})
##                    
##                elif 'V' in needP and not('mur' in haveP and 'epsr' in haveP):
##                    if (('mur' in haveP and kwargs.get('mur') > 1. and 'Z' not in haveP) or
##                        ('epsr' in haveP and kwargs.get('epsr') > 1. and 'Z' not in haveP)):
##                    # prefer to set default Z first
##                        kwargs.update({'Z':self.Zbase})
##                        defaults.append('Z')
##                    else:
##                        kwargs.update({'V':1})
##                        defaults.append('V')
##                elif 'Z' in needP:
##                    kwargs.update({'Z': self.Zbase})
##                    defaults.append('Z')
##                else:
##                    raise NotImplementedError('Scatter.trlCoax : need more defaults')
##                    
##        # print(sol)
##        
##        if sol :
##            kwargs.update(sol)
##            if maxErr > tol:
##                print(' overconstrained : ', RHS)
##            if not physical :
##                print(' not physical')
##                
##        if kwargs.get('lnDs'):
##            if kwargs.get('OD'):
##                if not kwargs.get('ID'):
##                    kwargs['ID'] = kwargs['OD']/np.exp(kwargs['lnDs'])
##                    
##            elif kwargs.get('ID'):
##                kwargs['OD'] = kwargs['ID']*np.exp(kwargs['lnDs'])
##
##        return kwargs, defaults, itercount
##
    #===========================================================================
    #  s e r i e s _ i m p e d a n c e
    #
    def series_impedance(self,port1,Zs):
        
        N = 2 * self.Zbase + Zs
        s = Zs / N
        t = 2*self.Zbase / N
        self.extendport(port1,np.array([[s,t],[t,s]]))
        
    #===========================================================================
    #  p a r a l l e l _ i m p e d a n c e
    #
    def parallel_impedance(self,port1,Zp):
        N = 2 + self.Zbase / Zp
        s = - self.Zbase/Zp / N
        t = 2 / N
        self.extendport(port1,np.array([[s,t],[t,s]]))
    #===========================================================================
    #  p a r a l l e l _ a d m i t t a n c e
    #
    def parallel_admittance(self,port1,Yp):
        N = 2 + self.Zbase * Yp
        s = - self.Zbase * Yp / N
        t = 2 / N
        self.extendport(port1,np.array([[s,t],[t,s]]))
    
    #===========================================================================
    # e x t e n d p o r t (a)
    #     
    def extendport(self,ports,S2=np.array([[0j,1],[1,0j]])):
        """
        Extend "ports" with a two-port scattering matrix S2 : the second row of S2
        (of the type Matrix) pertains to the new port, the first row is the port that
        connects to port1 (and is eliminated).
        """
        if type(ports) == list:
            #FIXME: why not self += Scatter .... ? because this does not work: self is not updated
            
#             print(self)
#             print(Scatter(fMHz=self.fMHz,Zbase=self.Zbase,iwt=self.iwt,Portnames=ports,S=S2))
            result = self + Scatter(fMHz=self.fMHz,Zbase=self.Zbase,iwt=self.iwt,Portnames=ports,S=S2)
            self.S        = result.S
            self.Portname = result.Portname
            self.Ssigs    = result.Ssigs
            self.Signal   = result.Signal
            
        else:
            if type(ports) == str:
                p1 = self.findport(ports)     # find the index of the port
            else:
                p1 = ports                    # this is an index to the port
        
            qn = 1-(S2[0,0]*self.S[p1,p1])
            if qn != 0:
                q21 = (S2[1,0] / qn) * self.S[[p1],:]           ## this the new row on port p1, we will insert in later into self.S
                q21[0,p1] = S2[1,1] + (S2[0,1]*q21[0,p1])       ##   just setting the future self.S[p1,p1] straight
                q11 = (S2[0,0] / qn) * self.S[[p1],:]           ## We will also need it for the signals
                q12 = (S2[0,1]/qn)*self.S[:,[p1]]               ## This is the new column on port p1 to be inserted later
                self.S = self.S + np.dot(self.S[:,[p1]],q11)    ## This should be the coefs except on row an column p1
                self.S[:,[p1]] = q12                            ## Insert column first because the new row has the correct S[p1,p1]
                self.S[[p1],:] = q21[0,:]                       ## Finally insert the new row
                if len(self.Signal):                          ## take care of the signals now
                    m12 = self.Ssigs[:,[p1]]                  ## column of port p1 : need to insert later
                    self.Ssigs = self.Ssigs + np.dot(m12,q11) ## these are OK now except for column p1
                    self.Ssigs[:,[p1]] = (S2[0,1]/qn)*m12       ## Finally insert the new column
            else:                               ##  S2[1,1] as well as S[p1,p1] have amplitude 1 therefore
                self.S[p1,p1] = S2[1,1]         ##  all other coefficients on the row and column are 0:
                                                ##  S[p1,p1] was an isolated port and is is replaced
                                                ##  by another isolated port S2[1,1]            
                
    #===========================================================================
    # j o i n N p o r t s
    #      
    def joinNports(self,*ports,newname=None,SJ=None):
        """this code's purpose is to test other code
        """
        
        if newname is None:
            # in the old implemetation 'newname=' was not really required as there
            # were 2 and only 2 ports to be connected into a 3rd one
            
            # we check here if this could be the case by assuming that if the
            # the last port of *ports is not known it is in fact the new name
            
            try:
                p = self.findport(ports[-1])
            except:
                newname = ports[-1]
                ports = ports[:-1]
                
        
        if newname is None:
            self.connectports(*ports, Jc=SJ)
        
        M = len(ports) + 1
        
        if SJ is None:
            # if SJ is not given use ideal junction
            SJ = np.array([[2/M - (1 if kc is kr else 0) for kc in range(M)]
                              for kr in range(M)])
        else:
            # check that S fits the number of ports given
            if np.array(SJ).shape != (M,M):
                raise ValueError('Scatter.joinNports: supplied array SJ does'
                                 ' not fit number of supplied ports')
        if newname is None:
            # there was really no newname given, now we need to reduce the
            # supplied/assumed SJ by one port terminated on open before passing
            # on to connectports.
            
            Jc = SJ[:-1,:-1] + np.dot(SJ[:-1,-1], SJ[-1,:-1])/(1 - SJ[-1,-1])
            self.connectports(*ports, Jc=Jc)
                
        portlist = [p if isinstance(p,str) else self.Portname[p] for p in ports
                   ] + [newname]
                                   
        self += self.clone_except(S=SJ, Portname=portlist, Signal=[], Ssigs=[])

        
    #===========================================================================
    # j o i n 2 p o r t s
    #      
    def join2ports(self,port1,port2,newname=None,SJ=np.array([[-1,+2,+2],
                                                              [+2,-1,+2],
                                                              [+2,+2,-1]])/3.):
        """
        combine the ports m and n of scattering object, append the new port as last
        
        The new port's name will be 'newname' and the result returned has one port less,
        however if newname is empty or not supplied the resulting port is terminated by
        an open circuit and the returned result will have two ports less.

        ::
        
              +----------+
              |Z         |
              |       1 -+- <-- i1
              |          |   v1
             ...        ...
              |          |
              |       m -+- <-- im -+
              |          |    vm    +-- <-- ip
              |       n -+- <-- in -+     vp
              |          |    vn
             ...        ...                 |
              |          |                  |
              | # ports -+- <-- i#          | append port p as last port of Zt
              |          |   v#             | if 'newname' is not empty
             ...        ...                 |
              |          |        <---------+
              +----------+
              
        this code is kept for documentation purposes (it was the old implementation
        of joinports that would combine only 2 ports)
        """

        ## extract sub matrices :
        J11 = SJ[0:2,0:2]
        J1c = SJ[2:,0:2]
        Jc1 = SJ[0:2,2].reshape((2,1))
        jcc = SJ[2,2]

        ## replace ports passed as named by their index
        if type(port1) == str:
            p1 = self.findport(port1)
        else:
            p1 = port1
            
        if type(port2) == str:
            p2 = self.findport(port2)
        else:
            p2 = port2
                    
        ## swap ports intelligently
        nP1 = len(self.Portname)-1 ## index of last port
        nP2 = nP1-1                ## index of last but one port
        if p2 != nP2: ## check for correct swapping
            self.swapports(p1,nP2)
            self.swapports(p2,nP1)
        else:
            if p1 != nP1:
                self.swapports(p2,nP1)
                self.swapports(p1,nP2)
            else:
                self.swapports(p1,p2)
        
        ## perform the elimination
        if len(self.Portname) > 2:  ## more than 2 ports -> general case
            S11 = self.S[:nP2,:nP2]
            S2n = self.S[nP2:,:nP2]
            Sn2 = self.S[:nP2,nP2:]
            Snn = self.S[nP2:,nP2:]
            IJS = inv(np.eye(2) - np.dot(J11,Snn))
            ISJ = inv(np.eye(2) - np.dot(Snn,J11))
            Sn2IJS = np.dot(Sn2,IJS)
            J1cISJ = np.dot(J1c,ISJ)
            r1 = np.hstack(( S11+np.dot(np.dot(Sn2IJS,J11),S2n), 
                             np.dot(Sn2IJS,Jc1) ))
            r2 = np.hstack(( np.dot(J1cISJ,S2n), 
                             jcc+np.dot(np.dot(J1cISJ,Snn),Jc1) ))
            self.S = np.vstack((r1,r2 ))
            if len(self.Ssigs):
                M11 = self.Ssigs[:,:nP2]
                Mm2IJS = np.dot(self.Ssigs[:,nP2:],IJS)
                self.Ssigs = np.hstack((M11+np.dot(np.dot(Mm2IJS,J11),S2n),
                                        np.dot(Mm2IJS,Jc1)))
            
        else:  ## circuit has only 2 ports which are combined to a single port
            if len(self.Ssigs):
                self.Ssigs = np.dot(self.Ssigs,
                                    inv(np.eye(2)-np.dot(np.dot(J11,self.S),Jc1)))
                
            self.S = jcc + np.dot(np.dot(np.dot(J1c,
                                                inv(np.eye(2)-np.dot(self.S,J11))),
                                  self.S),Jc1)

        if not newname:     # a new name was not supplied the port is terminated 
                            # by an open circuit
            self.termport(nP2)
        else:
            self.Portname[nP2] = newname ## set the port's name
        
        ## do houskeeping on the Portname

        self.Portname = self.Portname[:len(self.S)]

    #===========================================================================
    # j o i n p o r t s
    #      
    def joinports(self,*ports,newname=None,SJ=None):
        """
        connect the M ports scattering object together into a new port and 
        append the new port as last
        
        The new port's name will be 'newname' and the result returned has M-1 
        ports less, however if newname is empty or not supplied the resulting 
        port is terminated by an open circuit and the returned result will have
        M ports less.

        ::
        
              +----------+
              |Scatter   |
              |       1 -+- <-- i1
              |          |   v1
             ...        ...
              |          |
              |       m -+- <-- im -+
              |          |    vm    |
             ...        ...       --+-- <-- ip, vp
              |          |          |
              |       n -+- <-- in -+       |
              |          |    vn            |
             ...        ...                 |
              |          |                  |
              | # ports -+- <-- i#          | append port p as last port of the
              |          |   v#             | Scatter object if 'newname' is not
             ...        ...                 | empty
              |          |        <---------+
              +----------+
        """

        if newname is None:
            # in the old implemetation 'newname=' was not really required as there
            # were 2 and only 2 ports to be connected into a 3rd one
            
            # we check here if this could be the case by assuming that if the
            # the last port of *ports is not known it is in fact the 'newname'
            
            try:
                self.findport(ports[-1])
            except:
                newname = ports[-1]
                ports = ports[:-1]
                
        M = len(ports)

        if SJ is None:
            # if SJ is not given use ideal junction
            SJ = np.array([[2/(M+1) - (1 if kc is kr else 0) for kc in range(M+1)]
                              for kr in range(M+1)])
        else:
            # check that S fits the number of ports given
            if np.array(SJ).shape != (M+1,M+1):
                raise ValueError('Scatter.joinMports: supplied array SJ does'
                                 ' not fit number of supplied ports')
                
        if newname is None:
            # there was really no newname given, now we need to reduce the
            # supplied/assumed SJ by one port terminated on open before passing
            # on to connectports.
            
            Jc = SJ[:-1,:-1] + np.dot(SJ[:-1,-1], SJ[-1,:-1])/(1 - SJ[-1,-1])
            
            self.connectports(*ports, Jc=Jc)
            return

        ## extract sub matrices :
        J11 = SJ[0:M,0:M]
        J1c = SJ[M:,0:M]
        Jc1 = SJ[0:M,M].reshape((M,1))
        jcc = SJ[M,M]

        portlist = [p if isinstance(p,str) else self.Portname[p] for p in ports]

        nP = len(self.Portname)
        nPm = nP-1-(M-1)
        for k,p in enumerate(reversed(portlist)):
            self.swapports(nP-1-k,p)

        
        ## perform the elimination
        if len(self.Portname) > M:  ## more than 2 ports -> general case
            S11 = self.S[:nPm,:nPm]
            S2n = self.S[nPm:,:nPm]
            Sn2 = self.S[:nPm,nPm:]
            Snn = self.S[nPm:,nPm:]
            IJS = inv(np.eye(M) - np.dot(J11,Snn))
            ISJ = inv(np.eye(M) - np.dot(Snn,J11))
            Sn2IJS = np.dot(Sn2,IJS)
            J1cISJ = np.dot(J1c,ISJ)
            r1 = np.hstack(( S11+np.dot(np.dot(Sn2IJS,J11),S2n), 
                             np.dot(Sn2IJS,Jc1) ))
            r2 = np.hstack(( np.dot(J1cISJ,S2n), 
                             jcc+np.dot(np.dot(J1cISJ,Snn),Jc1) ))
            self.S = np.vstack((r1,r2 ))
            if len(self.Ssigs):
                M11 = self.Ssigs[:,:nPm]
                Mm2IJS = np.dot(self.Ssigs[:,nPm:],IJS)
                self.Ssigs = np.hstack((M11+np.dot(np.dot(Mm2IJS,J11),S2n),
                                        np.dot(Mm2IJS,Jc1)))
            
        else:  ## circuit has only M ports which are combined to a single port
            if len(self.Ssigs):
                self.Ssigs = np.dot(self.Ssigs,
                                    inv(np.eye(M)-np.dot(np.dot(J11,self.S),Jc1)))
            self.S = jcc + np.dot(np.dot(np.dot(J1c,
                                                inv(np.eye(M)-np.dot(self.S,J11))),
                                  self.S),Jc1)

        self.Portname[nPm] = newname ## set the port's name
        
        ## do houskeeping on the Portname

        self.Portname = self.Portname[:len(self.S)]

    #===========================================================================
    # c o n n e c t 2 p o r t s
    #
    def connect2ports(self,port1,port2,Jc=np.array([[0,1],[1,0]])):
        """
        connect2ports : connects ports1 and ports2 through connection described 
            by SC could be bone by joinports(port1,port2,SJ) i.e. no newname and
            a standard junction of which one port has been extended by SC but a 
            more efficient code can be inserted here.
            
        old depreciated code kept for reference
        """
        ## replace ports passed as named by their index
        if type(port1) == str:
            p1 = self.findport(port1)
        else:
            p1 = port1
            
        if type(port2) == str:
            p2 = self.findport(port2)
        else:
            p2 = port2

        ## swap ports intelligently
        nP1 = len(self.Portname)-1 ## index of last port
        nP2 = nP1-1                ## index of last but one port
        if p2 != nP2: ## check for correct swapping
            self.swapports(p1,nP2)
            self.swapports(p2,nP1)
        else:
            if p1 != nP1:
                self.swapports(p2,nP1)
                self.swapports(p1,nP2)
            else:
                self.swapports(p1,p2)
        
        JISnnJ = np.dot(
                   np.dot(Jc,
                          inv(np.eye(2)-np.dot(self.S[nP2:,nP2:],Jc))),
                   self.S[nP2:,:nP2])

        self.S = self.S[:nP2,:nP2] + np.dot(self.S[:nP2,nP2:],
                                            JISnnJ)
        self.Portname = self.Portname[:-2]
        if len(self.Ssigs):
            self.Ssigs = self.Ssigs[:,:nP2] + np.dot(self.Ssigs[:,nP2:],
                                                     JISnnJ)

    #===========================================================================
    # c o n n e c t p o r t s
    #
    def connectports(self,*ports,Jc=None):
        """
        connectports : connects ports1 and ports2 through connection described 
            by Jc could be done by joinports(port1,port2,SJ) i.e. no newname and
            a standard junction of which one port has been extended by Jc but a 
            more efficient code can be inserted here.
        """
                
        M = len(ports)
        
        if Jc is None:
            # if Jc is not given use ideal junction
            Jc = np.array([[2/M - (1 if kc is kr else 0) for kc in range(M)]
                              for kr in range(M)])
        else:
            # check that S fits the number of ports given
            if np.array(Jc).shape != (M,M):
                raise ValueError('Scatter.connectports: supplied array Jc does'
                                 ' not fit number of supplied ports')

        portlist = [p if isinstance(p,str) else self.Portname[p] for p in ports]
        
        nP = len(self.Portname)
        nPm = nP-1-(M-1)
        for k,p in enumerate(reversed(portlist)):
            self.swapports(nP-1-k,p)
                    
        JISnnJ = np.dot(
                   np.dot(Jc,
                          inv(np.eye(M)-np.dot(self.S[nPm:,nPm:],Jc))),
                   self.S[nPm:,:nPm])

        self.S = self.S[:nPm,:nPm] + np.dot(self.S[:nPm,nPm:],
                                            JISnnJ)
        self.Portname = self.Portname[:-M]
        if len(self.Ssigs):
#             print(self.Ssigs.shape)
#             print(self.Ssigs[:,:nPm].shape)
#             print(self.Ssigs[:,nPm:nP].shape) #FIXME: we should not have had to use nP
#             print(JISnnJ.shape)
            self.Ssigs = self.Ssigs[:,:nPm] + np.dot(self.Ssigs[:,nPm:],
                                                     JISnnJ)

    #===========================================================================
    # OptimizeExcitation
    #
    def OptimizeExcitation(self,signalnumbers,target):
        """
        excitation = OptimizeExcitation(SZ,signalnumbers,target)

        finds the excitation to be applied at the ports in order to best
        fit (LSF) the target for the signalnumbers.

        Input :

        signalnumbers : signals entries in SZ.Sigs[2*#sigs,#ports ]to fit to the
                        target (python odd numbers will be currents and even 
                        numbers will be voltages)
                        
        target        : target values for the signals arranged as a column matrix
                        matrix([[v1],[v2],[v3], ... ,[vN]])

        Output :

        excitation : the excitation dictionary to be used with 
                     SZ.calcsigs(signalsrequested,excitation)

        Note : len(signalnumbers) must be equal or larger than len(SZ.Portnames)
               for the result to have useful meaning.
        """

        CI = np.array([[self.Ssigs[signalnumbers[ks],m] 
                        for m in range(len(self.Portname)) ] 
                       for ks in range(len(signalnumbers))])
        Ib = np.dot(CI.T.conj(), target)
        M  = np.dot(CI.T.conj(), CI)
        Ap = np.dot(np.linalg.inv(M), Ib)   # this is the optimal excitation of
                                            # the ports to obtain the desired signals
        excitation = {}

        for kp, pname in enumerate(self.Portname):
            excitation[pname] = Ap[kp] # Ap is an np.matrix hence the 2nd index

        return excitation

    #===========================================================================
    # G e t _ Z ( ) ,   G e t _ Y ( ) ,  G e t _ S ( z r e f = s e l f .Z b a s e)
    #
    def Get(self, mtype, **kwargs):
        """
        return the mtype-matrix representation of SZ 
            (reference impedance Z0 if required)
            
           mytpe = 'Z', 'Y' or 'S'
        """

        if mtype == 'Z' :
            return self.Get_Z(**kwargs)
        elif mtype == 'Y' :
            return self.Get_Y(**kwargs)
        elif mtype == 'S' :
            return self.Get_S(**kwargs)
        else :
            raise ScatterError("Scatter.Get() : unknown mtype '%s'" % str(mtype))
             
    #===========================================================================
    
    def Get_Z(self,**kwargs):
        """
        return the Z-matrix representation of SZ
        """
        return np.dot(np.linalg.inv(np.eye(len(self.S))-np.array(self.S)),
                    (np.eye(len(self.S))+np.array(self.S)))*self.Zbase
    
    #===========================================================================
    
    def Get_Y(self,**kwargs):
        """
        return numpy.array of the Y-matrix representation of SZ
        """
        return np.dot(np.linalg.inv(np.eye(len(self.S))+np.array(self.S)),
                      (np.eye(len(self.S))-np.array(self.S)))/self.Zbase
    
    #===========================================================================

    def Get_S(self,**kwargs):
        """
        Get_S(Z0=self.Zbase) : return numpy.array of the S-matrix representation
                                of SZ on reference impedance Z0
        """
        Z0 = kwargs.get('Z0',self.Zbase)
        if Z0 == self.Zbase :
            return self.S.copy() # otherwise we can have strange side effects ?
        else :
            # 2010-09-21 FDe : change of base not involving Z 
            # (which may not always exists)
            Id = np.eye(len(self.S))
            p = (self.Zbase/Z0 + 1.)/2.
            m = p - 1.
            return np.dot(np.linalg.inv(p * Id + m * self.S), (m * Id + p * self.S))    

    #===========================================================================

    def Get_S1(self,**kwargs):
        """
        Get_S1(Z0=self.Zbase) : return numpy.array of the S-matrix representation 
                                of SZ on reference impedance Z0
                               
                               (this was to test another derivation which lead to
                               formulas that appeared slightly different)
        """
        Z0 = kwargs.get('Z0',self.Zbase)
        if Z0 == self.Zbase :
            return self.S
        else :
            # 2010-09-21 FDe : change of base not involving Z 
            # (which may not always exists)
            Id = np.eye(len(self.S))
            p = (Z0/self.Zbase + 1.)/2.
            m = 1. - p
            return np.dot((m*Id + p*self.S),np.linalg.inv(p*Id + m*self.S))    
        
    #===========================================================================
    #
    # P l o s s 
    #
    def Ploss(self, As):
        return Ploss(self.S, self.Zbase, As) 

#===============================================================================
#
#  S o m e   u s e f u l   r o u t i n e s
#

#===============================================================================
#
#  P l o s s 
#
def Ploss(S, Zbase, As):
    power = lambda Q: np.real(np.dot(np.conj(Q.T), Q))[0,0] / (2*Zbase)                     
    return power(As) - power(np.dot(S, As)) 

#===============================================================================
#
#  c o n v e r t _ g e n e r a l
#
def convert_general(Z2, S1, Z1, type1='std', type2='std'):
    
    try:
        N = S1.shape[0]
    except AttributeError:
        try:
            N = len(S1)
        except:
            N = 1
            
    U = np.eye(N)+0j
    O = np.ones(U.shape[:1])

    def coeffs(Z, typ):
        zz = np.array(Z)
        if len(zz.shape) == 0:
            Zd = Z * O              # Z was just a numeral
        elif len(zz.shape) == 1:
            Zd = zz                 # Z was a vector
        else:
            raise ValueError('Port impedance has to be a scalar or 1D vector')
        
        iZd = 1/Zd
        iRd = 1/np.sqrt(np.real(Zd))
        
        if typ == 'std':
            QaV, QaI = O/2,  Zd/2
            QbV, QbI = O/2, -Zd/2
            QVa, QVb = O, O
            QIa, QIb = iZd, -iZd
            
        elif typ == 'gsm':
            QaV, QaI = iRd/2,  Zd * iRd / 2
            QbV, QbI = iRd/2, -np.conj(Zd) * iRd / 2
            QVa, QVb = np.conj(Zd) * iRd, Zd * iRd
            QIa, QIb = iRd, -iRd
        
        elif typ == 'hfss':
            iRd = 1/np.sqrt(Zd)
            QaV, QaI = iRd/2,  Zd * iRd / 2
            QbV, QbI = iRd/2, -Zd * iRd / 2
            QVa, QVb = Zd * iRd, Zd * iRd
            QIa, QIb = iRd, -iRd
        
        else:
            raise ValueError('type must be "std" or "gsm" got %r' % typ)
        
        if False:
            print('QaV',QaV)
            print('QbV',QbV)
            print('QaI',QaI)
            print('QbI',QbI)
            
            print('QVa',QVa)
            print('QVb',QVb)
            print('QIa',QIa)
            print('QIb',QIb)
            
            print()
            
            QQ = np.vstack((np.hstack((np.diag(QaV), np.diag(QaI))),
                            np.hstack((np.diag(QbV), np.diag(QbI)))))
            iQQ = np.linalg.inv(QQ)
            jQQ = np.vstack((np.hstack((np.diag(QVa), np.diag(QVb))),
                             np.hstack((np.diag(QIa), np.diag(QIb)))))
            D = iQQ - jQQ
            dmax = np.max(np.max(np.abs(D)))
            if dmax > 1E-3:
                print('dmax =', dmax)
                printMA(QQ,pfmt='%5.3f<%5.1f')
                printMA(iQQ,pfmt='%5.3f<%5.1f')
                printMA(jQQ,pfmt='%5.3f<%5.1f')
                print()
         
        return QaV, QaI, QbV, QbI, QVa, QVb, QIa, QIb
    
    SaV, SaI, SbV, SbI, SVa, SVb, SIa, SIb = coeffs(Z1, type1)  # @UnusedVariable
    TaV, TaI, TbV, TbI, TVa, TVb, TIa, TIb = coeffs(Z2, type2)  # @UnusedVariable
    
    
    V, I = np.diag(SVa) + np.diag(SVb) @ S1, np.diag(SIa) + np.diag(SIb) @ S1
    A2, B2 = np.diag(TaV) @ V + np.diag(TaI) @ I, np.diag(TbV) @ V + np.diag(TbI) @ I
    S2 = B2 @ np.linalg.inv(A2)
    
    if False:
        def issym(M, mesg):
            m = np.max(np.max(np.abs(M - M.transpose())))
            if m > 1E-3:
                print(mesg, m)
            
        issym(S1,'S1')
        issym(V, 'V')
        issym(I, 'I')
        issym(A2, 'A2')
        issym(B2, 'B2')
        issym(S2,'S2')
         
    return S2

#===============================================================================
#
#  c o n v e r t _ s t d 2 g s m
#
def convert_std2gsm(Zgen2, Sgen1, Zgen1):
    """
    convert the standard voltage wave scattering matrix Sgen1 with real port 
    impedances (the same for all ports) Zgen1 to one with (complex) port impedances
    (possibly different for each port) Zgen2
    """
    
    return convert_general(Zgen2, Sgen1, Zgen1, 'std', 'gsm')
 
#===============================================================================
#
#  c o n v e r t _ g s m 2 s t d
#
def convert_gsm2std(Zgen2, Sgen1, Zgen1):
    """
    convert the general power wave scattering matrix Sgen1 with (complex) port 
    impedances Zgen1 to one with (real) port impedance the same for each port Zgen2
    """
    
    return convert_general(Zgen2, Sgen1, Zgen1, 'gsm', 'std')
 
#===============================================================================
#
#  c o n v e r t _ g e n e r a l _ o l d
#
def convert_general_old(Zgen2, Sgen1, Zgen1):
    """
    convert the general power wave scattering matrix Sgen1 with (complex) port 
    impedances Zgen1 to one with (complex) port impedances Zgen2
    """

    def RZ(Zgen):
        
        try:
            if len(Zgen) is len(Sgen1):
                Zg = Zgen
            
            elif len(Zgen) is 1:
                # we are still here 
                Zg = [Zgen[0]] * len(Sgen1)
 
            else:
                # definitely a mismatch of array lengths
                raise ValueError('convert+general: dimension mismatch')
            
        except TypeError:
            # we just got a scalar
            Zg = [Zgen] * len(Sgen1)
            
        sqrtReZs = [np.sqrt(np.real(z)) for z in Zg]
        R = np.diag(sqrtReZs)                                         # R is reciprocal
        invR = np.diag( [1/r for r in sqrtReZs] )                     # invR is reciprocal
        Z = np.diag( [g / np.conj(z) for g, z in zip(sqrtReZs, Zg)] ) # Z is reciprocal
        
        return R, Z, invR
            
    Id = np.eye(len(Sgen1))
    
    R1, Z1, invR1 = RZ(Zgen1)
    R2, Z2, invR2 = RZ(Zgen2)
    
    Q21 = invR2.dot(R1).dot(Id+Sgen1).dot(inv(Id-Sgen1)).dot(inv(Z1)).dot(Z2) # Q21 is reciprocal if Sgen1 is
    Sgen2 = (Q21 - Id).dot(inv(Q21 + Id))                                     # Sgen is reciprocal if Sgen1 is
    
    return Sgen2
 
#===============================================================================
#
#  T L _ G R L C
#
def TL_GRLC(gamma, Zc, **kwargs):
    """
    try and find the TL G, R, L, C from gamma and Zc
    
    gamma**2 = (R +jwL)*(G + jwC) = -w**2 L C + R G + jwL G + jwC R
    Zc**2 = (R + jwL)**2 / gamma**2/(G+jwC)**2 = (R + jwL) / (G + jwC)
    
    iwt is found from the sign of imag(gamma)
    
    kwargs:
        fMHz : frequency [MHz]
        verbose : print minimization information
    
    returns
        G [Mho/m]
        R [Ohm/m]
        L [nH/m] if fMHz is given else wL [Ohm/m]
        C [pF/m] if fMHz is given else wC [Mho/m]
        
    """
    
    fMHz = kwargs.pop('fMHz', None)
    verbose = kwargs.pop('verbose', False)
    
    if verbose:
        print('gamma =', gamma, ', Zc =', Zc)

    iwt = -1 if gamma.imag < 0 else +1
    
    def fm(X):
        G, R, wL, wC = tuple(X)
        gm = np.sqrt((np.abs(R)+iwt*1j*np.abs(wL))*(np.abs(G)+iwt*1j*np.abs(wC)))
        Zm = (np.abs(R)+iwt*1j*np.abs(wL)) / gm 
        err = np.abs(gm/gamma -1)**2 + np.abs(Zm/Zc -1)**2
        if verbose:
            print('G = %f Mho/m' % G, ', R = %f Ohm/m' % R,
                   ', wL = %f Ohm/m' % wL, ', wC = %f Mho/m' % wC, 
                   ', gm = %f%+fj' % (gm.real, gm.imag),
                   ', Zm = %f%+fj' % (Zm.real, Zm.imag),
                   ', fun = %g' % err )
        return err
    
            
    wL, wC = Zc.real * gamma.imag, gamma.imag / Zc.real

    X = fmin(func=fm, x0=[0., 0., wL, wC], disp=False)
    
    if verbose:
        print(X)
        
    G, R, wL, wC = tuple(X)
    
    if fMHz :
        w = 2E6*np.pi*fMHz
        return np.abs(G), np.abs(R), np.abs(wL)/w*1E9, np.abs(wC)/w*1E12
    
    else:
        return np.abs(G), np.abs(R), np.abs(wL), np.abs(wC)
        
    
#===============================================================================
#
# C o a x A t t e n u a t i o n
#
def CoaxAttenuation(fMHz, Z, OD, rho_inner=2E-8, rho_outer=2E-8, V=1.):
    """
    returns the attenuation [m**-1] for a coaxial TL with outer diameter OD [m],
    characteristic impedance Z [Ohm], and specific resitances for inner conductor
    and outer conductor, resp. rho_inner, rho_outer [Ohm.m] at frequency fMHz [MHz]
    
    This only takes into account resistive losses of the conductors and was an
    approximation : alpha = rTL/(2*Z)
    """
    w = 2e6 * np.pi * fMHz
    epsilon = 1 / ((V * speed_of_light)**2 * mu_0)
    Zvac = np.sqrt(mu_0/epsilon)/(2*np.pi)
    wL = w * Z * np.sqrt(mu_0 * epsilon)
    RsO = np.sqrt(w * mu_0 * rho_outer/2)
    RsI = np.sqrt(w * mu_0 * rho_inner/2)
    ID = OD * np.exp(-Z/Zvac)
    rh = (RsO/OD + RsI/ID)/np.pi/wL
    alpha = w * np.sqrt(epsilon*mu_0*(np.sqrt(1 + rh**2)-1)/2)
    
#     print('epsilon_r:', epsilon/epsilon_0)
#     print('Zvac:', Zvac)
#     print('OD, ID:', OD, ID)
#     print('RsO, RsI:',RsO,RsI)
#     print('rh:',rh)
#     print('wL:',wL)
#     print('alpha:',alpha)

    return alpha

# approximation
#
#     return (  (np.sqrt(rho_outer) + np.exp(Z/Zvac)*np.sqrt(rho_inner)) *
#                np.sqrt(0.1*fMHz) / (OD * Z)  )

#===============================================================================
#
# H y b r i d C o u p l e r 4 P o r t
#
def HybridCoupler4Port(fMHz, Zbase=30., kdB=None, lc=None, lp=0.0, iwt = 1):
    """
    portnames are: 'A1', 'A2' (for strip 'A') and          A1             A2
                   'B1', 'B2' (for strip 'B'); ports        |             |
                   with equal numerals face each other.     +-------------+
                                                            +-------------+
                                                            |             |
                                                           B1             B2

    return the Scatter object of a hybrid coupler defined by :
    
      frequency           fMHz   [MHz]
      ref. impedance      Zbase  [Ohm] (defaults to 30. Ohm)
      coupler length      lc     [m]   (defaults to quarter wave)
      coupling constant   kdB    [dB]  (defaults to 20.*np.log10(2.) = 3. db)
      port connections    lp     [m]   char. imp. Zbase (defaults to 0.)
      time dependence     iwt          defaults to +1                             
    """
    
    blc = 2E6*np.pi*fMHz/speed_of_light*(lc if lc else speed_of_light/(4E6*fMHz))
    cosblc, sinblc = np.cos(blc), np.sin(blc)
    
    k = (10 ** (-(kdB /20.))) if kdB else np.sqrt(0.5)
    s1k2 = np.sqrt(1 - k**2)
    
    N = s1k2 * cosblc + iwt*1j*sinblc
    A, B = s1k2 / N, iwt * 1j * k * sinblc / N
    SZ = Scatter(fMHz = fMHz,
                 Zbase=Zbase,
                 Portnames=['A1','A2','B1','B2'],
                 S=[[0., A, B, 0.],
                    [A, 0., 0., B],
                    [B, 0., 0., A],
                    [0., B, A, 0.]],
                 iwt=iwt)
    if lp > 0. :
        for p in SZ.Portname:
            SZ.trlAZV(p, lp)
            
    return SZ

#===============================================================================
#
# V I _ f r o m _ S 
#
def VI_from_S(S, Z0=30.):
    """
    returns the VI transformation matrix M from S on reference imedance Z0 :
                  +-------------------+
                  |                   |
            I2    |  [V2]       [V1]  |         I1
        ----->----+  [  ] = M . [  ]  +--------->-----
             ^    |  [I2]       [I1]  |         ^
         V2  |    |                   |         | V1
             |    +-------------------+         |
             
    note S is defined for an oposite sign of I1 shown above (i.e. standard)
    """
    if not isinstance(S, np.ndarray):
        S = np.array(S)
        
    if S.shape != (2,2):
        raise ValueError('VI_from_S is only possible for 2x2 S matrices')
    
    M2 = np.array([[  -S[0,1]  ,   -S[0,1]*Z0   ],
                   [ 1-S[1,1]  , -(1+S[1,1])*Z0 ]])
    
    M1 = np.array([[ S[0,0]-1  , -(1+S[0,0])*Z0 ],
                   [  S[1,0]   , -S[1,0]*Z0     ]])
    
    M = np.dot(np.linalg.inv(M2),M1)
    
    return M
    
##===============================================================================
#
# S _ f r o m _ V I 
#
def S_from_VI(M, Z0=30.):
    """
    returns the S on reference imedance Z0 from the VI transformation matrix M :
                  +-------------------+
                  |                   |
            I2    |  [V2]       [V1]  |         I1
        ----->----+  [  ] = M . [  ]  +--------->-----
             ^    |  [I2]       [I1]  |         ^
         V2  |    |                   |         | V1
             |    +-------------------+         |
             
    note S is defined for an oposite sign of I1 shown above (i.e. standard)
    """
    if not isinstance(M, np.ndarray):
        M = np.array(M)
        
    if M.shape != (2,2):
        raise ValueError('S_from_VI is only possible for 2x2 M matrices')
    
    S2 = np.array([[      1            ,     + Z0         ],
                   [ M[0,0]-Z0*M[1,0]  , M[0,1]-Z0*M[1,1] ]])
    
    S1 = np.array([[      1            ,     - Z0         ],
                   [ M[0,0]+Z0*M[1,0]  , M[0,1]+Z0*M[1,1] ]])
    
    S = np.dot(S2,np.linalg.inv(S1))
    
    return S
    
#===============================================================================
#
# Y _ f r o m _ S 
#
def Y_from_S(S,Zbase=30):
    """
    returns the admittance matrix Y from a scattering matrix S 
    defined on Zbase [Ohm]
    """
    return np.dot(inv(np.eye(len(S))+np.array(S)),
                  (np.eye(len(S))-np.array(S)))/Zbase

#===============================================================================
#
# Z _ f r o m _ S 
#
def Z_from_S(S,Zbase=30):
    """
    returns the impedance matrix Z from a scattering matrix S 
    defined on Zbase [Ohm]
    """
    return Zbase*np.dot(inv(np.eye(len(S))-np.array(S)),
                        (np.eye(len(S))+np.array(S)))

#===============================================================================
#
# S _ f r o m _ Z 
#
def S_from_Z(Z,Zbase=30):
    """
    returns the scattering matrix S defined on Zbase [Ohm] from an 
    impedance matrix Z
    """
    return np.dot((np.array(Z) - Zbase*np.eye(len(Z))),
                  inv(np.array(Z) + Zbase*np.eye(len(Z))))

#===============================================================================
#
# S _ f r o m _ Y 
#
def S_from_Y(Y,Zbase=30):
    """
    returns the scattering matrix S defined on Zbase [Ohm] from an 
    impedance matrix Z
    """
    return np.dot((np.eye(len(Y)) - Zbase*np.array(Y)),
                  inv(np.eye(len(Y)) + Zbase*np.array(Y)))

#===============================================================================
# read Riccardo's impedance matrices for the ILA
#
# this has moved into an ILA specific routine : ILAZ88_Class3_v2

##def readZtopica(fMHz,dplasma_cm, \
##                dirpath='/home/frederic/Documents/Programming/Python/Scattertoolbox/matrices', \
##                tilted=1):
##    """
##    reads Riccardo's impedance matrices files produces by Topica and interpolates between
##    files to match the requested dplasma_cm.
##    """
##    
##    def read1file(fname):
##        """
##        reads 1 of Riccardo's impedance matrices
##        """
##    # print 'readZtopica : ', fname
##        tfile = open(fname,'r')
##        tfile.readline() # skip this line
##        tfile.readline() # and that one
##        tfile.readline() # and this one also
##        Z = mat(zeros((8,8)),dtype=complex)
##        for aline in tfile.readlines():
##            t = aline.split()
##            Z[int(t[0])-1,int(t[1])-1] = float(t[2])+1j*float(t[3])
##        return Z
##    
##    # some tests ...
##    assert round(fMHz*100)/100. in [28.50, 33.00,42.00,47.00], (
##               "readZtopica","there is no Topica data for "+str(fMHz)+" MHz.")
##
##    if tilted:
##        fnames = [f for f in glob.glob(os.path.join(
##                                         dirpath,'Z_tilted_*cm_*MHz.txt'))
##                 ]
##    # print 'readZtopica : ', fnames
##    
##    else:
##        raise Exception("readZtopica","not tilted data not considered yet")
##    
##    tlst = []
##    for f in fnames:
##
##        # this will extract the two numbers (plasma distance, frequency) from
##        # the filename
##        # re.findall will make a list of all occurences of the regex it has found
##        # (I think that) because the regex has 2 groupings, an outer one and 
##        # an inner one, this list is a list of tuples for each grouping. 
##        # We are only interested in the  first of each tuple.
##
##        d, fq = tuple(map((lambda x : float(x[0]) ),
##                          re.findall(r'([+-]{0,1}\d+(\.\d*){0,1})',
##                                     os.path.basename(f))))
##        if abs(fq - fMHz) < 0.001:
##            tlst.append((d,f))
##    tlst.sort()
##    dmin = tlst[0][0]
##    dmax = tlst[-1][0]
##    assert (dplasma_cm >= dmin) and (dplasma_cm <= dmax), (
##        "readZtopica","dplamsa_cm "+str(dplasma_cm)+
##        " cm is outside of range ["+str(dmin)+
##        ","+str(dmax)+"].")
##
##    # we now have a sorted list of distances with filenames, lets find between
##    # which we need to interpolate
##    
##    idxs = map((lambda x: x[0] > dplasma_cm),tlst)   # build a list of comparisons
##    idxs.append(True)                      # make sure that there is always 1 True
##    idx = idxs.index(True)                 # find which one it was
##    if idx is len(tlst):
##        # just need to read 1 file
##        Z = read1file(tlst[-1][1])
##    else:
##        Z = ( (tlst[ idx ][0] - dplasma_cm) * read1file(tlst[idx-1][1]) + \
##              (dplasma_cm - tlst[idx-1][0]) * read1file(tlst[idx][1]) ) / \
##              (tlst[ idx ][0] - tlst[idx-1][0])
##
##    # OK we have the impedance matrix : now shift it back to the probe locations
##
##    Zfdrs = 20.0
##    lfdrs = 0.0995 - array([0.155,0.155,0.185,0.255,0.155,0.155,0.185,0.255],
##                           dtype=float)
##    s = Scatter(fMHz, Zbase=Zfdrs, Z = Z)
##    for k in range(8):
##        s.trlAZV([s.Portname[k],"b"+str(k+1)],lfdrs[k])
##
##    if 1 is 0:
##        print "Riccardo's before corrections ..."
##        for k in range(8):
##            for l in range(8):
##                print "%7.2f%+7.2fj " % (real(Z[k,l]), imag(Z[k,l])),
##            print
##        print
##        print "Riccardo's as a Scatter after corrections ..."
##        print s
##        
##    return Z_from_S(s.S,s.Zbase)
    
## =============================================================================
##
## Utilities for TL caluclations
##

#===============================================================================
#
# T L _ V m a x
#

def TL_Vmax(Lmax, Z0, Lwave, P= None, V= None, I= None, Z= None, dx= None):
    """
    returns the maximum voltage and the distance along the TL of characteristic
    impedance Z0 where this voltage maximum occurs. If it is past the maximum
    length Lmax, then the actual voltage maximum obtained in the interval is
    returned.
    if Lmax < 0 then the line is considered at least a wave length long.
    if dx is supplied and larger than 0. a tuple of vectors contaning x and
    V(x) will be supplied
    
    Note this is only valid for a lossless TL
    """
    if P is not None:
        if Z is not None:
            if (V is not None) or (I is not None):
                raise ValueError(
                        "TL_Vmax : should have (P,Z) xor (V,I) as arguments.")
            else:
                rho = (Z - Z0)/(Z + Z0)
                gamma = np.abs(rho)
                a0 = np.sqrt(2*Z0*P/(1 - gamma**2))
        else:
            raise ValueError(
                        "TL_Vmax : should have (P,Z) xor (V,I) as arguments.")
    elif V is not None :
        if I is not None :
            if (P is not None) or (Z is not None):
                raise ValueError(
                         "TL_Vmax : should have (P,Z) xor (V,I) as arguments.")
            else:
                rho = (V - Z0 * I)/(V + Z0 * I)
                gamma = np.abs(rho)
                a0 = np.abs(V + Z0 * I)/2
        else:
            raise ValueError(
                        "TL_Vmax : should have (P,Z) xor (V,I) as arguments.")
                
    phi = np.angle(rho)
    phi = phi + 2*np.pi * (phi < 0)
    maxl = Lwave * phi / (4 * np.pi)
    if (Lmax is None) or (maxl < Lmax):
        vmax = a0*np.abs(1 + gamma)
    else:
        V0 = np.abs(1 + rho)
        V1 = np.abs(1 + rho * np.exp(-4j*np.pi*Lmax/Lwave))
        if V0 > V1:
            vmax, maxl = V0*a0, 0.
        else:
            vmax, maxl = V1*a0, Lmax
            
    if dx is None:
        return vmax, maxl
    else:
        if type(dx) in [float,int]:
            assert dx > 0
            if Lmax is None:
                Lmax = Lwave/2
            x = np.linspace(0., Lmax, int(Lmax/dx+1))
        else: # dx is a list of points
            x = np.array(dx)
        Vx = a0 * np.abs(1 + rho * np.exp(-4j*np.pi*x/Lwave))
        Ix = a0 * np.abs(1 - rho * np.exp(-4j*np.pi*x/Lwave))/Z0
        return vmax, maxl, (x, Vx,  Ix)
    
    
#===============================================================================
#
# T L _ V I _ S W R
#

def TL_VI_SWR(fMHz, **kwargs):
    """"
    Return VI standing wave data of a TL of length Lmax with properties defined 
    by A, Z, V, G, R, L and/or C and BC's given by either (V#, I# | Z# | Y#) |
    (I#, Z# | Yk#) | (P#, Z# | Y#) for (# = 1 | 2 : i.e. either ends of the TL).
    
    port 1 is defined as the load port while port 2 is defined as the source
    port (of no importance for the moment : later when possibly mixed BC's will
    be possible (e.g. (V1, V2)) the SW will be built by convention from the load
    port towards the source port).
    
    contrary to TL_Vmax and TL_Imax : the default is building the SW going into
    the port (see inwards). To obtain a SW external to the port use inwards=False..
      
    kwargs:
    
    A, Z, V, G, R, L, C : the usual TL properties
    
    inwards : build the SW into the port (True (default) | False]
    
    (# = 1 | 2)
    
    V#, I#: Voltage current on both ports (currents into the ports)
    
    P#, Z#, Y# : Power and impedance on both ports (power into the ports,
                 impedance and admittance looking into the ports)
    
    It is assumed that A, or R and G are small quantities
        
    Lmax : TL length (defaults to speedoflight/fMHz i.e. a wave length in vacuum)
    
    dx : sample distance along TL (default speedoflight/fMHz / 72 i.e. every 
                                   5 degrees)
                                   
    num : number of intervals (default 72) [note the vectors returned will be
          sized num+1 to include the initial point]
          
    return a dict:
    
    {'xs' : np.array(float,shape=(num+1,)),     # position along the TL
     'Vs' : np.array(complex, shape=(num+1,)),  # complex line voltage
     'Is' : np.array(complex, shape=(num+1,))   # complex lin current
     'Ps' : np.arrat(float, shape=(num+1,))     # active power
     'Ds' : np.array(float, shape=(num+1,))     # power loss
    }
    """
    TLkwargs = dict([(kw, kwargs.pop(kw)) 
                     for kw in ['A','Z','V','G','R','L','C'] if kw in kwargs])
    
    inwards = kwargs.pop('inwards', True)
    
    wave = speed_of_light / (fMHz * 1E6)
    
    if 'Lmax' in kwargs:
        Lmax = kwargs.pop('Lmax')
        if 'num' in kwargs:
            # have Lmax and num : dx will be found from those
            num = kwargs.pop('num')
            
            # can't use dx any more
            if 'dx' in kwargs:
                raise ValueError('TL_VI_SWR: only 2 out of Lmax, dx or num allowed')
            
        else:
            dx = kwargs.pop('dx', speed_of_light/(fMHz * 1E6)/72)
            
            # have Lmax and dx : dx (might be by default) will be tweaked to
            #                    have an integer number of intervals 
            num = round(Lmax/dx)            
        
        dx = Lmax / num
        
    else:
        if 'dx' in kwargs:
            dx = kwargs.pop('dx')
            if 'num' in kwargs:
                # have dx and num : Lmax will be found from those
                num = kwargs.pop('num')
                Lmax = num * dx
            
            else:
                # only have dx : Lmax will be default and num set accordingly
                Lmax = wave
                num = round(Lmax / dx)
                Lmax = dx * num # get Lmax
                
        elif 'num' in kwargs:
            num = kwargs.pop('num')
            # have only num : Lmax will be default and dx found accordingly
            Lmax = wave
            dx = Lmax / num
            
        else:
            # nothing was given: use defaults
            Lmax = wave
            num = 72
            dx = Lmax / num
            
    xs = np.linspace(0., Lmax, num+1)
    
    SZ = Scatter(fMHz=fMHz) # Zbase defaults to 50 Ohm
    dM = SZ.TL_VImatrix(dx, **TLkwargs)
    
    # we need to find Z0 because it is not necessarily given
    Z0 = SZ.TLproperties(lossless=True, **TLkwargs)[3]
    
    # all computations possible from data on port 1 (load side)
    if 'V1' in kwargs:
        V1 = kwargs.pop('V1')
        if 'I1' in kwargs:
            I1 = kwargs.pop('I1')
            
        elif 'Z1' in kwargs:
            I1 = V1 / kwargs.pop('Z1')
            
        elif 'Y1' in kwargs:
            I1 = V1 * kwargs.pop('Y1')
                    
        else:
            I1 = V1 / Z0
            
        # (*) we will proceed from port 1 stepping to port 2 using VI transfor-
        # mation matrices (see Scatter.VI_from_S()) except if inwards is False 
    
        VI = [V1, -I1 if inwards else I1]
        
    elif 'I1' in kwargs:
        I1 = kwargs.pop('I1')
        if 'Z1' in kwargs:
            Z1 = kwargs.pop('Z1')
            V1 = Z1 * I1
        
        elif 'Y1' in kwargs:
            Y1 = kwargs.pop('Y1')
            V1 = I1 / Y1
            
        else:
            V1 = Z0 * I1
            
        VI = [V1, -I1 if inwards else I1] # see (*) above
        
    elif 'Z1' in kwargs:
        Z1 = kwargs.pop('Z1')
                    
        P1 = kwargs.pop('P1', 1.) # default 1W
        I1 = np.sqrt(2*P1*np.real(Z1))
        V1 = Z1 * I1
        
        VI = [V1, -I1 if inwards else I1] # see (*) above
            
    elif 'Y1' in kwargs:
        Y1 = kwargs.pop('Y1')
                    
        P1 = kwargs.pop('P1', 1.) # default 1W
        I1 = np.sqrt(2*P1*np.real(1/Y1))
        V1 = Z1 * I1
        
        VI = [V1, -I1 if inwards else I1] # see (*) above
            
    # all the computations possible from dat on port 2 (source side)
    else:
        if 'V2' in kwargs:
            V2 = kwargs.pop('V2')
            if 'I2' in kwargs:
                I2 = kwargs.pop('I2')
                
            elif 'Z2' in kwargs:
                I2 = V2 / kwargs.pop('Z2')
    
            elif 'Y2' in kwargs:
                I2 = V2 * kwargs.pop('Y2')
    
            else:
                I2 = V2 / Z0
        
            # (#) we will proceed from port 2 stepping to port 1 using VI transfor-
            # mation matrices (see Scatter.VI_from_S()) except if inwards is False 
        
            VI = [V2, -I2 if inwards else I2]
            
        elif 'I2' in kwargs:
            I2 = kwargs.pop('I2')
            
            if 'Z2' in kwargs:
                Z2 = kwargs.pop('Z2')
                V2 = Z2 * I2
            
            elif 'Y2' in kwargs:
                Y2 = kwargs.pop('Y2')
                V2 = I2 / Y2
                
            else:
                V2 = Z0 * I2
            
            VI = [V2, -I2 if inwards else I2] # see (#) above

        elif 'Z2' in kwargs:
            Z2 = kwargs.pop('Z2')
                        
            P2 = kwargs.pop('P2', 1.) # default 1W
            I2 = np.sqrt(2*P2*np.real(Z2))
            V2 = Z2 * I2
            
            VI = [V2, -I2 if inwards else I2] # see (#) above

        elif 'Y1' in kwargs:
            Y2 = kwargs.pop('Y2')
                    
            P2 = kwargs.pop('P2', 1.) # default 1W
            I2 = np.sqrt(2*P2*np.real(1/Y2))
            V2 = Z2 * I2
            
            VI = [V2, -I2 if inwards else I2] # see (#) above

        else:
            # seem to have very little to go on knowing what the user wants
            raise ValueError('TL_VI_SWR: could not produce SWRs from supplied '
                             'arguments %r' % kwargs )
                            
    if kwargs:
        raise ValueError('TL_VI_SWR: unkown arguments: %r' % kwargs)
    
    Vs = [VI[0]]
    Is = [VI[1]]
    for x in xs[1:]:                                           # @UnusedVariable
        VI = np.dot(dM, VI)
        Vs.append(VI[0])
        Is.append(VI[1])

    Vs, Is = np.array(Vs), np.array(Is)
    Ps = 0.5 * np.real(np.conj(Vs)*Is)
    Ps_spl = UnivariateSpline(xs,Ps,k=2,s=0.)
    twodx = 2 * dx

    Ds_spl = Ps_spl.derivative()(xs)
    Ps_spl = Ps_spl(xs)

    Ds = ([(Ps[1] - Ps[0])/dx] + 
          [(Ps[k+1]-Ps[k-1])/twodx for k in range(1,len(Ps)-1)] +
          [(Ps[-1] - Ps[-2])/dx]) 
        
    return {'xs':xs, 'Vs':Vs, 'Is':Is, 'Ps': Ps, 'Ds':Ds,
            'Ps_spl':Ps_spl, 'Ds_spl':Ds_spl} 
    
#===============================================================================
#
# T L _ I m a x
#
def TL_Imax(Lmax, Z0, Lwave, P= None, V= None, I= None, Z= None, dx= None):
    """
    returns the maximum current and the distance along the TL of characteristic
    impedance Z0 where this current maximum occurs. If it is past the maximum
    length Lmax, then the actual current maximum obtained in the interval is
    returned.
    if Lmax < 0 then the line is considered at least a wave length long.
    if dx is supplied and larger than 0. a tuple of vectors contaning x and
    I(x) will be supplied
    
    Note this is only valid for a lossless TL
    """
    if P is not None:
        if Z is not None:
            if (V is not None) or (I is not None):
                raise ValueError(
                        "TL_Vmax : should have (P,Z) xor (V,I) as arguments.")
            else:
                rho = (Z - Z0)/(Z + Z0)
                gamma = np.abs(rho)
                a0 = np.sqrt(2*Z0*P/(1 - gamma**2))
        else:
            raise ValueError(
                        "TL_Vmax : should have (P,Z) xor (V,I) as arguments.")
    elif V is not None :
        if I is not None :
            if (P is not None) or (Z is not None):
                raise ValueError(
                        "TL_Vmax : should have (P,Z) xor (V,I) as arguments.")
            else:
                rho = (V - Z0 * I)/(V + Z0 * I)
                gamma = np.abs(rho)
                a0 = np.abs(V + Z0 * I)/2
        else:
            raise ValueError(
                        "TL_Vmax : should have (P,Z) xor (V,I) as arguments.")
                
    phi = np.imag(np.log(rho))
    maxl = Lwave * (phi + np.pi) / (4 * np.pi)
    if (Lmax is None) or (maxl < Lmax):
        imax = a0*np.abs(1 + gamma)/Z0
    else:
        I0 = np.abs(1 - rho)/Z0
        I1 = np.abs(1 - rho * np.exp(-4j*np.pi*Lmax/Lwave))/Z0
        if I0 > I1:
            imax, maxl = I0*a0, 0.
        else:
            imax, maxl = I1*a0, Lmax
            
    if dx is None:
        return imax, maxl
    else:
        if type(dx) in [float,int]:
            assert dx > 0
            if Lmax is None:
                Lmax = Lwave/2
            x = np.linspace(0., Lmax, int(Lmax/dx+1))
        else: # dx is a list of points
            x = np.array(dx)
        Vx = a0 * np.abs(1 + rho * np.exp(-4j*np.pi*x/Lwave))
        Ix = a0 * np.abs(1 - rho * np.exp(-4j*np.pi*x/Lwave))/Z0
        return imax, maxl, (x, Vx,  Ix)

#===============================================================================
#
# E q u i v a l e n t 2 P o r t P a r a m e t e r s
#
def Equivalent2PortParameters(S2P, Z0TL, signX12=+1., fMHz=None, Vc=1.):
    """
    given a lossless 2 port scattering matrix S2P for a reference TL impedance 
    Z0TL at a frequency fMHz and a wave propagation velocity Vc units of c0 in
    vacuum ( <speed_of_light> m/s ) return a synthesis of the 2 port as a 
    reactance X12 in between 2 lenghts of TL, L1 and L2.

    if fMHz and Vc are not given then the electric lenghts are returned with
    beta = 2 pi fMHz / Vc * c0 [Mm/s] -> beta*L1 and beta*L2

    S2P must be lossless : S2P.conjugate().T . S2P) = Identidy
    """

    S2Pa = np.array(S2P)
    loss = np.max(np.abs(np.dot(S2Pa.conj().T,S2Pa)-np.eye(2))) 
    if loss > 1e-4 :
        print("Equivallent2Port : Warning S2P does not appear to be lossless"
              " (%g)" % loss)
        
    # X12 = (1. if signX12 > 0. else -1.) * Z0TL/2 * np.sqrt(np.abs(S2Pa[1,1])**(-2) - 1)
    X12 = (1 if signX12 > 0 else -1) * Z0TL/2 * np.abs(S2Pa[0,1]/S2Pa[0,0])
        
    bL1 = - np.angle(- S2Pa[0,0] * (2j*X12/Z0TL + 1))/2 
    if bL1 < 0. : bL1 += np.pi
    
    if False:
        # this may lead to a wrong sign on S2Pa[0,1]
        bL2 = - np.angle(- S2Pa[1,1] * (2j*X12/Z0TL + 1))/2
        if bL2 < 0. : bL2 += np.pi
    else:
        # this should give the correct solution
        z = 1j*X12/Z0TL
        bL2 = np.angle(z/(z+0.5)/S2Pa[0,1]*np.exp(-1j*bL1))
        if bL2 < 0. : bL2 += 2*np.pi

#     print('bL1 ',bL1,', bl2', bL2)
#     print('bL1 ',bL1,', bl2', bL2)
    
    errS12 = np.abs(SZfromL1L2X(bL1, bL2, X12, Z0TL, fMHz=None).S[0,1] - S2Pa[0,1])
    if errS12 > 1e-4:
        print('\nwe appear to have to wrong sign ... err =', errS12)
        print('can we solve it  by adding pi to (one of) the bL''s')
        bL2 += np.pi
        errS12b = np.abs(SZfromL1L2X(bL1, bL2, X12, Z0TL, fMHz=None).S[0,1] - S2Pa[0,1])
        if errS12b > 1e-4:
            print('\nwe appear to have to wrong sign ... err =', errS12)
            print(' and could not be solved it  by adding pi to one of the bL''s')
            print('---> err =', errS12b,'\n')
            raise ValueError('Equivallent2Port : found no solution')
    
    if fMHz :
        beta = (2E6 * np.pi * fMHz) / (speed_of_light * Vc)
        return bL1 / beta, bL2 / beta, X12

    else:
        return bL1, bL2, X12
    

#===============================================================================
#
# S Z f r o m L 1 L 2 X
#
def SZfromL1L2X(L1, L2, X, Z0TL, fMHz=None, Vc=1):
    """if fMHz is not given it will be set to Vc*c0/2pi so that the lengths are
       interpreted as electric phases (beta*L).
    """
    fMHz = fMHz if fMHz else speed_of_light*Vc/(2E6*np.pi)
#     SZ.trlAZV(['1','2'],L1)
#     SZ.parallel_impedance('2', 1j*X)
#     SZ.trlAZV('2',L2)
    
    z = 1j*X/Z0TL
    Qd, Qt = - 0.5/(z+0.5), z/(z+0.5)
    bL1 = 2E6*np.pi*fMHz/(Vc*speed_of_light)*L1
    bL2 = 2E6*np.pi*fMHz/(Vc*speed_of_light)*L2
    S = np.array([[Qd*np.exp(-2j*bL1), Qt*np.exp(-1j*(bL1+bL2))],
                  [Qt*np.exp(-1j*(bL1+bL2)), Qd*np.exp(-2j*bL2)]])
    SZ = Scatter(fMHz=fMHz, Zbase=Z0TL, V=Vc, Portnames=['1','2'], S=S)
    
#     print()
#     print('SZfromL1L2X',np.max(np.abs(SZ.S-S)))
#     printRI(SZ.S)
#     printRI(S)
    
#     print(np.abs(SZ.S-S))
#     print()
    
    return SZ
    
#===============================================================================
#
#  S e l f   t e s t i n g   c o d e 
#
if __name__ == "__main__":   ## self testing code: will not run during an import
    
    from scipy.interpolate.fitpack2 import UnivariateSpline
    import matplotlib.pyplot as pl
    from Utilities.printMatrices import printMA
    import sys
    
    
    alltests = ['repr','trlCoax','serie_impedance','SZfromL1L2X', 'TSFimport',
                'TL_Imax', 'check_AddTL', 'TL_GRLC', 'connectports', 'joinports'
                'TL_VI_SWR','convert_general','__iadd_-']
    
    
    tests = ['convert_general']

#---------------------------------------------------------------- test __iadd__    

    if '__iadd__' in tests:
        
        fMHz = 30.
        Lw = speed_of_light / (fMHz*1E6)
        SZ1 = Scatter(fMHz=30., Zbase=20.0, S=[[0j,1],[1,0j]], Portnames=['1','2'])
        SZ1.trlAZV('2',Lw/3)
        # SZ1.addsignal('2')
    
        print('SZ1.Sigs -> ', repr(SZ1.Ssigs))
        
        SZ2 = Scatter(fMHz=30., Zbase=20.0, S=[[0j,1],[1,0j]], Portnames=['2','3'])
        SZ2.trlAZV('3',2*Lw/3)
        # SZ2.addsignal('2')
        
        print('SZ2.Sigs -> ', repr(SZ2.Ssigs))
        
        SZ1 += SZ2
        print(SZ1)
        
#---------------------------------------------------------- test convert_general    

    if 'convert_general' in tests:
        
        if False:
            cnv = lambda m, a : m * np.exp(1j*np.pi/180*a)
                                           
            Sgen = [[cnv(0.437902776891251, 160.527003320128),
                     cnv(0.534371906100679, -24.9967921827212),
                     cnv(0.585208268537266, -44.1373856394698),
                     cnv(0.424522209968546, -27.9351558402061)],
                     
                    [cnv(0.534371906100678, -24.9967921827213),
                     cnv(0.503485927056904, -178.250150582642), 
                     cnv(0.534371809330247, -24.9973972218985),
                     cnv(0.418802288202546, -29.2159417628974)], 
                     
                    [cnv(0.585208268537261, -44.1373856394697),
                     cnv(0.534371809330244, -24.9973972218984),
                     cnv(0.437902873152083, 160.52547398406),
                     cnv(0.424522232483903, -27.9361659632805)],
                    
                    [cnv(0.424522209968545, -27.935155840206),
                     cnv(0.418802288202547, -29.2159417628975),
                     cnv(0.424522232483906, -27.9361659632805),
                     cnv(0.681297593361692, 168.164960901797)]]
            
    #         SZ = Scatter(fMHz=47.5, Zbase=20.,S=[[0j,1],[1,0j]],Portnames=['a','b'])
    #         SZ.trlAZV('b', 0.001, Z=50.)
    # #         SZ.trlAZV(['c','d'], 0.0, Z=50.)
    # #         SZ.trlAZV(['e','f'], 0.0, Z=50.)
    # #         SZ.joinports('b','d','f')
    #         
    #         print(SZ)
    #         printMA(SZ.S)
    #         S50 = convert_general(50.,SZ.S,SZ.Zbase)
    #         printMA(S50)
    #         printMA(convert_general(SZ.Zbase,S50,50.))
    #         printMA(SZ.Get_Z())
            
            Z0s = [12.6, 12.6, 12.6, 23.7]
            Zbase = 50.
            S50 = convert_general(50., Sgen, [12.6, 12.6, 12.6, 23.7], 'std', 'std')
            print('Sgen mixed std Z= %r Ohm' % Z0s)
            printMA(Sgen)
            print('S50(Sgen) std Z= %r Ohm' % Zbase)
            printMA(S50)
            Sgen2 = convert_general([12.6, 12.6, 12.6, 23.7], S50, 50, 'std', 'std')
            print('Sgen(S50) mixed std Z= %r Ohm' % Z0s)
            printMA(Sgen2)
    
    ##      Obsolete
    # 
    #         S50_2 = convert_general_old(50., Sgen, [12.6, 12.6, 12.6, 23.7])
    #         print('S50(Sgen) old')
    #         printMA(S50_2-S50)
        
        elif True:
            pass
        

    
    if 'TL_VI_SWR' in tests:
        fMHz, Zbase, lngth, A = 11., 30., 1.5, 0.05
        SZ = Scatter(fMHz=fMHz, Zbase=Zbase)
        SZ.AddTL(['1','2'], lngth, A=A)
        SZ.addsignal('1')
        SZ.termport('1', 0.5*np.exp(1j*30*np.pi/180))
        sigs = SZ.calcsignals({'1':['V','I','P'],'2':['V','I','P']}, {'2':1.})
        V1, I1, P1 = sigs['1'][0], sigs['1'][1], sigs['1'][2]
        V2, I2, P2 = sigs['2'][0], sigs['2'][1], sigs['2'][2]
        print(P1, P2)
        
        
        pl.figure()
        
        x1s, V1s, I1s = TL_Vmax(lngth, Zbase, 300./fMHz, V=V1, I=-I1, dx=0.05)[2]
        SWnew = TL_VI_SWR(fMHz, Z=Zbase, Lmax=lngth, A=A, V1=V1, I1=I1, dx=0.05)
        
        pl.plot(x1s, V1s,'r.-', label='TL_Vmax [1]')
        pl.plot(SWnew['xs'],np.abs(SWnew['Vs']), 'b.', label='TL_VI_SW [1]')
        
        
        x2s, V2s, I2s = TL_Vmax(lngth, Zbase, 300./fMHz, V=V2, I=-I2, dx=0.05)[2]
        SW2new = TL_VI_SWR(fMHz, Z=Zbase, Lmax=lngth, A=A, V2=V2, I2=I2, dx=0.05)
                
        pl.plot(x2s[-1]-x2s, V2s,'m', label='TL_Vmax [2]')
        pl.plot(SW2new['xs'][-1] -  SW2new['xs'],np.abs(SW2new['Vs']), 'c', label='TL_VI_SW [2]')
        
        
        x3s, V3s, I3s = TL_Vmax(lngth, Zbase, 300./fMHz, V=V2, I=I2, dx=0.05)[2]
        SW3new = TL_VI_SWR(fMHz, Z=Zbase, Lmax=lngth, A=A, V2=V2, I2=I2, dx=0.05,
                           inwards=False)
        
        pl.plot(x3s + lngth, V3s,'m-.', label='TL_Vmax [3]')
        pl.plot(SW3new['xs'] + lngth,np.abs(SW3new['Vs']), 'c-.', label='TL_VI_SW [3]')
        
        
        pl.grid()
        pl.xlabel('distance [m]')
        pl.ylabel('voltage [V')
        pl.legend(loc='best')
                                
    if 'joinports' in tests:
        # the old way
        SZ = Scatter(fMHz=11.)
        SZ.AddTL(['a','b'], 2.)
        SZ.AddTL(['c','d'], 2.5)
        SZ.AddTL(['e','f'], 2.55)
        SZ.join2ports('a', 'c', newname='e_')
        SZ.join2ports('e_', 'e', newname='g')
        SZ.sortports()
        # print(SZ)
        
        # the new way
        SZ2 = Scatter(fMHz=11.)
        SZ2.AddTL(['a','b'], 2.)
        SZ2.AddTL(['c','d'], 2.5)
        SZ2.AddTL(['e','f'], 2.55)
        SZ2.joinports('a', 'c', 'e',newname='g')
        SZ2.sortports()
        print(SZ2)

        # the adding way
        SZ1 = Scatter(fMHz=11.)
        SZ1.AddTL(['a','b'], 2.)
        SZ1.AddTL(['c','d'], 2.5)
        SZ1.AddTL(['e','f'], 2.55)
        SZ1.joinNports('a', 'c', 'e',newname='g')
        SZ1.sortports()
        print(SZ1)
        
    if 'connectports' in tests:
        SZ = Scatter(fMHz=30., Zbase=30.)
        SZ.AddTL(['a','b'], 1.0)
        SZ.AddTL(['c','d'], 1.0)
        SZ.AddTL(['e','f'], 1.0)
        SZ.joinports('b', 'c','_e')
        SZ.connect2ports('e', '_e')
        SZ.sortports()
        print(SZ)
        
        SZ = Scatter(fMHz=30., Zbase=30.)
        SZ.AddTL(['a','b'], 1.0)
        SZ.AddTL(['c','d'], 1.0)
        SZ.AddTL(['e','f'], 1.0)
        SZ.connectports('b', 'c', 'e')
        SZ.sortports()
        print(SZ)
        

    if 'TL_GRLC' in tests:
        fMHz = 11.
        w = 2E6*np.pi*fMHz
        Zc = 30.
        gamma = 1j*w/3E8
        SZ = Scatter(fMHz=fMHz)
        gamma1, Zc1 = SZ.TLproperties(G=0.01, R=0.005, L=100., C=111.11111)
        gamma1, Zc1 = SZ.TLproperties(A=0.001, Z=30, V=1)
        print(gamma1, Zc1)
        G, R, L, C = TL_GRLC(gamma1, Zc1, fMHz=fMHz)
        print(G, 'Mho/m', R, 'Ohm/m', L, 'nH/m', C, 'pF/m', 
          np.sqrt(L*1E-9/(1E-12*C)), 'Ohm', 
          np.sqrt(1/((1E-9*L*1E-12*C)))/1E6, 'Mm/s')

             
    if 'TL_VI_SWR_' in tests:
        fMHz = 50.
        lngth = 5. ## [m]
        num= 20
        Zin = (5 + 0j)
        V1 = 1.
        A, Z, V = 0.05, 30., 1.
        
        result = TL_VI_SWR(fMHz, num=num, A=A, Z=Z, V=V, V1=V1, Z1=Zin)
            
        xs, Vs, Is = result['xs'], result['Vs'], result['Is']
        Ps, Ds = result['Ps'], result['Ds']
        Ps_spl, Ds_spl = result['Ps_spl'], result['Ds_spl']
        
        pl.figure(figsize=(12,12))
        pl.subplot(3,1,1)
        pl.plot(xs, np.abs(Vs))
        pl.title('Line Voltage [V]')
        pl.grid()
        
        pl.subplot(3,1,2)
        pl.plot(xs, Ps,'r',label='origin')
        pl.plot(xs,Ps_spl,'b',label='spline')
        pl.grid()
        pl.title('Power Flow [W]')
        pl.legend(loc='best')
        
        pl.subplot(3,1,3)
        pl.plot(xs, Ds,'r',label='fin.diff')
        pl.plot(xs, Ds_spl, 'b', label='spline')
        pl.grid()
        pl.title('Power Loss [W/m]')
        pl.legend(loc='best')
        pl.xlabel('position [m]')
        
        pl.tight_layout()
            
    
    if 'check_AddTL' in tests:
        for iwt in [-1,1]:
            print('checking AddTL method ... (iwt=%r)' % iwt)
        
            print('using trlAZV:')
            SZold = Scatter(fMHz=10., Zbase=30., iwt=iwt)
            SZold.trlAZV(['p1','p2'], 1.23, A=0.02, Z=40., V=0.8)
            print(SZold)
        
            print('using AddTL method ...')
            SZnew = Scatter(fMHz=10., Zbase=30., iwt=iwt)
            SZnew.AddTL(['p1','p2'], 1.23, A=0.02, Z=40., V=0.8)
            print(SZnew)
        
        for iwt in [-1,1]:
            print('checking AddTL method ... (iwt=%r)' % iwt)
        
            print('using trlGRLC:')
            SZold = Scatter(fMHz=10., Zbase=30., iwt=iwt)
            SZold.trlGRLC(['p1','p2'], 1.23, G=0.01, R=0.02, L=180., C=125.)
            print(SZold)
        
            print('using AddTL method ...')
            SZnew = Scatter(fMHz=10., Zbase=30., iwt=iwt)
            SZnew.AddTL(['p1','p2'], 1.23, G=0.01, R=0.02, L=180., C=125.)
            print(SZnew)
        
    if 'repr' in tests :
        # standard case
        SZ = Scatter(fMHz=10.)
        SZ3 = Scatter(fMHz=10.)
        SZ.trlAZV(['a','b'],2.)
        SZ4 = SZ.copy()
        print(SZ)
        print(repr(SZ))

        # try the eval thingy :
        SZ2 = eval(repr(SZ))
        print(SZ2)
        print("is eval(repr(SZ))  == SZ  ?", SZ2 == SZ)
        print("is eval(repr(SZ3)) == SZ3 ?", eval(repr(SZ3)) == SZ3)
        print("is SZ4 == SZ              ?", SZ4 == SZ)
        print("is SZ3 == SZ              ?", eval(repr(SZ3)) == SZ)
        print("is SZ == SZ               ?", SZ == SZ)
        print(repr(SZ3))
        print(SZ.__dict__)
        print(SZ3.__dict__)
        print('fMHz     ', SZ.fMHz == SZ3.fMHz)
        print('Signal   ',SZ.Signal == SZ3.Signal)
        print('Ssigs    ',SZ.Ssigs == SZ3.Ssigs)
        print('S        ',SZ.S == SZ3.S)
        print('iwt      ',SZ.iwt == SZ3.iwt)
        print('Zbase    ',SZ.Zbase == SZ3.Zbase)
        print('Portname ',SZ.Portname == SZ.Portname)
        print('pfmt     ',SZ.pfmt == SZ3.pfmt)
        print(SZ.__dict__ == SZ3.__dict__)

    if 'trlCoax' in tests:

        def printdict(d):
            print('Fail' if d[2] is 5 else '    ',end='')
            for k, v in [ (x, y) for x, y in sorted(d[0].items()) if x != '-']:
                print(' %s%s = %10.4f' % ('*' if k in d[1] else ' ',k,v), end=',')
            print('     ',d[1])
            
        SZ = Scatter(fMHz=42.,Zbase=20.)
        SZ.trlCoax(['p1','p2'],1.,info=True)
        print(SZ)

        # all of these should produce the same TL
        args = {'Z': 30, 'V': 0.75, 'L': 90., 'C': 130., 'epsr': 2., 'mur': 3.,
                'lnDs': 0.6, '-': None} # , 'OD': 0.231, 'ID': 0.140, 'NN': None}
        pars = ['Z', 'V', 'L', 'C', 'epsr', 'mur', 'lnDs']

        # print(SZ.trlCoax(['p1','p2'],1.,**{}))
        
        for k1, p1 in enumerate(pars):
            for k2, p2 in enumerate(pars):
                if k2 >= k1:
                    for k3, p3 in enumerate(pars):
                        if k3 >= k2 :
                            kwargs = { p1: args[p1], p2 : args[p2], p3:args[p3] }
                            print('%5s, %5s, %5s :' % (p1,p2,p3), end='')
                            printdict(SZ.trlCoax(['p1','p2'],1.,**kwargs))
            print()
    
    if 'series_impedance' in tests:
        print(__version__)
        SZ = Scatter(fMHz=42.,Zbase=50.)
        SZ.trlAZV(['A','B'], 0.25, A=0., Z=20., V=1.)
        SZ.series_impedance('B', 1/(1j*2e6*np.pi*42*50e-12))
        print(SZ)
        
#-------------------------------------------------------------- test SZfromL1L2X

    if 'SZfromL1L2X' in tests:
#         print(__version__)
#         S = [[-0.8429   -0.0236j,     -0.5310   -0.0840j], 
#              [-0.5310   -0.0840j,      0.8090   +0.2376j]]
#         print(S)
        
        fMHz = 150 # wavelength is 2m
        Z0TL = 2
        L10, L20, X0 = 0.65, 0.35, +2.
        SZ = SZfromL1L2X(L10,L20,X0,Z0TL,fMHz=fMHz)
        print(SZ)
        for s in [-1,1]:
            L1, L2, X = Equivalent2PortParameters(SZ.S, fMHz=fMHz, signX12=s,Z0TL=Z0TL)
            print(L1,L2,X)
            SZ0 = SZfromL1L2X(L1,L2,X,Z0TL,fMHz=fMHz)
            print(SZ0)

#----------------------------------------------------------------------- __sub__

    if '__sub__' in tests:
        
        # need some suitable Scatter object
        
        fMHz = 150 # wavelength is 2m
        Z0TL = 2
        L10, L20, X0 = 0.65, 0.35, +2.
        SZ = SZfromL1L2X(L10,L20,X0,Z0TL,fMHz=fMHz)
        print(SZ)
        SZ0 = SZ + SZ.renameports({'1':'3','2':'4'})
        print(SZ0)
        SZ0.joinports('2','4','2')
        print(SZ0)

        SZ2 = SZ.renameports({'2':'3'})
        print(SZ2)
        
        SZ3 = SZ + SZ2
        print(SZ3)
        
        SZ4 = SZ3 - SZ2
        print(SZ4.sortports(SZ.Portname))
        print(SZ)
        
#--------------------------------------------------------------------- TSFimport

    if 'TSFimport' in tests:
        SZ = Scatter(fMHz=42,Zbase=30,
                     TSF='/home/frederic/Dropbox/PythonRepository'
                     '/ILA_simulator/ILA_simulator/Data/HFSS-MWS-models'
                     '/servicestub.s2p')
        print(SZ)
        
#------------------------------------------------------------------------TL_Imax

    if 'TL_Imax' in tests:
        l1 = 1.5
        fMHz = 30. # MHz so that a wavelength is (about) 10m
        Z0 = 30.   # Ohm
        SZ = Scatter(fMHz=fMHz,Zbase=Z0)
        SZ.trlAZV(['short','feed'], L=l1)
        print(SZ)
        SZ.termport('short', -1)
        Z = Z_from_S(SZ.S, SZ.Zbase)[0,0]
        Imax, Lmax = TL_Imax(None,Z0,300./fMHz,V=Z,I=1.)
        print('Z={Z.real:.3f}{Z.imag:+.3f}j Ohm, Lmax={Lmax:.3f}m, '
              'Imax={Imax:.3f}'.format(Z=Z,Lmax=Lmax,Imax=Imax))
        X = np.imag(Z)
        alpha = np.arctan2(Z0,X)
        l2min = (alpha/np.pi-0.5)*150/fMHz
        l2min += 150./fMHz if l2min < 0 else 0.
        print('alpha={alpha:.3f}rad, l2min={l2min:.3f}m'.format(alpha=alpha,
                                                                l2min=l2min))
        cbl = np.cos(2*np.pi*fMHz/300*l2min)
        sbl = np.sin(2*np.pi*fMHz/300*l2min)
        print('Vr={Vr:.3f}V, Ir={Ir:.3f}A, Ir*={Irs:.3f}A'.format(
               Vr = ((cbl+Z0/X*sbl)*np.abs(Z)), 
               Ir= ((sbl-Z0/X*cbl)/Z0*np.abs(Z)),
               Irs= 1/(Z0*X)*np.sqrt(X**2+Z0**2)*np.abs(Z)
              ))
    
#-------------------------------------------------------------------------------
    
    pl.show()
    
#===============================================================================
# /
#/  Self testing code
#
##    if 10 in tests:
##        print('---------------------------------------------------------------'
##        print("Test 10 : Get_S, Get_S1"
##        print('---------------------------------------------------------------'
##        
##        Z = np.array([[1,-1j],[-1j, 2]])
##        print 'Scatter object with Zbase = 30. created with Z:'
##        print Z
##        print
##        SZ = Scatter(10.,Zbase=30,Z = Z)
##        print SZ
##        
##        print 'Scatter.Get_S(Z0=30):'
##        print SZ.Get_S(Z0=30)
##        print
##
##        print 'Scatter.Get_S1(Z0=30):'
##        print SZ.Get_S1(Z0=30)
##        print
##
##
##        S1 = SZ.Get_S(Z0=60)
##        print 'Scatter.Get_S(Z0=60):'
##        print S1
##        print
##
##        S2 = SZ.Get_S1(Z0=60)
##        print 'Scatter.Get_S1(Z0=60):'
##        print S2
##        print
##
##        SZ1 = Scatter(10.,Zbase=60., S=S1)
##        print 'SZ1.Get_Z():'
##        print SZ1.Get_Z()
##        print
##        
##        SZ2 = Scatter(10.,Zbase=60., S=S2)
##        print 'SZ2.Get_Z():'
##        print SZ2.Get_Z()
##        print
##        
####--------------------------------------------------------------------------------------##
##    
##    if 9 in tests:
##        print '---------------------------------------------------------------'
##        print "Test 9 : Get, Get_S, Get_Z, Get_Y"
##        print '---------------------------------------------------------------'
##
##        Z = np.array([[1,-1j],[-1j, 2]])
##        print 'Scatter object with Zbase = 30. created with Z:'
##        print Z
##        print
##        SZ = Scatter(10.,Zbase=30,Z = Z)
##        print SZ
##        
##        print 'Scatter.Get_S(Z0=30):'
##        print SZ.Get_(Z0=30)
##        print
##        print 'Inverse of Scatter.Get("Y"):'
##        print np.linalg.inv(SZ.Get('Y'))
##        print
##
##        print 'Z_from_S(Scatter.Get("S",Z0=15.),Zbase=15.)'
##        print Z_from_S(SZ.Get('S',Z0=15),Zbase=15.)
##        print
##        
####--------------------------------------------------------------------------------------##
##    
##    if 8 in tests:
##        print '---------------------------------------------------------------'
##        print "Test 8 : Equivalent2PortParameters"
##        print '---------------------------------------------------------------'
##
##        fMHz, Z0TL = 30., 20.
##
##        La, Lb, Lss = 1., 1.001, 25./fMHz-0.001
##        SZ = Scatter(fMHz,Zbase=Z0TL)
##        SZ.trlAZV(['a','b'],La)
##        SZ.trlAZV(['sc','ss'],Lss)
##        SZ.termport('sc',-1.)
##        SZ.joinports('b','ss','b')
##        SZ.trlAZV('b',Lb)
##        
##        print '\nStarting SZ : La=%.3fm, Lb=%.3fm, Ls=%.3fm\n' % (La,Lb,Lss)
##        print SZ
##
##        L1, L2, X12 = Equivalent2PortParameters(SZ.S,Z0TL,signX12=+1.,fMHz=fMHz)
##        LS = np.arctan(X12/Z0TL)/(2*np.pi*fMHz/300.)
##        print L1, L2, LS
##
##        SZ2 = Scatter(fMHz,Zbase=Z0TL)
##        SZ2.trlAZV(['a','b'],L1)
##        SZ2.trlAZV(['sc','ss'],LS)
##        SZ2.termport('sc',-1.)
##        SZ2.joinports('b','ss','b')
##        SZ2.trlAZV('b',L2)
##
##        print '\nSZ from Equivalent Parameters La=%.3fm, Lb=%.3fm, Ls=%.3fm\n' % (L1,L2,LS)
##        print SZ2    
##
####--------------------------------------------------------------------------------------##
##    if 7 in tests:
##        print '---------------------------------------------------------------'
##        print "Test 7 : test sortports"
##        print '---------------------------------------------------------------'
##
##        SZ = Scatter(30.,Zbase=20.,S=[[0.5,-0.3,-0.3],[-0.2,0.4,-0.2],[-0.1,-0.1,0.3]],Portnames=['a','b','c'])
##        print SZ
##        print SZ.sortports(['c','a','b'])
##        print SZ.sortports(cmp)
##        print SZ.sortports(lambda x, y : -cmp(x,y))
##        print SZ.sortports(None)
##        
####--------------------------------------------------------------------------------------##
##    if 6 in tests:
##        print '---------------------------------------------------------------'
##        print "Test 6 : test calcsignals"
##        print '---------------------------------------------------------------'
##
##        fMHz = 42
##        
##        s1 = Scatter(fMHz, Zbase = 30) # part 1 : with some signals
##        s2 = Scatter(fMHz, Zbase = 30) # part 2 : with other signals
##        s3 = Scatter(fMHz, Zbase = 30) # the same but in one go
##
##        d = 0.25
##        N = 3
##
##        sigls1 = {}
##        sigls2 = {}
##        for k in range(N):
##            
##            s1.trlAZV(["s1p%03d" % k,"s1p%03d" % (k+1)],d)
##            if k is 0:
##                s1.addsignal("s1p%03d" % (k),"s1a%03d" % (k))
##                sigls1['s1a%03d' % (k)] = ['V','I','Z-','P-','P+','PWR']
##            s1.addsignal("s1p%03d" % (k+1),"s1a%03d" % (k+1))
##            sigls1['s1a%03d' % (k+1)] = ['V','I','Z-','P-','P+','PWR']
##            
##            s2.trlAZV(["s2p%03d" % k,"s2p%03d" % (k+1)],d)
##            if k is 0:
##                s2.addsignal("s2p%03d" % (k),"s2a%03d" % (k))
##                sigls2['s2a%03d' % (k)] = ['V','I','Z-','P-','P+','PWR']
##            s2.addsignal("s2p%03d" % (k+1),"s2a%03d" % (k+1))
##            sigls2['s2a%03d' % (k+1)] = ['V','I','Z-','P-','P+','PWR']
##
##        s4 = s1 + s2
##        print
##        print 's4 ...'
##        print s4
##        s4.connectports('s2p000','s1p%03d'%N)
##        print
##        print 's4 ... reconnected '
##        print s4
##
##        sigls3 = {}
##        for k in range(2*N):
##            
##            s3.trlAZV(["s3p%03d" % k,"s3p%03d" % (k+1)],d)
##            if k is 0:
##                s3.addsignal("s3p%03d" % (k),"s3a%03d" % (k))
##                sigls3['s3a%03d' % (k)] = ['V','I','Z-','P-','P+','PWR']
##            s3.addsignal("s3p%03d" % (k+1),"s3a%03d" % (k+1))
##            sigls3['s3a%03d' % (k+1)] = ['V','I','Z-','P-','P+','PWR']
##            
##        print
##        print 's3 ...'
##        print s3
##
##        sigls4 = sigls1.copy()
##        sigls4.update(sigls2)
##        print "S4 ..."
##        tsigls4 = s4.calcsignals(sigls4,{'s1p000':1,'s2p%03d' % N:0.5*np.exp(0j*np.pi/180)}) # fwd wave= 0 ==> is like a dummy load
##        for sig in s4.Signal:
##            print ('%8s: V = %7.3f, I = %7.3f, Z = %7.3f%+8.3fj, P- = %7.3f, P+ = %7.3f, PWR = %7.3f' %
##               (sig,abs(tsigls4[sig][0]),abs(tsigls4[sig][1]),
##                np.real(tsigls4[sig][2]),np.imag(tsigls4[sig][2]),
##                tsigls4[sig][3],tsigls4[sig][4],tsigls4[sig][5]))
##
##        print "S3 ..."
##        tsigls3 = s3.calcsignals(sigls3,{'s3p000':1,'s3p%03d' % (2*N):0.5*np.exp(0j*np.pi/180)}) # fwd wave= 0 ==> is like a dummy load
##        for sig in s3.Signal:
##            print ('%8s: V = %7.3f, I = %7.3f, Z = %7.3f%+8.3fj, P- = %7.3f, P+ = %7.3f, PWR = %7.3f' %
##               (sig,abs(tsigls3[sig][0]),abs(tsigls3[sig][1]),
##                np.real(tsigls3[sig][2]),np.imag(tsigls3[sig][2]),
##                tsigls3[sig][3],tsigls3[sig][4],tsigls3[sig][5]))
##
####        # make a connextion between porta now
####        len3 = 0.3
####        s3.trlAZV(['s3p0','s3p1'],N*d)
####        
####        tsigls4 = s4.calcsignals(sigls,{'p000':1,'p%03d' % N:0.5*np.exp(30j*np.pi/180),'s3p0':0.0,'s3p1':0.0}) # fwd wave= 0 ==> is like a dummy load
####        
####        for sig in ['a%03d' % N,'p000']:
####            print ('%8s: V = %7.3f, I = %7.3f, Z = %7.3f%+8.3fj, P- = %7.3f, P+ = %7.3f, PWR = %7.3f' %
####               (sig,abs(tsigls4[sig][0]),abs(tsigls4[sig][1]),
####                np.real(tsigls4[sig][2]),np.imag(tsigls4[sig][2]),
####                tsigls4[sig][3],tsigls4[sig][4],tsigls4[sig][5]))
####        
####
####        sigls4 = {'s3p0':['V','I','Z-','P-','P+','PWR'],'p%03d'%0:['V','I','Z-','P-','P+','PWR']}
####        tsigls5 = s4.calcsignals(sigls4,{'s3p0':1,'p%03d' % 0:0.5*np.exp(30j*np.pi/180)}) # fwd wave= 0 ==> is like a dummy load
####        for sig in ['s3p0','p000','a%03d'%N]:
####            print ('%8s: V = %7.3f, I = %7.3f, Z = %7.3f%+8.3fj, P- = %7.3f, P+ = %7.3f, PWR = %7.3f' %
####               (sig,abs(tsigls5[sig][0]),abs(tsigls5[sig][1]),
####                np.real(tsigls5[sig][2]),np.imag(tsigls5[sig][2]),
####                tsigls5[sig][3],tsigls5[sig][4],tsigls5[sig][5]))
####        
####        
####        slist = tsigls.keys()
####        slist.sort()
####        Vs = []
####        Is = []
####        for sig in slist:
####            if sig == 'p000':
####                Vs = [tsigls[sig][0]] + Vs 
####                Is = [tsigls[sig][1]] + Is
####            else:
####                Vs.append(tsigls[sig][0])
####                Is.append(tsigls[sig][1])
####        
####        figure()
####        subplot(2,1,1)
####        plot(range(len(Vs)),np.angle(Vs),'r.',label=r'$phase V$')
####        plot(range(len(Is)),np.angle(Is),'b.',label=r'$phase I$')
####        plb.legend(shadow=False,loc='upper right').draw_frame(0)
####        
####        subplot(2,1,2)
####        plot(range(len(Vs)),np.abs(Vs),'r.',label=r'$ampl V$')
####        plot(range(len(Is)),np.abs(Is)*s1.Zbase,'b.',label=r'$ampl Z_0I$')
####        plb.legend(shadow=False,loc='upper right').draw_frame(0)
####        
####        show()
##        
####--------------------------------------------------------------------------------------##
##    if 5 in tests:
##        print '---------------------------------------------------------------'
##        print "Test 5 : read and interpolate Riccardo's matrices for d = 2cm"
##        print '---------------------------------------------------------------'
##        Zant = readZtopica(42.0,2)
##        for k in range(8):
##            for l in range(8):
##                print "%7.2f%+7.2fj " % (real(Zant[k,l]), imag(Zant[k,l])),
##            print
##        
####--------------------------------------------------------------------------------------##
##    if 4 in tests:
##        print '----------------------------------------------------'
##        print "Test 4 : read and interpolate Riccardo's matrices"
##        print '----------------------------------------------------'
##        lds = arange(-2.0,2.0,0.1)
##        lZs = []
##        for d in lds:
##            Zant = readZtopica(42.0,d)
##            lZs.append(real(Zant[0,0]))
##        for k in range(8):
##            for l in range(8):
##                print "%7.2f%+7.2fj " % (real(Zant[k,l]), imag(Zant[k,l])),
##            print
###        plot(lds,lZs)
###        show()
##        
####--------------------------------------------------------------------------------------##
##    if 3 in tests:
##        print '----------------------------------------------------'
##        print 'Test 3 : creating another simple circuit'
##        print '----------------------------------------------------'
##        s1 = Scatter(30,Zbase=30,iwt=-1) ## creates a Scatter object without ports
##        s1.trlRLC(["p1","p2"],0.3,R=4,L=180,C=165)
##        s1.addsignal("p1","sp1")
##        s1.addsignal("p2","sp2")
##        s1.termport("p1",-1)
##        print s1
##        s1.trlAZV("p2",75./s1.fMHz,A=0,V=1,Z=30)
##        s1.addsignal("p2","sp3")
##        s1.renameport("p2","pp2")
##        print s1
##        
####--------------------------------------------------------------------------------------##
##    if 2 in tests:
##        print '----------------------------------------------------'
##        print 'Test 2 : read an ILA touchstone file'
##        print '----------------------------------------------------'
##        s1 = Scatter(42, Zbase = 50, Portnames=["a1","a2","a3","a4","a5","a6","a7","a8"],TSF="ILA-D=00.0cm.s8p")
##        print s1
##        for k in range(8):
##            s1.addsignal("a"+str(k+1),"a"+str(k+1))
##        print s1
##        Zant = Z_from_S(s1.S,s1.Zbase)
##        for k in range(8):
##            for l in range(8):
##                print "%7.2f%+7.2fj " % (real(Zant[k,l]), imag(Zant[k,l])),
##            print
##        s2 = Scatter(42, Portnames=["a1","a2","a3","a4","a5","a6","a7","a8"],TSF="ILA-D=00.0cm.s8p")
##        print s2
##        Zant2 = Z_from_S(s2.S,s2.Zbase)
##        for k in range(8):
##            for l in range(8):
##                print "%7.2f%+7.2fj " % (real(Zant2[k,l]), imag(Zant2[k,l])),
##            print
##
####--------------------------------------------------------------------------------------##
##    if 1 in tests:
##        print '----------------------------------------------------'
##        print 'Test 1 : create a simple circuit (just a TL)'
##        print '----------------------------------------------------'
##        s1 = Scatter(42,Zbase=30,iwt=-1) ## creates a Scatter object without ports
##        lTL = 12.0     # [m] length of TL
##        ZTL = 30.0    # [Ohm] characteristic impedance
##        N = 100        # [1] number of points
##        for k in range(N):
##            s1.trlAZV(["p"+str(k),"p"+str(k+1)],lTL/float(N),A=0.00,V=1,Z=ZTL)
##            if k is 0:
##                s1.addsignal("p"+str(k),"sp"+str(k))
##            s1.addsignal("p"+str(k+1),"sp"+str(k+1))
##        print s1
##        print
##        s1.termport("p0",-1)
##        xs = [lTL/float(N)*k for k in range(0,len(s1.Ssigs),2)]
##        vs = [abs(s1.Ssigs[k].item()) for k in range(0,len(s1.Ssigs),2)]
##        plot(xs,vs,'ro-')
##        show()
##        
##        print "checking with a direct computation"
##        print
##        s2 = Scatter(42,Zbase=30,iwt=-1) ## creates a Scatter object without ports
##        s2.trlAZV(["p0","p"+str(N)],lTL,A=0.00,V=1,Z=ZTL)
##        s2.addsignal("p0","sp0")
##        s2.addsignal("p"+str(N),"sp"+str(N))
##        print s2
##        
####--------------------------------------------------------------------------------------##
##    if 0 in tests:
##        print '----------------------------------------------------'
##        print 'Test 0 : create a simple circuit (just a TL)'
##        print '----------------------------------------------------'
##        s1 = Scatter(30,Zbase=30,iwt=-1) ## creates a Scatter object without ports
##        lTL = 2.0     # [m] length of TL
##        ZTL = 50.0    # [Ohm] characteristic impedance
##        s1.trlAZV(["p0","p10"],lTL,A=0.00,V=1,Z=ZTL)
##        s1.addsignal("p0","sp0")
##        s1.addsignal("p10","sp10")
##        print s1
##        print 'Ok now try in two half parts\n'
##        s2 = Scatter(30,Zbase=30,iwt=-1) ## creates a Scatter object without ports
##        s2.trlAZV(["p0","p10"],lTL/2.0,A=0.00,V=1,Z=ZTL)
##        s2.addsignal("p0","sp0")
##        print s2
##        s2.trlAZV("p10",lTL/2.0,A=0.00,V=1,Z=ZTL)
##        s2.addsignal("p10","sp10")
##        print s2
