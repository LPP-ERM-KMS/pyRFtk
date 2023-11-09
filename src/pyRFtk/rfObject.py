__updated__ = "2023-11-08 11:19:50"

if __name__ == '__main__':
    import sys
    sys.path.append('../pyRFtk2 test')
    print('importing test_rfObject...')
    import test_rfObject                                     # @UnresolvedImport


#FIXME: import config ?

import numpy as np
from scipy.interpolate import interp1d
import copy

from .ReadTSF import ReadTSF
from .ConvertGeneral import ConvertGeneral
from .S_from_Y import S_from_Y
from .S_from_Z import S_from_Z
from ._check_3D_shape_ import _check_3D_shape_

from .config import tLogger, logit, ident
from .config import _newID
from .config import rcparams
from .config import fscale
from .config import FUNITS

from .rfTRL import rfTRL
from .whoami import whoami

#===============================================================================
#
#  r f O b j e c t
#
class rfObject():
    """rfObject
    
    The purpose of rfObject is to create a frequency vs. S-matrix "glorified" LUT
    
    The creation of the rfObject can be done in several manners:
    
    1) the simplest is to load a touchstone file e.g. produced by Ansys/HFSS and
       the likes
       
       myObject = rfObject(touchstone=<path_to_touchstone>)
        
    2) another method is the fill the table manually
    
       portnames = <list of ports>
       myObject = rfObject(ports = portnames]
       
       for fHz in fHzs:
           S = <calculate S>
           myObject.setS(fHz, S)
           
    Once an rfObject is created the 2 main methods of use are:
    
    rfObject.getS(fHz, Zbase=None)
    
    and 
    
    excitation = {port1:a1, ... }
    Vamx, rfObject.maxV(fHz, excitation)
    
   
    """
    #===========================================================================
    #
    #  _ _ i n i t _ _
    #
    def __init__(self, *args, **kwargs):
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
            
            fixme: no parameters to set the Zcs and Gms port properties
        """
        
        if not hasattr(self,'args'):
            self.args = args
        
        if not hasattr(self,'kwargs'):
            self.kwargs = kwargs.copy()
        
        self.Id = kwargs.pop('Id',f'{type(self).__name__}_{_newID()}')
        
        #TODO: validate allowed port names
        self.ports = kwargs.get('ports',[])
        self.alias = {}
        if isinstance(self.ports, str):
            self.ports = self.ports.split()
        self.N = len(self.ports)
        
        self.Zbase = kwargs.get('Zbase', rcparams['Zbase'])
        self.funit = kwargs.get('funit', rcparams['funit'])
        self.fscale = FUNITS[self.funit.upper()]
        self.flim = kwargs.get('flim', [0.001, 1000.])
        
        self.interp = kwargs.get('interp',rcparams['interp'])
        self.interpkws = kwargs.get('interpkws',rcparams['interpkws'])
        
        self.xpos = kwargs.get('xpos', [0.]*self.N)
        
        # initialize arrays and stuff
        self.fs = np.array([],dtype=float).reshape((0,))
        self.sorted = True
        self.Ss = np.array([], dtype=complex).reshape((0, self.N, self.N))
        self.Zcs = np.array([], dtype=complex).reshape(0, self.N)
        self.Gms = np.array([], dtype=complex).reshape(0, self.N)
        self.process_kwargs()
        # self.type = TYPE_GENERIC        
        
        self.touchstone = kwargs.get('touchstone', None)
        if self.touchstone:
            Tkwargs = {'Zbase': self.Zbase, 'funit':'HZ'}
            for (kw, value) in kwargs.items():
                if kw in Tkwargs:
                    Tkwargs[kw] = value
            
            self.read_tsf(self.touchstone, **Tkwargs)
        
        self.interpOK = False
        
        self.f, self.S = None, None
                
    #===========================================================================
    #
    # c o p y
    #
    def copy(self):
        
        return self.__deepcopy__(self)
    
    #===========================================================================
    #
    # _ _ c o p y _ _
    #
    def __copy__(self):
        
        return self.__deepcopy__(self)
    
    #===========================================================================
    #
    # _ _ d e e p c o p y _ _
    #
    def __deepcopy__(self, memo={}):
        
        other = type(self)()
        for attr, val in self.__dict__.items():
            try:
                # print(f'copying {attr}')
                other.__dict__[attr] = copy.deepcopy(val)
            except:
                print(f'{whoami(__package__)}: could not deepcopy {attr}')   # @UndefinedVariable
                raise
        
        return other
     
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
        d['Zcs'] = self.Zcs.tolist()
        d['Gms'] = self.Gms.tolist()
        
        return d
    
    #===========================================================================
    #
    #  _ _ s e t s t a t e _ _
    #
    def __setstate__(self, d):
        
        for var, value in d.items():
            if var not in ['fs', 'Ss','Zcs','Gms']:
                self.__dict__[var] = value
        
        # get the arrays back
        self.fs = np.array(d['fs'])
        self.Ss = np.array(d['Ss'])
        self.Zcs = np.array(d['Zcs'])
        self.Gms = np.array(d['Gms'])

    #===========================================================================
    #
    #  _ _ r e p r _ _
    #
    def __repr__(self):
        s = '%s( # \n' % (type(self).__name__)
        s += '  Portnames=[   # %d \n' % len(self.ports)
        for p in self.ports:
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
    def __str__(self, full=False):
        #FIXME: adhere to format compatible with rfTRL, rfRLC, circuit, ...
        s =  self.__repr__() + '\n'
        s += '# %d frequencies\n' % len(self.fs)
        
        if self.S is not None:
            s += f'|\n+ last evaluation at {self.f:_.1f} Hz: \n'
            for sk in str(self.S).split('\n'): 
                s += '|   ' + sk + '\n'
                
        return s
    
    #===========================================================================
    #
    #  _ _ l e n _ _
    #
    def __len__(self):
        return len(self.ports)
    
    #===========================================================================
    #
    #  _ _ i t e r _ _
    #
    def __iter__(self):
        for k in range(len(self.fs)):
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
    #  c o p y
    #
    def _copy_(self):
        
        new_self = type(self)(**self.kwargs)
        new_self.sorted = self.sorted
        new_self.Portnames = self.Portnames[:]
        new_self.fs = self.fs.copy()
        new_self.Ss = self.Ss.copy()
        new_self.interpOK = False    # avoid to copy interpolation functions
        new_self.type = self.type
        new_self.ports = self.ports[:]
        new_self.alias = self.alias.copy()
        new_self.N = self.N
        new_self.Zbase = self.Zbase
        new_self.funit = self.funit
        new_self.fscale = self.fscale
        new_self.flim = self.flim.copy()
        new_self.Zcs = self.Zcs.copy()
        new_self.Gms = self.Gms.copy()

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
            idxs = [self.ports.index(p1) for p1 in order]
            idxs += [k for k, p1 in enumerate(self.ports) if k not in idxs]
            if len(idxs) != len(self.ports):
                raise ValueError(
                    'rfObject.sort: either a bug or duplicate ports in order list')
        else:
            idxs = np.array(self.ports).argsort()
        
        self.ports = [self.ports[k] for k in idxs]
        self.Ss = self.Ss[:,:,idxs][:,idxs,:]
        self.Zcs = self.Zcs[:,idxs]
        self.Gms = self.Gms[:,idxs]
    
    #===========================================================================
    #
    #  _ s o r t _ f s _
    #
    def _sort_fs_(self):
        
        if not self.sorted:
            
            idxs = self.fs.argsort()
            self.fs = self.fs[idxs]
            self.Ss = self.Ss[idxs,:,:]
                        
            self.Zcs = self.Zcs[idxs,:]
            self.Gms = self.Gms[idxs,:]
            
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
            
            interpkws = self.interpkws.copy()
            interpkws.update({'kind':min(len(self.fs)-1, 
                                         interpkws.get('kind',3))})
            if interpkws['kind'] == 0:
                interpkws.update({'fill_value':'extrapolate'})
            
            
            self.interpM = interp1d(self.fs, M, axis=0, **interpkws)
            self.interpA = interp1d(self.fs, A, axis=0, **interpkws)
                            
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
    def convert2newZbase(self, newZbase):
        """convert_Zbase
        
            converts the rfObject's Ss and Zbase to newZbase
            
            newZbase: target reference impedance 
        """
        
        self.Ss = ConvertGeneral(newZbase, self.Ss, self.Zbase, 'V', 'V')
        self.Zbase = newZbase
    
    #===========================================================================
    #
    #  g e t S
    #
    def getS(self, fs, **kw):
        
        debug = logit['DEBUG']
        debug and tLogger.debug(ident(
            f'> [rfObject.getS] '
            f'fs = {fs}, kw = {kw} [sNp={self.touchstone}]',
            1
        ))
        
        fs = np.array(fs) * FUNITS[kw.pop('funit', self.funit).upper()]/self.fscale
        Zbase = kw.pop('Zbase',self.Zbase)
        
        def get1S(fk):
            k = np.where(self.fs == fk)[0]
            if k.size:
                return self.Ss[k[0]]
            else:
                # print('--interpolating--')
                return self._interp_(fk)[0]
        
        if hasattr(fs,'__iter__'):
            Ss = []
            for f in fs:
                tS = get1S(f)
                Ss.append(
                    tS if Zbase == self.Zbase
                    else ConvertGeneral(Zbase, tS, self.Zbase)
                )
            self.f, self.S = f, Ss[-1]
            
        else:
            tS = get1S(fs)
            Ss = (tS if Zbase == self.Zbase 
                  else ConvertGeneral(Zbase, tS, self.Zbase))
        
            self.f, self.S = fs, Ss
            
        debug and tLogger.debug(ident('< [rfObject.getS]',-1))
        return np.array(Ss)
    
    #===========================================================================
    #
    #  s e t S
    #
    def setS(self, f, S, Portnames=None):
        
        whoami = 'pyRFtk2.rfObject.setS'
        
        self.interpOK = False
        self.interpA = None
        self.interpM = None
        
        try:
            *_, S =  _check_3D_shape_(S)
        except ValueError:
            raise ValueError(f'{whoami}: Unexpected shape: {S.shape}')
        
        f =  np.array(f) if hasattr(f, '__iter__') else np.array([f])
                    
        if len(f) != len(S):
            raise ValueError(
                f'{whoami}: number of frequencies ({len(f)}) does match number of '
                f'S-matrices supplied ({len(S)}.')
        
        if Portnames == None:
            if not self.ports:
                # no port names were defined yet: implicitely set to default names
                self.ports = ['%02d' % k for k in range(S.shape[-1])]
        else:
            if self.ports:
                raise ValueError(f'{whoami}: cannot redefine the portnames')
            else:
                self.ports = Portnames
                
        # self.ports is defined 
        if len(self.ports) != S.shape[-1]:
            raise ValueError(
                f'{whoami}: number of object\'s Portnames ({len(Portnames)}) '
                f'does not match shape of S ({len(S.shape[-1])})')
        
        if len(self.fs) == 0:
            # the object has no frequency entries yet ...
            self.Ss = self.Ss.reshape((0,)+S.shape[-2:])
            
        if self.Ss.shape[-2:] != S.shape[-2:]:
            # we should never come here
            raise SystemError(f'{whoami}: Shape of S does not match ')
                
        self.fs = np.append(self.fs,[f])
        self.Ss = np.append(self.Ss, S, axis=0)
        
        # do not sort each time
        self.sorted = self.sorted and (
            (len(self.fs) == 1) or (self.fs[-2] < self.fs[-1])
        )
        
    #===========================================================================
    #
    #  m a x V
    #
    def maxV(self, f, E, Zbase=None, ID='<rfObject>', **kwargs):
        
        # kwargs : catchall for other parameters
        
        Zbase = self.Zbase if Zbase == None else Zbase
        Ea = [E[p] for p in self.ports]
        Eb = self.getS(f, Zbase=Zbase) @ Ea   # implicitly updates and solves the circuit
        Vi = Ea + Eb
        maxVi = np.max(Vi)
        k = np.where(Vi == maxVi)[0][0]
        
        return maxVi, ID+'.'+self.ports[k], ([0],[maxVi])
    
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
        r = ReadTSF(src, **kwargs)
        self.fs = r['fs']   # * (FUNITS[r['funit'].upper()] 
                            #  / FUNITS[self.funit.upper()])
        self.Ss = r['Ss']
        self.ports = r['ports'] if not self.ports else self.ports
        self.Zbase = r['Zbase']
        self.part = r['part']
        self.numbering = r['numbering']
        self.markers = r['markers']
        self.shape = r['shape']
        self.Zcs = np.array(r['Zc'])
        self.Gms = np.array(r['Gm'])
        self.xpos = [0.] * len(self)
        
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
    
    #===========================================================================
    #
    # d e e m b e d
    #
    def deembed(self, ports):
        """
        deembeds the ports with a simple lossless TL of length L and characteristic
        impedance Z
        
        ports is a dict containing the ports to be deembeded: the value is either 
        a scalar (int, float) giving L assuming the default Z being the object's
        Zbase or a 2-tuple (L, Z) or a dict which is passed as kwargs to rfTRL.
        
        Positive L means the deembeding is into the port 
        
        """
          
        self.interpOK = False      
        
        for kp, port in enumerate(self.ports):
            if port in ports:
                v = ports.pop(port, (0., self.Zbase))
                if isinstance(v,tuple) and len(v) == 2:
                    L, Z = v
                    TRL = rfTRL(L=-L, Z0TL=Z, Zbase=self.Zbase)
                elif isinstance(v, [int, float]):
                    L, Z = v, self.Zbase
                    TRL = rfTRL(L=-L, Z0TL=Z, Zbase=self.Zbase)
                elif isinstance(v, dict):
                    v.pop('Zbase') # we need it in self.Zbase
                    TRL = rfTRL(**v, Zbase=self.Zbase)
                else:
                    raise ValueError(
                        f'pyRFtk2.rfObject.deembed: invalid deembed data for port {port}')
    
                
                for kS, (f, Sobj) in enumerate(zip(self.fs, self.Ss)):
                    Strl = TRL.getS(f)
                                        
                    qn = 1 - Strl[0,0] * Sobj[kp,kp]
                    if np.abs(qn) != 0:
                        # ak_old == Strl[0,0] * bk_old + Strl[0,1] * ak_new
                        # bk_old == sum(j!=k, Skj * aj_old + Skk*ak_old
                        # => ak_old = function({aj (j!=k), ak_new})
                        # => bk_old = function({aj (j!=k), ak_new})
                        # bk_new = Strl[1,0] * bk_old + Strl[1,1] * ak_new
                        # => bk_new = function({aj (j!=k), ak_new})
                        
                        bk_new = Strl[1,0]/qn *  Sobj[[kp],:]
                        bk_new[0,kp] = bk_new[0,kp] * Strl[0,1] + Strl[1,1]
                        
                        q11 = Strl[0,0] / qn * Sobj[[kp],:]
                        q12 = Strl[0,1] / qn * Sobj[:,[kp]]
                        Sobj = Sobj + Sobj[:,[kp]] @ q11
                        Sobj[:,[kp]] = q12
                        Sobj[[kp],:] = bk_new[0,:]
                                        
                    else:
                        # all coeffs have magnitude < 1, therefore for qn == 0
                        # this means that Strl[0,0] as well as Sobj[kp,kp] have
                        # a magnitude of 1. Therefore port kp is an isolated port
                        # it does not depend on any other port.
                        # Furthermore Strl[0,0] is not a good deembedder !! as
                        # Strl[0,1] == 0 and this isolates the port being
                        # deembedded
                        raise RuntimeError(
                            'pyRFtk2.rfObject.deembed: this should not happen !')
                
                    self.Ss[kS,:,:] = Sobj[:,:]

        if ports:
            raise ValueError(
                f'pyRFtk2.rfObject.deembed: port(s) {",".join([p for p in ports])}'
                ' not found.')

