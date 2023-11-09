__updated__ = '2023-11-07 16:56:18'

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
import time
import warnings

from .config import logident, logit, _newID
from .whoami import whoami
from .printMatrices import strM
from .ConvertGeneral import ConvertGeneral
from . import Z_from_S, Y_from_S
from ._check_3D_shape_ import _check_3D_shape_
        
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
        get1S, getS
        tsf2str, write_tsf
        maxV
    """
    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, *args, **kwargs):
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
        _debug_ and logident('>', printargs=False)
        
        type_rfBase = type(self).__name__ == 'rfBase'
        
        # only save the args and kwargs when creating a rfBase object
        if not hasattr(self,'args'):
            self.args = args
        if not hasattr(self,'kwargs'):
            self.kwargs = kwargs.copy()
        
        self.Id = kwargs.pop('Id', f'{type(self).__name__}_{_newID()}')
        self.Zbase = kwargs.pop('Zbase', 50.)
        
        # self.kwused = ['Id', 'Zbase']
        
        if 'Ss' in kwargs:
            _, self.Ss = _check_3D_shape_(np.array(kwargs.pop('Ss'), dtype=complex)) # Ss is there
            _debug_ and logident(f'self.Ss.shape={self.Ss.shape}')
            _debug_ and logident(f'self.Ss={self.Ss}')
        
            # self.kwused.append('Ss')
            
            if 'ports' in kwargs:
                self.ports = kwargs.pop('ports')
                if len(self.ports) != self.Ss.shape[1]:
                    raise ValueError(
                        'pyRFtk2.rfBase: inconsitent "ports" and "S" given'
                    )
                # self.kwused.append('ports')
            else:
                self.ports = [f'{p_}' for p_ in range(1,self.Ss.shape[1]+1)]
            
        elif 'ports' in kwargs:
            self.ports = kwargs.pop('ports')
            self.Ss = 1j*np.zeros((0, len(self.ports),len(self.ports)))
            # self.kwused.append('ports')
            
        else:
            self.ports = []
            self.Ss = 1j*np.zeros((0, 0, 0))
        
        self.fs = np.array(kwargs.pop('fs', range(self.Ss.shape[0])), dtype=float)
        # self.kwused.append('fs')
        
        _debug_ and logident(f'self.fs={self.fs}')
        if self.Ss.shape[0] != self.fs.shape[0]:
            raise ValueError(
                f'size mismatch between supplied fs [{self.fs.shape[0]}] and '
                f'Ss [{self.Ss.shape[0]}].'
            )
        
        self.xpos = kwargs.pop('xpos', [0.] * len(self))
        # self.kwused.append('xpos')
        
        self.f, self.S = None, None
        
        if kwargs and type_rfBase:
            msg = f'unprocessed kwargs: {", ".join([kw for kw in kwargs])}'
            _debug_ and logident(msg)
            warnings.warn(msg)
        
        self.kwused = [kw for kw in self.kwargs if kw not in kwargs]
        
        # this also initialized the setable attributes of child classed to none
        # by default. The child class will have to set it explicitly if required.
        
        self.attrs = []
        
        _debug_ and logident('<')
    
    #===========================================================================
    #
    # a s s t r
    #
    def asstr(self, full=0):
        return self.__str__(full)
    
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
    
    def __deepcopy__(self, memo=None):                       # @UnusedVariable
        other = type(self)(*self.args, **self.kwargs)
        for attr, val in self.__dict__.items():
            try:
                # print(f'copying {attr}')
                other.__dict__[attr] = copy.deepcopy(val)
            except:
                print(f'{whoami(__package__)}: could not deepcopy {attr}')  # @UndefinedVariable
                raise
        
        # change the Id
        other.Id += '(copy)'
        return other
    
    #===========================================================================
    #
    # s e t 
    #
    def set(self, **kwargs):
        """set attributes of the object if present
        """
        modified = False
        for kw, val in kwargs.items():
            try:
                if kw in self.attrs:
                    if hasattr(self, kw):
                        modified |= getattr(self, kw) != val
                        setattr(self, kw, val)
                    else:
                        raise ValueError(
                            f'{whoami(__package__)}: {self.Id} has no attribute {kw}'  # @UndefinedVariable
                        )
                else:
                    raise AttributeError()
            except AttributeError:
                raise ValueError(
                        f'{whoami(__package__)}: attribute {kw} of {self.Id} '  # @UndefinedVariable
                        'cannot be set'
                    )
        if modified:
            self.f, self.S = None, None
            
        return modified
                
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
                f'{whoami(__package__)}: f={f}Hz is not available'  # @UndefinedVariable
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
    # w r i t e _ t s f
    #
    def tsf2str(self, **kwargs):
        """tsf2str
            return a multiline string which is the touchstone v1 representation 
            of the rfBase object's data.
            
            kwargs:
                fs:    [Hz]  (default self.fs) frequencies to collect in the
                             touchstone
                             
                Zbase: [Ohm] (default self.Zbase) get the S-matrix data on 
                             reference impedance Zbase
                
                tfmt : []    (default f'#MHZ S MA R {self.Zbase}') the touchstone 
                             file format
                
                comments: [] (default self.Id) a multiline comment string added
                             at the top of the file
            
            Note: Zbase is only used when R is not supplied in tfmt to define
                  the reference impedance Zref for the touchstone. If Zref is
                  supplied in the tfmt then it overrules Zbase.
        """
        Zbase = kwargs.pop('Zbase', self.Zbase)
        tfmt = kwargs.pop('tfmt', f'# MHZ S MA R {Zbase}')
        comments = kwargs.pop('comments', f'{self}')
        fs = kwargs.pop('fs', None)
        
        if kwargs:
            raise TypeError(f'{whoami(__package__)}: ' # @UndefinedVariable
                            f'Unknown kwargs {", ".join([kw for kw in kwargs])}') 
                    
        elmfmt, mtype, Zref, fscale, funit = 'MA', 'S', 50.0, 1.0, 'MHZ'
        fscales = {'HZ': 1, 'KHZ':1e3, 'MHZ':1e6, 'GHZ':1e9}
        elstrs = {
            'RI': lambda z: '  %11.8f %11.8f' % (z.real, z.imag),
            'MA': lambda z: '  %15.12f %9.4f' % (np.abs(z), np.angle(z,deg=1)),
            'DB': lambda z: '  %15.10f %9.4f' % (20*np.log10(z).real, 
                                                 np.angle(z,deg=1)),
        }
        
        fmts = tfmt.upper().split()
        if fmts[0] == '#':
            fmts = fmts[1:]
            
        last = ''
        for k, tk in enumerate(fmts):
            if last == 'R':
                pass # skip it as it is the value for R
            
            elif tk == 'R':
                Zref = float(fmts[k+1])
            
            elif tk in fscales:
                funit = tk
                fscale = fscales[funit]
            
            elif tk in ['RI', 'MA', 'DB']:
                elmfmt = tk
                elstr = elstrs[elmfmt]
            
            elif tk in ['S', 'Y', 'Z']:
                mtype = tk
            
            else:
                raise ValueError(f'{whoami(__package__)}: '  # @UndefinedVariable
                                 f'unknown touchstone format directive {tk}')
            last = tk
            

        eoln = '\n'
        
        s = ''
        if isinstance(comments,str):
            comments = comments.split('\n')
        for c in comments:
            s += f'! {c}'+eoln
        s += eoln
        
        s += f'! TouchStone file ({type(self).__name__} version {__updated__})'+eoln
        s += f'! Date : {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}'+eoln
        s += eoln
        s += f'{tfmt}'+eoln
        s += f'!PORTS {", ".join(self.ports)}'+eoln
        s += eoln

        if fs is None:
            fs = self.fs
            Ss = self.Ss
        else:
            Ss = self.getS(fs)
            
        Ms = {'S' : ( lambda S_: ConvertGeneral(Zref, S_, self.Zbase) ),
              'Z' : ( lambda S_: Z_from_S(S_, self.Zbase) ),
              'Y' : ( lambda S_: Y_from_S(S_, self.Zbase) )        
             }[mtype](Ss)

        Fs = np.array(fs) / fscale
        
        freq_fmt = '%9.6f '
        freq_fmt_len = 10
        
        if True:
            # find the optimal frequency format that is able to represent the
            # smallest frequency step as well as the maximun frequency
            
            if len(Fs) > 1:
                for df in list(set(sorted(np.diff(Fs)))):
                    if df > 0.:
                        break
            else:
                df = Fs[0]
            
            m0, dm = 15, 3
            m = m0
            df0 = round(df,m)
            while m >= 0 and df0 == round(df,m):
                m -= dm
            m += dm
            freq_frac = '.%df' % m
            n = len(('%'+freq_frac) % np.max(Fs))
            freq_fmt = ('%%%d' % n) + freq_frac + ' '
            freq_fmt_len = n + 1

        CN = 4
        for fk, Sk in zip(Fs, Ms):
            
            if len(self) == 2: # touchstone convention for 2-ports
                Sk = np.transpose(Sk)
                
            s += freq_fmt % fk
            for kr, Srow in enumerate(Sk):
                for kc, Scol in enumerate(Srow):
                    s += elstr(Scol) + '  '
                    
                    if (kc % CN is (CN-1)) and (kc is not len(self)-1):
                        if kc is (CN-1):
                            s += ' !    row %d' % (kr+1)
                        s += eoln+' '*freq_fmt_len
                s += eoln
                if kr is not len(self)-1:
                    s += ' '*freq_fmt_len
            s += eoln
        return s
    
    #===========================================================================
    #
    # w r i t e _ t s f
    #
    def write_tsf(self, tsbasename, **kwargs):
        
        with open(f'{tsbasename}.s{len(self)}p', 'w') as f:
            f.write(self.tsf2str(**kwargs))
        
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
            raise ValueError(f'{whoami(__package__)}: {msg}')  # @UndefinedVariable

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
