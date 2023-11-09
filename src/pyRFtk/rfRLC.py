__updated__ = "2023-09-21 15:29:27"

import numpy as np
import copy

#TODO: integrate config imports ...

if __name__ == '__main__':
    from config import _newID, logit, logident # @UnresolvedImport @UnusedImport
    from whoami import whoami               # @UnresolvedImport @UnusedImport
    
else:
    from .config import _newID, logit, logident                      # @Reimport
    from .whoami import whoami                                    # @Reimport


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
    def __init__(self, *args, **kwargs):
        """kwargs:
            Zbase : reference impedance [50 Ohm]
            ports : port names [['s','p']]
            Rp : parallel resistance [+inf Ohm]
            Lp : parallel inductance [+inf H]
            Cp : parallel capacity [0 F]
            Rs : series resistance [0 Ohm]
            Ls : series inductance [0 H]
            Cs : series capacity [+inf F]
            xpos : list of the port positions (order 's', 'p')
        """
            
        if not hasattr(self,'args'):
            self.args = args
            
        if not hasattr(self,'kwargs'):
            self.kwargs = kwargs.copy()
        
        self.Id = kwargs.pop('Id', f'{type(self).__name__}_{_newID()}')
        self.Zbase = kwargs.pop('Zbase', 50.)
        self.ports = kwargs.pop('ports',['s','p'])
        self.Rs = kwargs.pop('Rs', 0.)
        self.Ls = kwargs.pop('Ls', 0.)
        self.Cs = kwargs.pop('Cs', +np.inf)
        self.Rp = kwargs.pop('Rp', +np.inf)
        self.Lp = kwargs.pop('Lp', +np.inf)
        self.Cp = kwargs.pop('Cp', 0.)
        self.f = None
        self.S = np.array([[0,1],[1,0]], dtype=complex)
        self.xpos = kwargs.pop('xpos',[0., 0.])
        self.attrs = ['Rs', 'Ls', 'Cs', 'Rp', 'Lp', 'Cp']
    
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
    def __deepcopy__(self, memo=None):
        
        debug = logit['DEBUG']
        debug and logident('>')
        other = type(self)(*self.args, **self.kwargs)
        for attr, val in self.__dict__.items():
            try:
                other.__dict__[attr] = copy.deepcopy(val)
            except:
                msg = f'{whoami(__package__)}: could not deepcopy {attr}'  # @UndefinedVariable
                debug and logident(msg)
                raise RuntimeError(msg)
            
        debug and logident('<')
        return other

    #===========================================================================
    #
    # a s  s t r
    #
    def asstr(self,full=False):
        return self.__str__(full)
        
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, full=0):
        s  = f'{type(self).__name__} Id={self.Id} at {hex(id(self))}\n'
        if full:
            for elem, value, default, unit, scale in zip(
                ['Rs', 'Ls', 'Cs', 'Rp', 'Lp', 'Cp'],
                [self.Rs, self.Ls, self.Cs, self.Rp, self.Lp, self.Cp],
                [0., 0., +np.inf, +np.inf, +np.inf, 0.],
                ['Ohm', 'nH', 'pF','Ohm', 'nH', 'pF'],
                [1e0, 1e-9, 1e-12, 1e0, 1e-9, 1e-12]):
                if value != default:
                    s += f'|  {elem} = {value/scale:.6g} {unit} \n'
        s += '^\n'
        return s
    
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
        
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', printargs=True)

        modified = False
        for kw, val in kwargs.items():
            if not hasattr(self, kw) or kw not in self.attrs:
                raise ValueError(f'rfTRL.set: parameter {kw} not present')
        
            modified |= getattr(self, kw) != val
            setattr(self, kw, val)
            self.kwargs[kw] = val
        
        if modified:
            self.f, self.S = None, None
            
        _debug_ and logident('<')
            
        return modified
    
    #===========================================================================
    #
    # g e t S
    #
    def getS(self, fs, Zbase=None, params={}, **kw):
        
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
    def maxV(self, f, E, Zbase=None, ID=None, **kwargs):
        
        # kwargs: catch all for other parameters 
        ID =  self.Id if ID is None else ID
        Zbase = self.Zbase if Zbase == None else Zbase
        
        Ea = [E[p] for p in self.ports]
        Eb = self.getS(f, Zbase) @ Ea   # implicitly updates and solves the circuit
        Vi = np.abs(Ea + Eb)
        maxVi = np.max(Vi)
        k = np.where(Vi == maxVi)[0][0]
        
        return maxVi, ID+'.'+self.ports[k], (np.zeros(Vi.shape),Vi)

        return 

#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    from config import setLogLevel             # @UnresolvedImport @UnusedImport
    tRLC = rfRLC(Rs=10)
    print(type(tRLC))
    print(tRLC)
    
    setLogLevel('DEBUG')
    
    def f1(t):
        def f2(t):
            def f3(t):
                logident('> hello world')
                r = t.copy()
                logident('<')
                return r
            
            logident('>')
            r = f3(t)
            logident('<')
            return r
        
        logident('>')
        r = f2(t)
        logident('<')
        return r
    
    rRLC = f1(tRLC)
