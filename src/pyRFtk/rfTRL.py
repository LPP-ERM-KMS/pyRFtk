__updated__ = "2022-05-16 14:00:49"

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import copy

#===============================================================================
#
# sub pakacges imports
#
from .config import tLogger, logit, ident
from .config import _newID
from .config import rcparams
from .config import fscale

from .ConvertGeneral import ConvertGeneral
from .S_from_VI import S_from_VI
from .maxfun import maxfun
from .ReadTSF import ReadTSF

from .resolveTLparams import TLresolver

from .whoami import whoami

#===============================================================================
#
# r f T R L 
#
class rfTRL():
    """this class defines a TL rf object
    
    :param ID: 
        an identifier (optional) could be used for logging errors and warnings
    
    :param ports:
        (default ['1','2']) names the ports on either side of the TL section
    
    :param L:
        (default 0[m]) total length of the TL section
        
    :param Zbase:
        (default 50[Ohm]) reference impedance of the S-matrix representing the
        TL section
        
    :param dx:
        (default int(72)) dimensional step along the axis of the TL section that
        is used to solve the telegraphist's ODE as well as the requested VI 
        standing waves.
        
        If an int-like is given the intitial step is estimated as a 1/Nth of the
        wavelength along the transmission line.
        
        If a float-like is given it is taken as the dimensional step size. This 
        initial estimate is refined so that an integer number of steps fits the 
        TL section's length. 
    
        if a [floats]-like is given end points are added as necessary and the list
        is used as the node points along the TL. TODO: implement this.
          
    :param odeparams:
        (default {'rtol':1E-12, 'atol':1E-12}) 
        
    :param xpos:
        (default 0[m]) relative position of the first port of the TL section.
        useful to position VI standing waves on figures.
        
    :param sNp: 
        sNp-object (default=None) TODO: implement
        
    
    TL parameters can be given as: 
    
    float-like
         just a constant value along the length of the TL section
         
    [floats]-like
          the TL section is split in len([floats])-1 segments that have a
          piece-wise linear varying value of the paramter (note: continuous).
          A list of 1 long is treated as a float-like
          
    [[floats],[floats]]-like
        the node positions are given in the first [floats] and their values in
        the second [floats]. The node positions do not need to start or end on
        the TL boundaries: the values are linarly interpolated or extrapolated
        to the nearest as the case may be.
              
    In case any of the supplied parameters are not constant the solving for the
    TL's S-matrix and VI standing waves will invoke an ODE solver to solve the
    telegraphist's equation. Otherwise the (constant) complex characteristic
    impedance and propagation constant is (derived form the supplied parameters
    and) used to explicitly solve the telegraphist's equation.
    
    Global TL properties:
    
        Z0TL: [Ohm] characteristic impedance of the TL
    
        LTL: [H/m] line inductance of the TL
        CTL: [F/m] line capacitance of the TL 
    
        rTL: [Ohm/m] line resistance of the TL
           rTLO: [[Ohm/m] line resistance of the outer conductor of the TL
           rTLI: [Ohm/m] line resistance of the inner conductor of the TL
           
        gTL: [Mho/m] medium conductance of the TL

        A: [1/m] attenuation constant
        AdB: [dB/m] attenuation constant in dB/m
        
    Conductor properties:
    
        rho: [Ohm.m] specific resistivity of the conductors
           rhoO: [Ohm.m] specific resistivity of the outer conductor
           rhoI: [Ohm.m] specific resistivity of the inner conductor
    
        murO: [mu0] outer conductor's permeability
        murI: [mu0] inner conductor's permeability
        
    Medium properies:
    
        sigma: [Mho/m] specific conductivity of the medium
        tand: [1] medium dielectric loss tangent
    
        epsr: [epsilon0] medium's relative permitivity
        mur: [mu0] medium's relative permeability
        
        etar: [eta0] medium's relative impedance
        
        vr: [c0] medium's relative wave velocity 
    
    Geometric properties:
    
        OD: [m] outer diameter of a circular coaxial TL
        ID: [m] inner diameter of a circular coaxial TL
        
        qTL: [1/m] rTL = (w rho mur mu0 / 2)**0.5 * qTL
        qTLO: [1/m] rTLO= (w rhoO murO mu0 / 2)**0.5 * qTLO
        qTLI: [1/m] rTLI = (w rhoI murI mu0 / 2)**0.5 * qTLI
        
    
    Implemented methods:
    
    __init__(**kwargs):
        performs the initial build of the TL section's properties
    
    __str__:
        returns a nicely formated string describing the defined TL section
    
    __len__:
        (mandatory for RF objects) returns the number of ports
    
    set(**kwargs):
        (mandatory for RF objects) modifies some of the properties of the TL 
        section
        
        currently implemented:
        
            L for a constant TL section
            TODO: expand on the parameters and remove constant condition
            
    solveVI(f->float, [V0 -> np.complex, I0-> np.complex]): -> np.array
        solves the telegraphist's equation either explicitly for constant TL 
        sections or through an ODE solver otherwise.
        returns depending on the input parameters (see method)
    
    getS(fs, Zbase=None, params={})
        returns the S-matrix/ces for the requested frequency/ies [Hz] for the
        reference impedance Zbase (or the object's reference impedance if not
        supplied). Prior to the evaluation the params-dict is passed to the
        object's set method as **params.
        
    VISWs(f, E, Zbase=None)
        returns xs, (
        
    maxV(f, E, Zbase=None)
    
    """
    def __init__(self, *args, **kwargs):

        if not hasattr(self,'args'):
            self.args = args
            
        if not hasattr(self,'kwargs'):
            self.kwargs = kwargs.copy()
            
        self.Id = kwargs.pop('Id','rfTRL_'+_newID())
        self.ports = kwargs.pop('ports',['1','2'])
        self.L = kwargs.pop('L', 0.)
        self.Zbase = kwargs.pop('Zbase', 50.)
        self.dx = kwargs.pop('dx', 72) # int -> 1/int-th of wavelength 
        self.odeparams = kwargs.pop('odeparams',{'rtol':1E-12, 'atol':1E-12})
        self.sNp = kwargs.pop('sNp', None)
        self.constant = True
        self.f = np.NaN
        self.S = None
        self.attrs = ['L']
        
        xpos = kwargs.pop('xpos', [0., self.L])
                        
        if isinstance(xpos, list):
            if len(xpos) == 2:
                self.xpos = xpos
            else:
                raise ValueError(
                    'rfTRL.__init__: port count mismatch'
                )
        
        elif isinstance(xpos, (float,int)):
            self.xpos = [xpos, self.L + xpos]
        
        #------------------------------------------------------------ touchstone
        # FIXME: do we need this here ??
        if self.sNp:
            self.sNpdata = ReadTSF(src = self.sNp, 
                                   funit='Hz',
                                   Zbase=self.Zbase, 
                                   ports=self.ports)
 
            # prepare the interpolation functions
            
            fs = self.sNpdata['fs'] # fs already in Hz
            Ss = self.sNpdata['Ss']
            fmt = self.sNpdata['datafmt'].upper()
            
            d = {'MA': (lambda z: [np.abs(z), np.unwrap(np.angle(z))]),
                 'DB': (lambda z: [20*np.log10(z).real, np.unwrap(np.angle(z))]),
                 'RI': (lambda z: [z.real, z.imag])
                }[fmt](Ss[:])
                        
            f1 = lambda f: interp1d(fs, d[0], axis=0, kind=min(len(fs)-1,3),
                                    bounds_error=False, 
                                    fill_value='extrapolate')(f)
                                    
            f2 = lambda f: interp1d(fs, d[1], axis=0, kind=min(len(fs)-1,3),
                                    bounds_error=False, 
                                    fill_value='extrapolate')(f)
            
            self.sNpinter = {
                'MA': (lambda f: f1(f) * np.exp(1j*f2(f))),
                'DB': (lambda f: 10**(f1(f)/20) * np.exp(1j*f2(f))),
                'RI': (lambda f: f1(f) + 1j * f2(f)),
            }[fmt]     
                
            # print(fs, np.max(np.max(np.abs(Ss - self.sNpinter(fk)))))
            # print(Ss,'\n')
            
        #-------------------------------------------------------- TRL parameters
        self.TLP = TLresolver(L=self.L, Zbase=self.Zbase, **kwargs)
        
        self.rjwL = self.TLP.rjwL
        self.gjwC = self.TLP.gjwC
        
        self.constant = not any([
            self.TLP.kwset[_]['ddx'] for _ in ['rjwL','gjwC']
        ])
        
        self.solved = {}
        
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
        debug and tLogger.debug(ident(f'> circuit.__deepcopy__ {self.Id}', 1))
        
        other = type(self)(*self.args, **self.kwargs)
        for attr, val in self.__dict__.items():
            try: 
                debug and tLogger.debug(ident(f'copy {attr}'))
                other.__dict__[attr] = copy.deepcopy(val)
            except:
                print(f'{whoami(__package__)}: could not deepcopy {attr}')  # @UndefinedVariable
                raise
        
        debug and tLogger.debug(ident(f'< circuit.__deepcopy__ {self.Id}', -1))
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
    def __str__(self, full=False):
        
        s = f'{self.Id}: rfTRL at {id(self)} (version {__updated__}) \n'
        s += f'| Zbase: {self.Zbase} Ohm \n'
        s += f'| ports: {self.ports} \n'
        s += f'| length: {self.L:.3f}m; '
        if isinstance(self.dx, int):
            s += f'dx: {self.dx} pts per wavelength \n'
        else:
            s += f'dx: {self.dx:.3f}m \n'

        if self.constant:
            s += f'| constant TL parameters \n'
        else:
            s += f'| ODE integration parameters: {self.odeparams} \n'
        
        self.TLP.f0 = self.f if self.f != np.NaN else None
        for _s in self.TLP.__str__(full).split('\n'):
            s += f'| {_s}\n'
    
        if self.S is not None:
            s += f'|\n+ last evaluation at {self.f:_.1f} Hz: \n'
            for sk in str(self.S).split('\n'): 
                s += '|   ' + sk + '\n'
        s += '^' 
        return s 
    
    #===========================================================================
    #
    # _ _ l e n _ _
    #        
    def __len__(self):
        return 2
    
    #===========================================================================
    #
    # s e t
    #
    def set(self, **kwargs):
        
        if kwargs and not self.constant:
            raise ValueError(
                f'{whoami(__package__)}: not possible because the transmission '  # @UndefinedVariable
                             'line parameters are not constant')
        modified = False
        for kw, val in kwargs.items():
            if not hasattr(self, kw):
                raise ValueError(
                    f'{whoami(__package__)}: parameter {kw} not present')  # @UndefinedVariable
                
            modified |= getattr(self,kw) != val
            setattr(self, kw, val)
            self.kwargs[kw] = val
            if kw == 'L':
                self.xpos = [self.xpos[0], self.L + self.xpos[0]]
        
        self.solved = {}
        
        return modified
    
    #===========================================================================
    #
    # s o l v e V I
    #
    def solveVI(self, f, V0 = None, I0 = None):
        """
        Solves the telegraphist's equation 

        The telegraphist's equation is solved for frequency f[Hz] either explicitly
        for a TL section with constant TL parameters or using scipy.integrate.odeint
        otherwise. 
        
        The node points along the TL are defined through the atribute 'dx'.
        
        TODO: for an int-like 'dx' and a non-constant TL section the estimation
              of the node points is not correctly implemented. 
        
        Note: in the telegrapher's equation the currents are flowing out of the
              'sink' port (V0,I0) and into the 'source' port (would be 
              V1=self.U[-1], I1= self.I[-1].
        """
        
        w = 2*np.pi*f 
        
        #=======================================================================
        #
        # e q n V I
        #
        def eqnVI(VI4, x):
            """
            VI4 = [V.real, V.imag, I.real, I.imag]
            """
            # jwLr, jwCg = self.TLprops(x,verbose=verbose)
            
            # print(f'eqnVI x={x}')
            
            dVC = self.rjwL(w, x) * (VI4[2]+1j*VI4[3])
            dIC = self.gjwC(w, x) * (VI4[0]+1j*VI4[1])
        
            return [dVC.real, dVC.imag, dIC.real, dIC.imag]
        
        #=======================================================================
        #
        # w a v l e n
        #
        def wavlen(w,x):
            """
            returns the local wavelength along the TL
            
            used to determine the node points for the ODE solver
            """
            
            gm = (self.rjwL(w,x) * self.gjwC(w, x))**0.5
            try:
                return 2 * np.pi / np.imag(gm)
            except ZeroDivisionError:
                return +np.inf

        #=======================================================================
        
        if f not in self.solved:
            
            if isinstance(self.dx, int): # step = fraction of wavelength
                
                if self.constant:

                    num = int(np.ceil(self.L / (wavlen(w,0)/self.dx)))
                    xs = np.linspace(0, self.L, max(2,num))
            
                else: # non constant TL parameters

                    xs = [0.]
                    while xs[-1] < self.L:
                        xs.append(xs[-1] + wavlen(f,xs[-1])/self.dx)
                    
                    # may have overshot the end point so we rescale everything
                    xs = np.array(xs) * self.L/xs[-1]
                    
            else: # step explicitly given
                if self.dx < 0:
                    x, xs = 0., []
                    while x <= self.L:
                        xs.append(x)
                        x -= self.dx
                    if xs[-1] < self.L:
                        xs.append(self.L)
                    xs = np.array(xs)
                    
                else:
                    num = int(np.ceil(self.L/self.dx))
                    xs = np.linspace(0, self.L, max(2,num))
                                       
            if xs[0] == xs[-1]:
                
                # special case of a 0-length TL section
                
                xs = np.array([xs[0],xs[-1]])
                U10 = np.ones(np.array(xs).shape, dtype=complex)
                I10 = np.zeros(np.array(xs).shape, dtype=complex)
                U01 = np.zeros(np.array(xs).shape, dtype=complex)
                I01 = np.ones(np.array(xs).shape, dtype=complex)
                
            elif self.constant:
                
                # all TL parametersare constant with respect to x
                # so one does not have to solve the differential eqn
                
                gm = (self.rjwL(w,0) * self.gjwC(w,0)) ** 0.5
                Zc = self.rjwL(w,0) / gm
                
                egmx = np.exp(gm * xs)
                egmx1 = 1/egmx
                
                U10 = 0.5 * (egmx + egmx1)
                I10 = 0.5 * (egmx - egmx1) / Zc
                U01 = Zc **2 * I10
                I01 = U10
                
            else:
                
                sol10 = odeint(eqnVI, [1.,0.,0.,0.], xs, **self.odeparams)
                U10 = sol10[:,0]+1j*sol10[:,1]
                I10 = sol10[:,2]+1j*sol10[:,3]
                
                sol01 = odeint(eqnVI, [0.,0.,1.,0.], xs, **self.odeparams)
                U01 = sol01[:,0]+1j*sol01[:,1]
                I01 = sol01[:,2]+1j*sol01[:,3]
            
            VIt = np.array([[U10, U01],
                            [I10, I01]])
    
            S = S_from_VI(VIt[:,:,-1], self.Zbase)
            
            self.solved[f] = {
                'xs'    : xs,
                'S'     : S,
                'VI'    : VIt,
                'VISWs' : {}
            }
            
            # pp.pprint(self.solved[f])
            
        if I0 is None and V0 is None:
            
            return self.solved[f]['S']
            
        else:
            if I0 is None:
                I0 = 0.
                
            if V0 is None:
                V0 = 0.
                
            UIS = self.solved[f]
            VI1s = np.tensordot(UIS['VI'], np.array([V0,I0]), axes=([1],[0]))
            
            return self.solved[f]['xs'], VI1s[0], VI1s[1]
            
    #===========================================================================
    #
    # g e t S
    #
    def getS(self, fs, Zbase=None, params={}, **kw):
        """
        returns an (array of) S-matrice(s) at the frequency(ies)fs 

        Returns a single S-matrix for a float-like frequency fs [Hz] or a 1D
        array of S-matrices for a [floats]-like fs.
        
        The returned S-matrice(s) is(are) defined at the object's reference 
        impedance or at the supplied reference impedance Zbase.
        
        The object's set function self.set(**kwargs) is called with **params
        prior to the evaluation.
        """
        
        def get1S(f):
            
            if self.sNp:
                M = self.sNpinter(f)
            else:
                M = self.solveVI(f)
            
            self.f, self.S = f, M 
            
            if Zbase and Zbase != self.Zbase:
                M = ConvertGeneral(Zbase,M,self.Zbase)
            
            return M
        
        self.set(**params)
        
        Ss = []
        if hasattr(fs,'__iter__'):
            for f in fs:
                Ss.append(get1S(f))
            Ss = np.array(Ss)
        else:
            Ss = get1S(fs)
            
        return Ss
    
    #===========================================================================
    #
    # V I S W s  
    #
    def VISWs(self, f, E, Zbase=None):
        # note E = {portname: complex value, ... }

        As = np.array([E[p] for p in self.ports])
        Ahash = tuple(As.tolist())
        if f not in self.solved or (Ahash, Zbase) not in self.solved[f]['VISWs']:
        
            Zbase = Zbase if Zbase else self.Zbase
            Bs = self.getS(f, Zbase) @ As
            Vs = As + Bs
            Is = -(As - Bs) / Zbase
            
            _, Vsw, Isw = self.solveVI(f, Vs[0], Is[0])
            
            self.solved[f]['VISWs'][(Ahash, Zbase)] = (Vsw, Isw, E)
            
        return self.solved[f]['xs'], self.solved[f]['VISWs'][(Ahash, Zbase)]
    
    #===========================================================================
    #
    # m a x V  
    #
    def maxV(self, f, E, Zbase=None, ID='<top>',**kwargs):
        # kwargs: catchall for other parameters
        xs, (Vsw, _, _) = self.VISWs(f, E, Zbase)
        Usw = np.abs(Vsw)
        xymax = maxfun(xs, Usw)
        return xymax[1], f' node: {self.ports[0]} @ {xymax[0]:.3f}m', (xs, Usw)
        

