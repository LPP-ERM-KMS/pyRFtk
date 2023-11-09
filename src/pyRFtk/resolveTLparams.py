__updated__ = "2023-11-09 11:11:56"

from inspect import signature

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import speed_of_light as c0, mu_0 as mu0, epsilon_0 as e0

from .whoami import whoami

#===============================================================================

eta0 = np.sqrt(mu0/e0)
eta0coax = eta0/(2*np.pi)

#===============================================================================

class TLresolver():
    
    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, **kwargs):
        
        self.f0 = kwargs.pop('f0', 1e6) # print results at f0 
        self.forget = {}
        for depends, kw in kwargs.pop('forget', {}).items():
            self.forget[tuple(sorted(depends + (kw,)))] = kw
            
        self.L = kwargs.pop('L', None)
            
        self.trl_kwargs = {
            'OD': {}, 'ID': {},
            'rho': {}, 'rhoO': {}, 'rhoI': {},
            'Rs': {}, 'RsO': {}, 'RsI':{},
            'rTL': {}, 'rTLO': {}, 'rTLI': {},
            'qTL': {}, 'qTLI': {}, 'qTLO': {},
            'murI': {}, 'murO': {}, 'A': {}, 'AdB': {},
            'gTL': {}, 'sigma': {}, 'tand': {},
            'LTL': {}, 'CTL': {}, 'Z0TL': {},
            'epsr': {}, 'mur': {}, 'vr': {}, 'etar': {},
        }
        
        self.Zbase = kwargs.pop('Zbase', 50.)
        
        # TODO:
        # is it beneficial to define all functions as f(w,x,L=1) = g(w, x/L) ?
        # or should this be done in a separate method: self.g(self.f, w, x, L) ?
        
        # each equation is a tuple: (kw, depends, d/dw, function)
        #    d/dw due to the equation itself 
        #    note none of the equations introduce a d/dx 
        
        def _ct_1_(w,x=0):                                                  #0,1
            return 1
        
        def _CTL__vr_Z0TL_(w,x=0):                                           # 5
            return 1 / (self.Z0TL(w,x) * c0 * self.vr(w,x))
        
        def _CTL__LTL_Z0TL_(w,x=0):                                          # 7
            return  self.LTL(w,x) / self.Z0TL(w,x) ** 2
        
        def _CTL__LTL_vr_(w,x=0):                                            #11
            return 1 / (((self.vr(w,x) * c0)**2) * self.LTL(w,x))
        
        def _rjwL__LTL_rTL_(w,x=0):                                          # 2
            return self.rTL(w,x) + 1j * w * self.LTL(w,x)
        
        def _gjwC__CTL_gTL_(w,x=0):                                          # 3
            return self.gTL(w,x) + 1j * w * self.CTL(w,x)
        
        def _gTL__etar_sigma_Z0TL_(w,x=0):                                   #16
            return 2*np.pi*self.sigma(w,x)*self.etar(w,x)*eta0coax/self.Z0TL(w,x)
        
        def _gTL__ID_OD_sigma_(w,x=0):                                       #15
            return 2*np.pi * self.sigma(w,x) / np.log(self.OD(w,x)/self.ID(w,x))
        
        def _LTL__vr_Z0TL_(w,x=0):                                           # 4
            return self.Z0TL(w,x) / (c0 * self.vr(w,x))
        
        def _LTL__CTL_Z0TL_(w,x=0):                                          # 6
            return self.Z0TL(w,x) ** 2 * self.CTL(w,x)
        
        def _LTL__CTL_vr_(w,x=0):                                            #10
            return 1 / (((self.vr(w,x) * c0)**2) * self.CTL(w,x))
        
        def _Z0TL__CTL_LTL_(w,x=0):                                          # 8
            return np.sqrt(self.LTL(w,x)/self.CTL(w,x))
        
        def _Z0TL__etar_ID_OD_(w,x=0):                                       #12
            return self.etar(w,x) * eta0coax * np.log(self.OD(w,x)/self.ID(w,x))
        
        def _ID__etar_OD_Z0TL_(w,x=0):                                       #13
            return self.OD(w,x) / np.exp(self.Z0TL(w,x)/(eta0coax*self.etar(w,x)))
        
        def _OD__etar_ID_Z0TL_(w,x=0):                                       #14
            return self.ID(w,x) * np.exp(self.Z0TL(w,x)/(eta0coax*self.etar(w,x)))
        
        def _vr__LTL_CTL_(w,x=0):                                            # 9
            return 1 / (c0 * np.sqrt(self.LTL(w,x) * self.CTL(w,x)))
        
        def _rTL__rTLI_rTLO_(w,x=0):                                         #17
            return self.rTLI(w,x) + self.rTLO(w,x)
        
        def _rTLI__rTL_rTLO_(w,x=0):                                         #18
            return self.rTL(w,x) - self.rTLO(w,x)
        
        def _rTLI__qTLI_RsI_(w,x=0):                                         #20
            return self.qTLI(w,x) * self.RsI(w,x)
        
        def _rTLO__rTL_rTLI_(w,x=0):                                         #19
            return self.rTL(w,x) - self.rTLI(w,x)
        
        def _rTLO__qTLO_RsO_(w,x=0):                                         #21
            return self.qTLO(w,x) * self.RsI(w,x)
        
        def _rTL__A_CTL_gTL_LTL_(w,x=0):
            wL = w * self.LTL(w,x)
            wC = w * self.CTL(w,x)
            gwC = self.gTL(w,x) / wC
            u = self.A(w,x) / np.sqrt(wL * wC)
            u2 = u ** 2
            D2 = (gwC ** 2 + 1) * (u2 + 1)
            rwL = - (1 + 2 * u2) * gwC + 2 * u * np.sqrt(D2) 
            return rwL * wL
            
        def _gTL__A_CTL_LTL_rTL_(w,x=0):
            wL = w * self.LTL(w,x)
            wC = w * self.CTL(w,x)
            rwL = self.rTL(w,x) / wL
            u = self.A(w,x) / np.sqrt(wL * wC)
            u2 = u ** 2
            D2 = (rwL ** 2 + 1) * (u2 + 1)
            gwC = - (1 + 2 * u2) * rwL + 2 * u * np.sqrt(D2) 
            return gwC * wC
        
        def _A__CTL_gTL_LTL_rTL_(w,x=0):
            # could be _A__gamma_ : np.real(self.gamma(w,x))
            wL = w * self.LTL(w,x)
            wC = w * self.CTL(w,x)
            gwC = self.gTL(w,x) / wC 
            rwL = self.rTL(w,x) / wL
            rwLgwC1 = (1 - rwL * gwC)
            a2 = rwLgwC1 * (np.sqrt(1 +((rwL + gwC)/rwLgwC1)**2) - 1) / 2
            return np.sqrt(a2 * wL * wC)
        
        def _gamma__gjwC_rjwL_(w,x=0):
            return (self.gjwC(w,x)*self.rjwL(w,x))**0.5
        
        def _Zc__gjwC_rjwL_(w,x=0):
            return (self.rjwL(w,x)/self.gjwC(w,x))**0.5
            
        #TODO: continue refactorting of lambdas -> named functions
        
        EQNS = [
            #--  -----------------------------------------------------------  --
            # reasonable assumptions (unless we add equations to find these: in
            #                         that case we should move them to the end)
            ('murI', (), 0., _ct_1_),                                        # 0
            ('murO', (), 0., _ct_1_),                                        # 1
            #--  -----------------------------------------------------------  --
            # mandatory
            ('rjwL', ('rTL','LTL'), 1., _rjwL__LTL_rTL_),                    # 2
            ('gjwC', ('gTL','CTL'), 1., _gjwC__CTL_gTL_),                    # 3
            #--  ------------------------------------------------------------ --            
            ('LTL', ('Z0TL','vr'), 0., _LTL__vr_Z0TL_),                      # 4
            ('CTL', ('Z0TL','vr'), 0., _CTL__vr_Z0TL_),                      # 5
            #--  ------------------------------------------------------------ --            
            ('LTL', ('Z0TL','CTL'), 0., _LTL__CTL_Z0TL_),                    # 6
            ('CTL', ('Z0TL','LTL'), 0., _CTL__LTL_Z0TL_),                    # 7
            ('Z0TL', ('LTL','CTL'), 0., _Z0TL__CTL_LTL_),                    # 8
            #--  -----------------------------------------------------------  --
            ('vr', ('CTL', 'LTL'), 0., _vr__LTL_CTL_),                       # 9
            ('LTL', ('CTL', 'vr'), 0., _LTL__CTL_vr_),                       #10
            ('CTL', ('LTL', 'vr'), 0., _CTL__LTL_vr_),                       #11
            #--  ------------------------------------------------------------ --
            ('Z0TL', ('etar', 'ID','OD'), 0., _Z0TL__etar_ID_OD_),           #12
            ('ID', ('etar', 'OD', 'Z0TL'), 0., _ID__etar_OD_Z0TL_),          #13
            ('OD', ('etar', 'ID', 'Z0TL'), 0., _OD__etar_ID_Z0TL_),          #14
            #--  -----------------------------------------------------------  --
            ('gTL', ('ID', 'OD', 'sigma'), 0., _gTL__ID_OD_sigma_),          #15
            ('gTL', ('sigma', 'Z0TL', 'etar'), 0., _gTL__etar_sigma_Z0TL_),  #16
            #--  -----------------------------------------------------------  --
            ('rTL', ('rTLI','rTLO'), 0., _rTL__rTLI_rTLO_),                  #17
            ('rTLI', ('rTL','rTLO'), 0., _rTLI__rTL_rTLO_),                  #18
            ('rTLO', ('rTL','rTLI'), 0., _rTLO__rTL_rTLI_),                  #19
            #--  -----------------------------------------------------------  --
            ('rTLI', ('RsI', 'qTLI'), 0., _rTLI__qTLI_RsI_),                 #20
            ('rTLO', ('RsO', 'qTLO'), 0., _rTLO__qTLO_RsO_),                 #21
            #--  -----------------------------------------------------------  --
            # only for circular coax (triggered by the presence of OD or ID)
            ('qTLI', ('ID',), 0., lambda w, x:                               #25
                1/ (np.pi * self.ID(w,x))
            ),
            ('qTLO', ('OD',), 0., lambda w, x:                               #26
                1/ (np.pi * self.OD(w,x))
            ),
            #--  -----------------------------------------------------------  --
            ('RsI', ('rhoI', 'murI'), 1., lambda w, x:                       #27
                np.sqrt(w * mu0 * self.murI(w, x) * self.rhoI(w, x) / 2)
            ),
            ('RsO', ('rhoO', 'murO'), 1., lambda w, x:                       #28
                np.sqrt(w * mu0 * self.murO(w, x) * self.rhoO(w, x) / 2)
            ),
            #--  -----------------------------------------------------------  --
            # medium ...
            ('etar', ('mur', 'epsr'), 0., lambda w, x:                       #29
                np.sqrt(self.mur(w,x) / self.epsr(w,x))
            ),
            ('mur', ('etar', 'epsr'), 0., lambda w, x:                       #30
                self.etar(w, x)**2 * self.epsr(w,x)
            ),
            ('epsr', ('etar', 'mur'), 0., lambda w, x:                       #31
                self.mur(w,x) / self.etar(w,x)**2
            ),
            #--  ------------------------------------------------------------ --
            ('vr', ('mur','epsr'), 0., lambda w, x:                          #32
                1 / np.sqrt(self.mur(w,x) * self.epsr(w,x))
            ),
            ('mur', ('vr', 'epsr'), 0., lambda w, x:                         #33
                self.epsr(w, x) / self.vr(w, x)**2
            ),
            ('epsr', ('vr','mur'), 0., lambda w, x:                          #34
                self.mur(w, x) / self.vr(w, x)**2
            ),
            #-- ------------------------------------------------------------- --
            ('mur', ('etar', 'vr'), 0., lambda w, x:                         #35
                self.etar(w,x) / self.vr(w,x)
            ),
            ('vr', ('etar', 'mur'), 0., lambda w, x:                         #36
                self.etar(w,x) / self.mur(w,x)
            ),
            ('etar', ('mur', 'vr'), 0., lambda w, x:                         #37
                self.mur(w,x) * self.vr(w,x)
            ),
            #-- ------------------------------------------------------------- --
            ('etar', ('Z0TL','ID','OD'), 0., lambda w, x:                    #38
                self.Z0TL(w,x) / (eta0coax * np.log(self.OD(w,x)/self.ID(w,x)))
            ),
            #-- ------------------------------------------------------------- --
            ('A', ('AdB',), 0., lambda w, x:                                 #39
                1-10**(self.AdB(w,x)/20)
            ),
            #-- ------------------------------------------------------------- --            
            # not approximations ...
            ('rTL', ('gTL', 'LTL', 'CTL', 'A'), 1, _rTL__A_CTL_gTL_LTL_),    #53
            ('gTL', ('rTL', 'LTL', 'CTL', 'A'), 1, _gTL__A_CTL_LTL_rTL_),    #53
            ('A', ('rTL', 'gTL', 'LTL', 'CTL'), 1, _A__CTL_gTL_LTL_rTL_),
            #-- ------------------------------------------------------------- --
            ('sigma', ('tand', 'epsr'), 0., lambda w, x:                     #52
                w * self.epsr(w,x) * e0 * self.tand(w,x)
            ),
            #-- ------------------------------------------------------------- --
            # resonable assumptions as we found nothing until now ...
            ('mur', (), 0., _ct_1_),                                         #53
            ('epsr', (), 0., _ct_1_),                                        #54
            ('Z0TL', (), 0., lambda w=0, x=0: self.Zbase),                   #55
            #--  -----------------------------------------------------------  --
            # these are a bit iffy but in case there is nothing else ...
            
            #--  -----------------------------------------------------------  --
            # these are only useful if one wants to compute all the parameters
            ('sigma', ('gTL', 'OD', 'ID'), 0., lambda w, x:                  #56
                self.gTL(w,x) * np.log(self.OD(w,x)/self.ID(w,x)) / (2 * np.pi)
            ),
            ('tand', ('sigma', 'epsr'), 1., lambda w, x:                     #57
                self.sigma(w,x) / (w * self.epsr(w,x) * e0)
            ),
            ('Zc', ('rjwL', 'gjwC'), 0., _Zc__gjwC_rjwL_),                   #58
            ('gamma', ('rjwL', 'gjwC'), 0., _gamma__gjwC_rjwL_),             #59
            ('beta', ('LTL', 'CTL'), 1., lambda w, x:                        #60
                w * np.sqrt(self.LTL(w,x) * self.CTL(w,x))
            ),
        ]
                        
        #-- ----------------------------------------------------------------- --
               
        self.kwset = {}
        self.order = []
        for kw, val in kwargs.items():
            if kw in self.trl_kwargs:
                fun, ddw, ddx = self._linfunL(val)
                setattr(self, kw, fun)               
                self.kwset[kw] = {'source':'input', 'ddx': ddx, 'ddw': ddw}
                self.order.append(kw)
                # print(f'input : {kw}')
            else:
                raise ValueError(
                    f'TLresolver: unrecognized kwarg: {kw}')
                
        avail = lambda *__:  [_ in self.kwset for _ in __]
        
        # complement for missing information
        
        if not any(avail('A', 'AdB')):
            if not any(avail('Rs','RsI','RsO',
                             'rTL','rTLO','rTLI', 'rho','rhoI','rhoO')):
                # have no attenuation and nothing to compute conductor losses
                # if any losses they must come from the medium -> rTL = 0
                setattr(self, 'rTL', lambda w, x: 0.)
                self.kwset['rTL'] = {'source':'missing conductor losses',
                                     'ddw': False, 'ddx': False}
                self.order.append('rTL')

            if not any(avail('sigma', 'tand', 'Am', 'AmdB')):
                # have no attenuation and nothing to compute medium losses
                # if any losses they must come from the conductors -> gTL = 0
                setattr(self, 'gTL', lambda w, x: 0.)
                self.kwset['gTL'] = {'source':'missing medium losses',
                                     'ddw': False, 'ddx': False}
                self.order.append('gTL')
                    
        else:
            if not any(avail('Ac', 'AcdB', 'Rs','RsI','RsO',
                             'rTL','rTLO','rTLI', 'rho','rhoI','rhoO',
                             'sigma', 'tand', 'Am', 'AmdB')):
                # we have an attenuation but nothing to compute either conductor
                # or medium losses -> we assume there are no medium losses
                setattr(self, 'gTL', lambda w, x: 0.)
                self.kwset['gTL'] = {'source':'missing medium losses',
                                     'ddw': False, 'ddx': False}
                self.order.append('gTL')
        
        if all(avail('rho')) and not any(avail('rhoI','rhoO')):
            for kw, cnd in zip(['rhoI','rhoO'],['inner','outer']):
                setattr(self, kw, self.rho)
                self.kwset[kw] = {'source':f'assume rho {cnd} is rho',
                                  'ddw': self.kwset['rho']['ddw'], 
                                  'ddx': self.kwset['rho']['ddx']}
                self.order.append(kw)
                    
        #-- ----------------------------------------------------------------- --
    
        def solveit(var = ('Zc','gamma')):
            
            def isavailable(d):
                return all([dk in self.kwset for dk in d])
            
            used = {}
            while not all([hasattr(self,p) for p in var]):
                found = False
                for eqn, (kw, depends, ddw, fun) in enumerate(EQNS):
                        
                    sig = tuple(sorted(depends + (kw,)))
                    if (sig in used 
                        or (sig in self.forget and self.forget[sig] == kw)):
                        continue # forget about equations that were already used
                    
                    gotkw, dependsOK = kw in self.kwset, isavailable(depends)
                    if not gotkw and dependsOK:
                        setattr(self, kw, fun)
                        self.order.append(kw)
                        used[sig] = kw
                        source = f'<-- {", ".join(depends)}' if len(depends) else 'assume'
                        self.kwset[kw] = {
                            'source': f'{source}',
                            'ddx': any([self.kwset[d].get('ddx',False) 
                                        for d in depends]),
                            'ddw': ddw or any([self.kwset[d].get('ddw',False) 
                                              for d in depends]),
                        }
                        found = True
                        # print(f'set {kw} from {depends} -> {self.kwset}' )
                        # print(f'... gTL : {getattr(self, "gTL")}')
                        break
                    
                    elif dependsOK and depends:
                        # we could try and see if we are consistent
                        # we discard to check for empty depends-list because
                        #   these are assumption-like settings for mur and epsr
                        #   when failing to find anything that solves with the
                        #   current set of found kw's
                        f0 = 1E6 if self.f0 in [np.NaN,None] else self.f0
                        x0 = 0
                        tol = 1e-7
                        got, new = getattr(self,kw)(f0,x0), fun(f0,x0)
                        if  np.abs(new - got) > tol * np.abs(new + got):
                            err = np.abs((new - got) / (new + got))
                            print(
                                f'warning: {kw} {self.kwset[kw]["source"]} [{got}] and '
                                f'from {depends} [{new}] differ [{err*100:.2f}%]')
                    
                if not found: break
            
            return found
        
        #-- ----------------------------------------------------------------- --
        
        if not solveit():
            print('TLresolve: could not not solve it')
            print(self)
            
        # else:
            # print(self)
            
    #===========================================================================
    #
    # _ _ str _ _
    #
    def __str__(self, full=True):
        l1 = max([len(self.kwset[kw]["source"]) for kw in self.order])
        s = ''
        f0 = self.f0 if self.f0 not in [np.NaN, None] else 1E6
        s += f'parameters / values at {f0/1e6:.3f} MHz, x=0, w = 2pi f\n'
        for kw in self.order:
            sig = signature(getattr(self, kw)).parameters
            missing = [p for p in ['w','x'] if p not in sig]
            missing = f'!! missing {",".join(missing)} !!' if missing else ''
            
            tsig = [p for p in sig if self.kwset[kw]['dd'+p]]
            if (full or kw in ['rjwL','gjwC'] 
                or self.kwset[kw]["source"].find('input') >= 0):  
                s += f'{kw:<10s} = f{"("+",".join(tsig)+")":7}{missing} '
                if full:
                    s += f'{self.kwset[kw]["source"]:<{l1}s}'
                s += f' = {getattr(self,kw)(2*np.pi*f0,0)} \n'
        return s
    
    #===========================================================================
    #
    # _ l i n f u n L
    #    
    def _linfunL(self, p):
        """
        returns lambda, ddw, ddx
        
        ddw, ddx: if True function depends on w resp. x
        """
        
        if hasattr(p, '__call__'):
            # this is already a function: just return it
            tsig = list(signature(p).parameters.keys()) # OrderedDict keys -> List
            if len(tsig) > 2:
                raise ValueError(
                    'TLresolver._linfun: expect function of at most 2 parameters')
                
            ddw, ddx = 'w' in tsig, 'x' in tsig
            if ddw:
                if tsig.index('w') == 0:
                    # normal w comes first
                    if ddx:
                        return p, ddw, ddx
                    else:
                        # only function of w
                        return (lambda w, x: p(w)), ddw, ddx
                else:
                    # reverse order x and w
                    return (lambda w, x: p(x,w)), ddw, ddx
            else:
                if ddx:
                    # only x
                    return  (lambda w, x: p(x)), ddw, ddx
                else:
                    # constant function
                    return p(), ddw, ddx
                  
        # we (should) have a scalar numerical or a 1/2-D array like
        p = np.array(p)
            
        if len(p.shape) == 0:
            return (lambda w, x: p), False, False              # @UnusedVariable
    
        elif len(p.shape) == 1:
            
            if len(p) == 1:
                # length 1 1D arrays are considered equivalent to scalars
                return lambda w, x: p[0], False, False         # @UnusedVariable
    
            else:
                # we have an 1D array: so no idea of the TL length -> assume 1
                # print(f'p.shape = {p.shape} == (#,)')
                
                # so to call for a TL length L call with f(w,x/L)
                # fixme: how does the caller know what to do ?
                
                xs = np.linspace(0., 1., num=len(p)) * self.L
                # print(f'  xs = {xs}')
                f = interp1d(xs, p, kind='linear', 
                             bounds_error=False, fill_value=(p[0],p[-1]))
                self.constant = False
                return lambda w, x: f(np.array(x)), False, True
            
        elif len(p.shape) == 2:
            # print(f'p.shape = {p.shape} == (2, #)')
            # here the x coordinate should include the begin and end points
            # print(p)
            f = interp1d(p[0,:], p[1,:], kind='linear', 
                         bounds_error=False, fill_value=(p[1,0],p[1,-1]))
            self.constant = False
            return lambda w, x: f(np.array(x)), False, True
        
        else:
            raise NotImplementedError(
                'rfTRL._linfunL: p must be a scalar or a 1-d array/list ' 
                '(shape= (N,)) of scalars or a 2-d array list of scalars '
                '(shape=(2,N))')
            
#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    
    import matplotlib.pyplot as pl
    
    fMHz = 47.5
    xs = 0.
    w = 2e6 * np.pi * fMHz
    
    kwargs = [
#         {'Z0TL':[30,30]},
#         {'Z0TL':30, 'A': 1.e-3},
#         {'Z0TL':30, 'A': 1.e-4},
#         {'Z0TL':30, 'A': 1.e-5},
#         {'Z0TL':30, 'A': 3.1437675233744445e-05, 'tand': 1e-4},
#         {'Z0TL':30, 'tand': 1e-4},
#         {'Z0TL':30, 'OD':0.230, 'rho':2e-8},
        {'OD': 223.0, 'ID': 140.0, 'epsr': 1.640, 'tand': 6.000E-5, 'rho': 0.},    
    ]
    
    for kwarg in kwargs[-1:]:
        print('='*80)
        print(f'kwargs: {kwarg}')
        print('-'*80)
        a = TLresolver(f0 = fMHz*1e6, **kwarg)
        print(a)
        
    print(a.__str__(full=True))
               
