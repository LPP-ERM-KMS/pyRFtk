################################################################################
#                                                                              #
# Copyright 2018                                                               #
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

"""
Created on 17 Dec 2018

@author: frederic

"""

#===============================================================================

import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import odeint

from scipy.constants import speed_of_light as c0, epsilon_0, mu_0
eta_0 = np.sqrt(mu_0/epsilon_0)
eta_0_coax = eta_0 / (2 * np.pi)

import pyRFtk.scatter3a as sc
from pyRFtk.scatter3a import S_from_VI

def count(kws, kwargs):
    ikw = [kw for kw in kwargs if kw in kws]
    return len(ikw)

#===============================================================================

class Conical_TL():
    def __init__(self, **kwargs):
        """
        mandatory :
        
          fMHz : frequency [MHz] (default: 0.)
          
          L : length [m]         (default: 0.)
          
        optional :
        
          Zbase : reference impedance [Ohm] (default 50.)
          
          ODs : outer diameters [m] float, list or tuple of length 2 (at 0, at L)
                (default: (0.2286, 0.2286) = 9" TL)
                
          IDs : inner diameters [m] float, list or tuple of length 2 (at 0, at L)
                (default: (0.0993, 0.0993) = 50. Ohm 9" TL)
                
          rhos : conductor material resitivities [Ohm.m] float, list or tuple of
                 length 2 (outer, inner) (default: (0., 0.))
                 
          maxdeg : [deg] fraction of wavelength in medium 
                   (default: 5 = 72pts / wavelength)
                   
          maxdx : [m] distance beteen points (overrides maxdeg) (default: None)
          
          xs : [m] list of points along the TL (overrides maxdx and maxdeg)
               (default: None)
          
          odeparams : dict passed to scipy.integrate.odeint
          
        at most one of :
          sigma : medium conductivity [Mho/m] (default: 0.)
          tand : tan(delta) [1] (default: 0.)
        
        at most two of:
          vr : medium wave velocity relative to vacuum [1] (default: 1.)
          etar : medium wave relative to vacuum [1] (default: 1.)
          epsr : medium dielectric permitivity relative to vacuum [1] (default: 1.)
          mur : medium  magnetic permeability relative to vacuum [1] (default: 1.)
          
        """
        
        self.L = kwargs.pop('L', 0.)
        self.fMHz = kwargs.pop('fMHz', 0.)
        
        self.Zbase = kwargs.pop('Zbase', 50.)
        
        self.ODs = kwargs.pop('ODs', (0.2286, 0.2286)) 
        self.IDs = kwargs.pop('IDs', (0.0993, 0.0993))
        self.rhos = kwargs.pop('rhos', (0., 0.))
        
        self.maxdeg = kwargs.pop('maxdeg', 5)
        self.maxdx = kwargs.pop('maxdx', None)
        self.xs = kwargs.pop('xs', None)
        
        if count(['sigma','tand'], kwargs) > 1:
            raise ValueError('Conical_TL.__init__: only one of "sigma" or "tand"'
                             ' allowed.')
            
        self.sigma = kwargs.pop('sigma', None)
        self.tand = kwargs.pop('tand', None)
        
        if count(['v0','epsr', 'mur','eta'], kwargs) > 2:
            raise ValueError('Conical_TL.__init__: only two of "v0", "eta", '
                             '"epsr" or "mur" allowed.')
            
        self.epsr = kwargs.pop('epsr', None)
        self.mur = kwargs.pop('mur',None)
        self.v0 = kwargs.pop('vr', None)
        self.eta = kwargs.pop('eta', None)
        
        self.odeparams = kwargs.pop('odeparams',{})
        
        if kwargs:
            raise ValueError('Conical_TL.__init__: unknown keyword(s): %r' %
                             [kw for kw in kwargs])
            
        #- ------------------------------------------------------------------- -
        
        self.w = 2e6 * np.pi * self.fMHz

        if self.v0:
            if self.eta:
                self.mur = self.eta / self.v0
                self.epsr = 1/(self.eta * self.v0)
                
            else:
                if self.epsr:
                    self.mur = 1 / (self.epsr * self.v0**2)
                
                else:
                    if self.mur is None:
                        self.mur = 1.0
                
                    self.epsr = 1 / (self.mur * self.v0**2)
            
                self.eta = np.sqrt(self.mur / self.epsr)
                
        elif self.eta:
            if self. epsr:
                self.mur = self.eta**2 * self.epsr
                
            else:
                if self.mur is None:
                    self.mur = 1.0
                
                self.epsr = self.mur / self.eta**2
            
            self.v0 = 1 / np.sqrt(self.mur * self.epsr)
            
        
        else: # only epsr and/or mur left
            if self.epsr:
                if self.mur is None:
                    self.mur = 1.0
                    
            elif self.mur: # assume epsr = 1.0
                self.epsr = 1.0
                
            else: # assume mur = epsr = 1.0
                self.mur = 1.0
                self.epsr = 1.0
                
            self.v0 = 1/np.sqrt(self.mur * self.epsr)
            self.eta = np.sqrt(self.mur / self.epsr)                
             
        #- ------------------------------------------------------------------- -
        
        if self.sigma:
            self.tand = self.sigma / (self.w * self.epsr * epsilon_0)
            
        elif self.tand:
            self.sigma = self.w * self.epsr * epsilon_0 * self.tand
        
        else:
            self.sigma = 0.
            self.tand = 0.
            
        #- ------------------------------------------------------------------- -
        
        if not isinstance(self.rhos,(tuple,list)):
            self.rhos = (self.rhos, self.rhos)
            
        self.Rs_pi = [np.sqrt(self.w * mu_0 * self.rhos[0] / 2) / np.pi,
                      np.sqrt(self.w * mu_0 * self.rhos[1] / 2) / np.pi]
        
        #- ------------------------------------------------------------------- -
        
        if self.xs is None:
            if self.maxdx is None:
                self.maxdx = 2*np.pi*self.v0*c0/self.w * self.maxdeg/360.
                
            self.xs = np.linspace(0., self.L, num=max(2,round(self.L/self.maxdx)))
            
        #- ------------------------------------------------------------------- -
        
        self.solveVI()

    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self):
        def isfun(var):
            if hasattr(var,'__call__'):
                varf = '(x)'
                vark = var(0), var(self.L)
            else:
                varf=''
                if isinstance(var,(list,tuple)):
                    vark = var[0], var[1]
                else:
                    vark = var, var
            return vark[0], vark[1], varf
            
        s  = 'Conical_TL(\n'
        s += '  fMHz    = %10.3f MHz\n' % self.fMHz
        s += '  L       = %10.3f m\n' % self.L
        s += '  ODs     = [%.4f, %.4f]%s m\n' % isfun(self.ODs)
        s += '  IDs     = [%.4f, %.4f]%s m\n' % isfun(self.IDs)
        s += '  mur     = %10.3f\n' % self.mur
        s += '  epsr    = %10.3f\n' % self.epsr
        s += '  eta     = %10.3f\n' % self.eta
        s += '  vr      = %10.3f\n' % self.v0
        s += '  sigma   = %10.3f muMho/m\n' % (self.sigma*1E6)
        s += '  tan d   = %10.5g\n' % self.tand
        s += '  rhos    = [%10.5g, %10.5g] Ohm.m\n' % tuple(self.rhos)
        s += '  Rs      = [%10.5g, %10.5g] Ohm\n' % (self.Rs_pi[0]*np.pi,
                                                     self.Rs_pi[1]*np.pi)
        s += ')'
        return s
    
    #===========================================================================
    #
    # O D
    #
    def OD(self,x):
        if hasattr(self.ODs, '__call__'):
            return self.ODs(x)
        
        elif isinstance(self.ODs,(list,tuple)):
            return self.ODs[0] + x * (self.ODs[1] - self.ODs[0]) / self.L
        
        else:
            return self.ODs
        
#         else:
#             raise ValueError('Conical_TL.OD: ODs must be a 1 parameter function '
#                              ' or a list/tuple of two values')
             
    #===========================================================================
    #
    # I D
    #
    def ID(self,x):
        if hasattr(self.IDs, '__call__'):
            return self.IDs(x)
        
        elif isinstance(self.IDs,(list,tuple)):
            return self.IDs[0] + x * (self.IDs[1] - self.IDs[0]) / self.L
        
        else:
            return self.IDs
        
#         else:
#             raise ValueError('Conical_TL.ID: ODs must be a 1 parameter function '
#                              ' or a list/tuple of two values')
             
    #===========================================================================
    #
    # Z 0
    #
    def Z0(self,x):
        self.ODx = self.OD(x)
        self.IDx = self.ID(x)
        self.logODIDx = np.log(self.ODx/self.IDx)
        return self.eta * eta_0_coax * self.logODIDx
    
    #===========================================================================
    #
    # T L p r o p s 
    #
    def TLprops(self, x, verbose = False):
        self.Z0x = self.Z0(x)
        self.rTLx = self.Rs_pi[0]/self.ODx + self.Rs_pi[1]/self.IDx
        self.gTLx = 2*np.pi / self.logODIDx * self.sigma
        jwLr = 1j * self.w * self.Z0x/(self.v0*c0) + self.rTLx
        jwCg = 1j * self.w / (self.Z0x * self.v0*c0) + self.gTLx
        
        if hasattr(x, '__iter__'):
            pl.figure('TLprops', figsize=(10,10))
            
            ax = pl.subplot(2,2,1)
            pl.plot(x, self.ODx,'g',label='OD')
            pl.plot(x, self.IDx,'b',label='ID')
            pl.grid()
            pl.xlabel('x [m]')
            pl.title('OD, ID [m]')
            pl.legend(loc='best')
            
            
            pl.subplot(2,2,3,sharex=ax)
            pl.plot(x, self.w * self.Z0x/(self.v0*c0), 'r', 
                    label='$\omega$L')
            
            pl.plot(x, self.Z0x, 'b', label='Z$_0$')
            
            pl.plot(x, (self.Z0x * self.v0*c0)/self.w,'g',
                    label='1/$\omega$C' )
            
            pl.grid()
            pl.xlabel('x [m]')
            pl.title('Z$_0$, $\omega$L, 1/$\omega$C [$\Omega$]')
            pl.legend(loc='best')
            
            pl.subplot(2,2,2, sharex=ax)
            pl.plot(x, self.rTLx*1E3, 'g.-', label='r$_{TL}$')
            pl.grid()
            pl.xlabel('x [m]')
            pl.title('r$_{TL}$ [m$\Omega$]')
            pl.legend(loc='best')
            
            pl.subplot(2,2,4, sharex=ax)
            pl.plot(x, self.gTLx*1E6, 'g.-', label='g$_{TL}$')
            pl.grid()
            pl.xlabel('x [m]')
            pl.title('g$_{TL}$ [M$\Omega^{-1}/m$]')
            pl.legend(loc='best')
            
            pl.tight_layout()
            
        elif verbose:
            print('%6.3f, %6.2f, %10.6f%+10.6fj, %10.6f%+10.6fj' %
                  (x, self.Z0x, jwLr.real, jwLr.imag, jwCg.real, jwCg.imag))
                
        return jwLr, jwCg
        
    #===========================================================================
    #
    # e q n V I
    #
    def eqnVI(self, VI4, x, verbose):
        """
        VI4 = [V.real, V.imag, I.real, I.imag]
        """
        jwLr, jwCg = self.TLprops(x,verbose=verbose)
        
        dVC = jwLr * (VI4[2]+1j*VI4[3])
        dIC = jwCg * (VI4[0]+1j*VI4[1])
        
        return [dVC.real, dVC.imag, dIC.real, dIC.imag]

    #===========================================================================
    #
    # s o l v e V I
    #
    def solveVI(self, VI = None, verbose=False):
        """
        solveVI ...
        """
        if VI is None:
            sol10 = odeint(self.eqnVI, [1., 0., 0., 0.], self.xs, 
                           args=(verbose,), **self.odeparams)
            self.U10 = sol10[:,0]+1j*sol10[:,1]
            self.I10 = sol10[:,2]+1j*sol10[:,3]
            
            sol01 = odeint(self.eqnVI, [0., 0., 1., 0.], self.xs, 
                           args=(verbose,), **self.odeparams)
            self.U01 = sol01[:,0]+1j*sol01[:,1]
            self.I01 = sol01[:,2]+1j*sol01[:,3]
            
            self.VIt = np.array([[self.U10[-1], self.U01[-1]],
                                 [self.I10[-1], self.I01[-1]]])
    
            self.S = S_from_VI(self.VIt, self.Zbase)
            
        else:
            if False:
                # solve ODE again ...
                VI0 = [VI[0].real, VI[0].imag, VI[1].real, VI[1].imag]
                sol = odeint(self.eqnVI, VI0, self.xs, 
                             args=(verbose,), **self.odeparams)
                self.U = sol[:,0]+1j*sol[:,1]
                self.I = sol[:,2]+1j*sol[:,3]
                
            else:
                # much quicker ...
                self.U = VI[0] * self.U10 + VI[1] * self.U01
                self.I = VI[0] * self.I10 + VI[1] * self.I01
            
            return self.U, self.I
            
    #===========================================================================
    #
    # tg e t _ S Z 
    #
    def get_SZ(self, **kwargs):
        """
        Zbase : reference impedance [Ohm] (default: self.Zbase)
        Portnames : (default: ['1', '2']
        """
        Zbase = kwargs.pop('Zbase', self.Zbase)
        Portnames = kwargs.pop('Portnames',['1','2'])

        if kwargs:
            raise ValueError('Conical_TL.get_SZ: unknown keywords: %r' %
                             [kw for kw in kwargs])
            
        SZ = sc.Scatter(fMHz=self.fMHz, Zbase=Zbase, Portnames=Portnames,
                        S = S_from_VI(self.VIt,Zbase)
                        #Todo: there are plenty of other parameters
                       )
        return SZ
    
    #===========================================================================
    #
    # p l o t s t u f f
    #
    def plotstuff(self):
        Pin10 = np.real(np.conj(self.U10) * self.I10) / 2
        Pin01 = np.real(np.conj(self.U01) * self.I01) / 2

        pl.figure('VSW')
        pl.plot(self.xs, np.abs(self.U10),'r',label='U$_{10}$')
        pl.plot(self.xs, np.abs(self.U01),'b',label='U$_{01}$')
        pl.grid()
        pl.xlabel('distance [m]')
        pl.ylabel('|U| [V]')
        pl.legend(loc='best')
        
        pl.figure('Power transfer')
        pl.plot(self.xs, Pin10,'r',label='sol10')
        pl.plot(self.xs, Pin01,'b',label='sol01')
        pl.grid()
        pl.xlabel('distance [m]')
        pl.ylabel('P [W]')
        pl.legend(loc='best')
        
        self.TLprops(self.xs)

#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    
    from pyRFtk import TouchStoneClass3a as _ts
    
    fMHz, Zbase, L = 47.5, 50., 10.000
    
    OD, ID = 0.2286, 0.0993           # (default diameters and Z0TL)
    Z0TL = eta_0_coax * np.log(OD/ID) # (default resulting Z0TL)
    
    # ==========================================================================
    # case 1: compare with scatter3a lossless
    #
    myCTL = Conical_TL(fMHz=fMHz, L=L, Zbase=Zbase,
                       # rhos=2E-8,
                       # tand=1E-4, 
                       ODs=OD, 
                       IDs=ID,
                       # maxdx=0.01
                       # odeparams={'rtol':1E-12, 'atol':1E-12}
                      )
    print(myCTL)
    CSZ = myCTL.get_SZ(Zbase=Zbase)
    CSZ.pfmt = ('RI','%+14.10f%+14.10fj')
    print(CSZ)
    
    SZ = sc.Scatter(fMHz=fMHz, Zbase=Zbase, pfmt=CSZ.pfmt)
    SZ.trlAZV(['1','2'], L, Z=Z0TL)
    print(SZ)
    
    print('%10.3e\n' % np.max(np.max(np.abs(CSZ.S - SZ.S))))
    
    # ==========================================================================
    # case 2: compare with scatter3a losses rho = 2E-8
    #
    rho = 2E-8
    
    myCTL = Conical_TL(fMHz=fMHz, L=L, Zbase=Zbase,
                       rhos=rho,
                       # tand=1E-4,
                       ODs=OD, 
                       IDs=ID,
                       # maxdx=0.01
                       # odeparams={'rtol':1E-12, 'atol':1E-12}
                      )
    
    print(myCTL)
    CSZ = myCTL.get_SZ(Zbase=Zbase)
    CSZ.pfmt = ('RI','%+14.10f%+14.10fj')
    print(CSZ)
    
    SZ = sc.Scatter(fMHz=fMHz, Zbase=Zbase, pfmt=CSZ.pfmt)
    A = sc.CoaxAttenuation(fMHz, Z0TL, D0=OD, rho_inner=rho, rho_outer=rho)
    SZ.trlAZV(['1','2'], L, Z=Z0TL, A=A)
    print(SZ)
    
    print('%10.3e\n' % np.max(np.max(np.abs(CSZ.S - SZ.S))))
    
    # ==========================================================================
    # case 3: create touchstone file of lossy (rho = 2E-8 Ohm.m)
    #         OD 0.2286 ID 0.050 -> 0.150 (frequency 35 -> 60 MHz)
    #         TL length 1.0m and 10.0m
    Ls, OD, IDs, rhos = [1.0, 10.0], 0.2286, [(0.0993, 0.0993), (0.050, 0.150)], [0.,2E-8]
    
    fMHzs = np.linspace(35., 60., 51)
    
    for ID in IDs:
        for rho in rhos:
            for L in Ls:
                tsCTL = _ts.TouchStone(tformat='MHZ RI S R %f' % Zbase)
                for fMHz in fMHzs:
                    myCTL = Conical_TL(fMHz=fMHz, L=L, Zbase=Zbase,
                                       rhos=rho,
                                       # tand=1E-4,
                                       ODs=OD, 
                                       IDs=ID,
                                       # maxdx=0.01
                                       # odeparams={'rtol':1E-12, 'atol':1E-12}
                                      )
                    tsCTL.Add_Scatter(myCTL.get_SZ())
                
                fname = "CTL_OD=%gmm_ID=%gmm_%gmm_L=%gm_rho=%gOhm.m.s2p" % (
                         OD*1E3, ID[0]*1E3, ID[1]*1E3, L, rho)
        
                print('writing %r' % fname)
                
                tsCTL.WriteFile(fname,'Created by Conical_TL.py\n' + str(myCTL))
    
    # ==========================================================================
    
    pl.show()
    