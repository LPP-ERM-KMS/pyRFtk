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

2019-08-07 attempt to improve speed and picklingness

"""

#===============================================================================

import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import pprint as pp
import re
import sys

import logging
LOGGER_NAME = 'General_TL'


from scipy.constants import speed_of_light as c0, epsilon_0, mu_0
eta_0 = np.sqrt(mu_0/epsilon_0)
eta_0_coax = eta_0 / (2 * np.pi)

import pyRFtk.scatter3a as sc
from pyRFtk.scatter3a import S_from_VI
from Utilities.SI_scale import SI_scale

#===============================================================================

def count(kws, kwargs):
    ikw = [kw for kw in kwargs if kw in kws]
    return len(ikw)

#===============================================================================

def isnummeric(a):
    # note that np.isrealobj is not useable as it is implemented a not iscomplexobj !
    # therefore isrealobj(of a non-nummeric returns True !
    # np.isreal is sort of useable but returns True for booleans and e.g. dicts (!)
    
    if np.iscomplexobj(a):
        return True # accept it here : it avoids a ComplexWarning further down
    
    try:
        np.isreal(a)
        return not isinstance(a, (bool, dict))
    except:
        return False

#===============================================================================
#
# S C A L E R  
#
# m = milli, u = micro, n = nano, p = pico
#
SCALER = {'OD': 'm', 'ID': 'm',
          'rho': 'u', 'rhoO': 'u', 'rhoI': 'u',
          'rTL': 'm', 'rTLO': 'm', 'rTLI': 'm',
          'qTL': '', 'qTLI': '', 'qTLO': '',
          'RsI': 'm', 'RsO': 'm', 'A': '', 'AdB': '',
          'murI': '', 'murO': '',
          'gTL': 'u', 'sigma': 'u', 'tand': '',
          'LTL': 'n', 'CTL': 'p', 'Z0TL': '',
          'epsr': '', 'mur': '', 'vr': '', 'etar': '',
          'L':'', 'w':'', 'beta':'',
          'Zc': '', 'gamma': '', 'rjwL': '', 'gjwC':'',
          'Zmedium':''
         }

#===============================================================================

class General_TL():
    
    EQNS = {#--- mandatory
            'LTL':[ #V01
                '$Z0TL / (c0 * $vr)',                                    #01*
                '$Z0TL**2 * $CTL',                                       #02*
                '$epsr * $mur / (c0**2 * $CTL)',                         #14* 2021/2/26 corrected !
                # where does this come from ??
                # '$mur / $epsr * $CTL',                                   #15*
            ],
            'CTL':[ #V02
                '$LTL / $Z0TL**2',                                       #02*
                '1 / ($Z0TL * c0 * $vr)',                                #03*
                '$epsr * $mur / (c0**2 * $LTL)',                         #14* 2021/2/26 corrected !
                # where does this come from ??
                # '$epsr / $mur * $LTL',                                   #15*
            ],
            'rTL':[ #V03
                '$rTLI + $rTLO',                                         #09*
                '$rTLO * (1 + ($RsI/$ID)/($RsO/$OD))',                   #12*
                '$rTLO * ($RsI/$RsO*np.exp($Z0TL/($etar*eta_0_coax))+1)',
                '$rTLI * (($RsO/$OD) / ($RsI/$ID) + 1)',                 #13*
                '$rTLI * (1+$RsO/$RsI*np.exp(-$Z0TL/($etar*eta_0_coax)))',
            #    '$A * 2 * $Z0TL'                                         #18
                '$w*$LTL*np.sqrt((2*($A/($w**2*$LTL*$CTL))**2+1)**2-1)',  #18'
            ],
            'gTL':[ #V04
                '2 * np.pi * $sigma / np.log($OD/$ID)',                  #17*
            ],
            #--- other TL properties
            'Z0TL':[ #V05
                '$LTL * c0 * $vr',                                       #01*
                'np.sqrt($LTL / $CTL)',                                  #02*
                '1 / ($CTL * c0 * $vr)',                                 #03*
                '$etar * eta_0_coax * np.log($OD / $ID)',                #04*
            ],
            'OD':[ #V06
                '$ID * np.exp($Z0TL / ($etar * eta_0_coax))',            #04*
                '$RsO / (np.pi * $rTLO)',                                #10*
                #done (FIXME): relations below are only valid for RsI == RsO
                '$RsO/$RsI * $ID * ($rTL/$rTLO - 1)',                    #12*
                '$RsI/$RsO * $ID / ($rTL/$rTLI - 1)',                    #13*
                
                # '$ID * np.exp(2 * np.pi * $sigma / $gTL)',               #17*
            ],
            'ID':[ #V07
                '$OD / np.exp($Z0TL / ($etar * eta_0_coax))',            #04*
                '$RsI / (np.pi * $rTLI)',                                #11*
                # done (FIXME): relations below are only valid for RsI == RsO
                '$RsI/$RsO * $OD / ($rTL/$rTLO - 1)',                    #12*
                '$RsO/$RsI * $OD * ($rTL/$rTLI - 1)',                    #13*
                
                '$OD * np.exp(- 2 * np.pi * $sigma / $gTL)',             #17*
            ],
            #--- conductor properties
            'rTLI':[ #V08
                '$rTL - $rTLO',                                          #09*
                '$rTL / (1 + $ID / $OD * $RsO / $RsI)',                  #13*
                '$rTL / (1+$RsO/$RsI*np.exp(-$Z0TL/($etar*eta_0_coax)))',
                '$RsI / (np.pi * $ID)',                                  #11*
                '$qTLI * $RsI',
            ],
            'rTLO':[ #V09
                '$rTL - $rTLI',                                          #09*
                '$rTL / (1 + $OD / $ID * $RsI / $Rs)',                   #12*
                '$rTL / ($RsI/$RsO*np.exp($Z0TL/($etar*eta_0_coax))+1)',
                '$RsO / (np.pi * $OD)',                                  #10*
                '$qTLO * $RsO',
            ],
            'RsI':[ #V10
                '$rTLI * np.pi * $ID',                                   #11*
                'np.sqrt($w * $murI * mu_0 * $rhoI / 2)',
                '$rTLI / $qTLI',
            ],
            'RsO':[ #V11
                '$rTLO * np.pi * $OD',                                   #10*
                'np.sqrt($w * $murO * mu_0 * $rhoO / 2)',
                '$rTLO / $qTLO',
            ],
            'qTLI':[
                '$rTLI / $RsI',
            ],
            'qTLO':[
                '$rTLO / $RsO',
            ],
            
            #--- medium properties
            'mur':[ #V12
                '$etar / $vr',                                           #06*
                '1 / ($epsr * $vr**2)',                                  #07*
                '$etar**2 * $epsr',                                      #05*
                'c0**2 * $LTL * $CTL / $epsr',                           #14* 2021/2/26 corrected !
                '$mur * $LTL / $CTL',                                    #15*
            ],
            'epsr':[ #V13
                '1 / ($etar * $vr)',                                     #08*
                '1 / ($mur * $vr**2)',                                   #07*
                '$mur / $etar**2',                                       #05*
                'c0**2 * $LTL * $CTL / $mur',                            #14* 2021/2/26 corrected !
                '$epsr * $CTL / $LTL',                                   #15*
                '$sigma / ($w * epsilon_0 * $tand)',                     #16*
            ],
            'etar':[ #V14
                '$Z0TL / eta_0_coax * np.log($OD / $ID)',                #04*
                'np.sqrt($mur / $epsr)',                                 #05*
                '$vr * $mur',                                            #06*
                '1 / ($epsr * $vr)',                                     #08*
            ],
            'vr':[ #V15
                '$Z0TL / (c0 * $LTL)',                                   #01*
                '1 / ($Z0TL * c0 * $CTL)',                               #03*
                '$etar / $mur',                                          #06*
                '1 / np.sqrt($mur * $epsr)',                             #07*
                '1 / ($epsr * $etar)',                                   #08*
            ],
            'sigma':[ #V16
                '$w * $epsr * epsilon_0 * $tand',                        #16*
                '$gTL * np.log($OD/$ID) / (2 * np.pi)',                  #17*
            ],
            'tand':[ #V17
                '$sigma / ($w * $epsr * epsilon_0)',                     #16*
            ],
            'A':[ #V18 assume the attenuation only due to conductor losses
            #     '$rTL / (2 * $Z0TL)',                                   #18
                 '$w*np.sqrt($LTL*$CTL*(np.sqrt(1+($rTL/($w*$LTL))**2)-1)/2)', #18'
                 '1-10**($AdB/20)',                                      #19*
            ],
            'AdB':[ #V19
                '20*np.log10(1-$A)',                                     #19*
            ],
                
            # other quantities evaluated when all is resolved
            
            'rjwL':  ['$rTL + 1j * $w * $LTL'],
            'gjwC':  ['$gTL + 1j * $w * $CTL'],
            'gamma': ['($rjwL * $gjwC)**0.5'],
            'Zc':    ['$rjwL/$gamma'],
            'beta':  ['$w*np.sqrt($LTL*$CTL)'],
            'Zmedium': ['$etar * eta_0_coax'],
    }

    def __init__(self, L, **kwargs):
        """
        for the moment this class implements only circular coaxial TLs.
        
        L : length [m] (could be None in which case only the TL properties are
                        evaluated) #TODO: this may not be possible in all cases

        mandatory:
        
          fMHz : frequency [MHz]
          
        optional:
        
          Zbase : reference impedance [Ohm] (default 50.) is used to return the
                  Scatter object and the S-matrix
                  
        TL properties:
          
          OD : outer diameters [m] float, list or tuple of length 2 (at 0, at L)
               (default: (0.2286, 0.2286) = 9" TL)
                
          ID : inner diameters [m] float, list or tuple of length 2 (at 0, at L)
               (default: (0.0993, 0.0993) = 50. Ohm 9" TL)
                
          rho, rhoI, rhoO : 
              conductor material resitivities [Ohm.m] float, list or tuple of
                 length 2 (outer, inner) (default: (0., 0.))
                 
          sigma, tand : 
              sigma : medium conductivity [Mho/m] (default: 0.)
              tand : tan(delta) [1] (default: 0.)
                 
          mur =vr : medium wave velocity relative to vacuum [1] (default: 1.)
          etar : medium wave relative to vacuum [1] (default: 1.)
          epsr : medium dielectric permitivity relative to vacuum [1] (default: 1.)
          mur : medium  magnetic permeability relative to vacuum [1] (default: 1.)

        Solution control:
                 
          maxdeg : [deg] fraction of wavelength in medium 
                   (default: 5 = 72pts / wavelength)
                   
          maxdx : [m] distance beteen points (overrides maxdeg) (default: None)
                  if positive then this is an exact distance and the endpoint
                  may or may not be included; if negative the distance is tweaked
                  to provide equidistant point that includes the end point.
          
          xs : [m] list of points along the TL (overrides maxdx and maxdeg)
               (default: None)
          
          odeparams : dict passed to scipy.integrate.odeint
          
          
        """
        
        self.LOGGER = logging.getLogger(LOGGER_NAME)
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
        np.seterr(all='raise')
        
        self.state = {'L':L, 'kwargs': kwargs.copy()}
        
        self.Nloop = 0
        self.constant = True
        self.const = {}
        self.available = {}
        self.f = {}
        
        self.d = {'OD': None, 'ID': None,
                  'rho': None, 'rhoO': None, 'rhoI': None,
                  'rTL': None, 'rTLO': None, 'rTLI': None,
                  'qTL': None, 'qTLI': None, 'qTLO': None,
                  'murI': 1.0, 'murO': 1.0, 'A': None, 'AdB': None,
                  'gTL': None, 'sigma': None, 'tand': None,
                  'LTL': None, 'CTL': None, 'Z0TL': None,
                  'epsr': None, 'mur': 1.0, 'vr': None, 'etar': None,
                 }
        
        self.units = {'OD': 'm', 'ID': 'm',
                      'rho': 'Ohm.m', 'rhoO': 'Ohm.m', 'rhoI': 'Ohm.m',
                      'RsO': 'Ohm', 'RsI': 'Ohm', 'A':'/m', 'AdB':'dB/m',
                      'qTL': '/m', 'qTLI': '/m', 'qTLO': '/m',
                      'rTL': 'Ohm/m', 'rTLO': 'Ohm/m', 'rTLI': 'Ohm/m',
                      'murI': 'mu_0', 'murO': 'mu_0',
                      'gTL': 'Mho/m', 'sigma': 'Mho/m', 'tand': '1',
                      'LTL': 'H/m', 'CTL': 'F/m', 'Z0TL': 'Ohm',
                      'epsr': 'eps_0', 'mur': 'mu_0', 'vr': 'c0', 'etar': 'eta_0',
                      'L':'m', 'w':'/s', 'beta':'/m',
                      'Zc': 'Ohm', 'gamma': '/m', 'rjwL': 'Ohm', 'gjwC':'Mho',
                      'Zmedium':'Ohm'
                     }
        
        
        self.scaler= SCALER.copy()
        
        # synonymns for compatibility with scatter3a
        synomyms = {'Z':'Z0TL', 'V':'vr',
                    'L':'LTL', 'C':'CTL', 
                    'R':'rTL', 'G':'gTL',
                    'dx':'maxdx'
                   } 
        for var, subst in synomyms.items():
            if var in kwargs:
                kwargs.update({subst:kwargs[var]})
                kwargs.pop(var)
                
        self.kw = [kw for kw in kwargs if kw in self.d]
        
        self.solpath = ['input %s' % kw for kw in self.kw]
        self.solpath += ' '
        self.solpath += ['default %s = %.5g %s%s' % (
            var, self.d[var], self.scaler[var], self.units[var])
            for var in self.d if var not in self.kw and self.d[var] is not None] 
        self.solpath += ' '

        #------------------------------------------------------ static / options
        
        self.maxdeg = kwargs.pop('maxdeg', 5)
        self.maxdx = kwargs.pop('maxdx', None)
        self.xs = kwargs.pop('xs', None)
        self.odeparams = kwargs.pop('odeparams',{'rtol':1E-12, 'atol':1E-12})
        self.scaler.update(kwargs.pop('scaler',{}))
        self.trace = kwargs.pop('trace',[])
        
        self.L = L                              # None -> only analyze the input
        self.fMHz = kwargs.pop('fMHz', 0.)
        self.w = 2e6 * np.pi * self.fMHz
        self.prepfun('w',self.w)
        
        self.Zbase = kwargs.pop('Zbase', 50.)
        
        #-------------------------------------------------------------- variable

        for var, defval in self.d.items():
            val = kwargs.pop(var, defval)
            self.prepfun(var, val, SI_scale(self.scaler[var]))
            
        #--------------------------------------------------------------- unknown
        
        if kwargs:
            raise ValueError('General.__init__: unknown keyword(s): %r' %
                             [kw for kw in kwargs])

        #-------------------------------------------------------- overconstraint
                
        if count(['sigma','tand'], self.available) > 1:
            raise ValueError('General.__init__: only one of "sigma" or "tand"'
                             ' allowed.')
            
        
        if count(['vr','epsr', 'mur','etar'], self.available) > 2:
            raise ValueError('General.__init__: only two of "vr", "etar", '
                             '"epsr" or "mur" allowed.')
        
        #------------------------------------------------------- trivial missing
        
        Ls = 0. if self.L is None else np.linspace(0.,self.L,101)
        
        if not (self.isavailable('A') or self.isavailable('AdB')):
            if not self.isavailable('rho'):
                if not (self.isavailable('rhoI') or self.isavailable('rhoO')):
                    self.solpath.append(
                        '+++ rho = 0. %s%s (no input for A, AdB, rho, rhoI, rhoO)' 
                        % (self.scaler['rTL'],self.units['rTL']))       
                    self.prepfun('rho', 0.)
                    self.LOGGER.info('__init__: rho set to 0.')
                
            if not self.isavailable('rhoI'):
                if self.isavailable('rho'):
                    if self.isconstant('rho'):
                        self.prepfun('rhoI',self.f['rho'](0))
                    else:
                        self.prepfun('rhoI',self.f['rho'])
                    self.solpath.append('+++ rhoI = $rho (no input for rhoI)')
                    self.LOGGER.info('__init__: rhoI set to rho')
                else:
                    self.solpath.append(
                        '+++ rhoI = 0. %s%s (no input for rho, rhoI)' % (
                        self.scaler['rhoI'],self.units['rhoI']))
                    self.prepfun('rhoI',0.)
                    self.LOGGER.info('__init__: rhoI set to 0.')
                
            if not self.isavailable('rhoO'):
                if self.isavailable('rho'):
                    if self.isconstant('rho'):
                        self.prepfun('rhoO',self.f['rho'](0))
                    else:
                        self.prepfun('rhoO',self.f['rho'])
                    self.solpath.append('+++ rhoO = $rho (no input for rhoO)')
                    self.LOGGER.info('__init__: rhoO set to rho')
                else:
                    self.solpath.append(
                        '+++ rhoO = 0. %s%s( no input for rho, rhoO)' % (
                        self.scaler['rTLO'],self.units['rTLO']))
                    self.prepfun('rhoO',0.)
                    self.LOGGER.info('__init__: rho_O set to 0.')
            
            if not self.isavailable('rTLI'):
                if np.max(self.f['rhoI'](Ls)) == 0.:
                    self.prepfun('rTLI', 0.)
                    self.solpath.append(
                        '+++ rTLI = 0. %s%s (rhoI = 0. %s%s) ' % (
                        self.scaler['rTLI'],self.units['rTLI'],
                        self.scaler['rhoI'],self.units['rhoI']))
                    self.LOGGER.info('__init__: rTLI set to 0.')
               
            if not self.isavailable('rTLO'):
                if np.max(self.f['rhoO'](Ls)) == 0.:
                    self.solpath.append(
                        '+++ rTLO = 0. %s%s (rhoO = 0. %s%s) ' % (
                        self.scaler['rTLO'],self.units['rTLO'],
                        self.scaler['rhoO'],self.units['rhoO']))
                    self.prepfun('rTLO', 0.)
                    self.LOGGER.info('__init__: rTLO set to 0.')
               
        if not (self.isavailable('sigma') or self.isavailable('tand')):
            self.prepfun('sigma', 0.)
            self.prepfun('tand', 0.)
            self.solpath.append(
                '+++ sigma = 0. %s%s ; tand = 0. (no input for sigma, tand)' % 
                                (self.scaler['sigma'],self.units['sigma']))
            self.LOGGER.info('__init__: sigma and tand set to 0.')
            
        if not self.isavailable('gTL'):
            # note the first overrides the second
            for var in ['sigma', 'tand']:
                if self.isavailable(var):
                    if np.max(self.f[var](Ls)) == 0.:
                        self.prepfun('gTL', 0.)
                        self.solpath.append('+++ gTL = 0. %s%s (%s = 0. %s%s)' % (
                            self.scaler['gTL'],self.units['gTL'],
                            var, self.scaler[var],self.units[var]))
                        self.LOGGER.info('__init__: gTL set to 0. (%s==0.)' % var)
                        break
                
        #---------------------------------------------------------- find missing
        # We need to find L, C, r, g as a minimum
        # it's a long and painfull process ...
        
        found = False
        while not found:
            self.resolvevars()
            self.LOGGER.info('__init__: currently available parameters: {\n ' +
                             pp.pformat(self.available, indent=20)[1:])
            if not self.isavailable(['LTL', 'rTL', 'CTL', 'gTL']):
                # try and make further reasonable assumptions
                if not self.isavailable('epsr'):
                    # we probably failed because we could not find epsr
                    self.prepfun('epsr', 1.)
                    self.solpath.append('*** assume epsr = 1.')
                    self.LOGGER.info('__init__: setting epsr to 1.0')
                    
                elif not self.isavailable('Z0TL'):
                    self.prepfun('Z0TL',self.Zbase)
                    self.solpath.append('*** assume Z0TL = %.5g (=Zbase)' 
                                        % self.Zbase)
                    self.LOGGER.warning('__init__: setting Z0TL to %.5g Ohm' 
                                     % self.Zbase)
                    
                elif not self.isavailable('OD'):
                    # we probably failed because we could not find OD
                    OD = 0.2286 / SI_scale(self.scaler['OD'])
                    self.solpath.append('*** assume OD = %.5g %s%s' % 
                                 (OD, self.scaler['OD'], self.units['OD']))
                    self.prepfun('OD', OD, SI_scale(self.scaler['OD']))
                    self.LOGGER.critical('__init__: setting OD to %.5g %s' % (
                                        OD, self.scaler['OD']+self.units['OD']))
                    
                else:
                    self.LOGGER.critical(
                        '__init__: nothing else to set reasonably by default')
                    break
                
                found = False
            else:
                found = True

        #- ------------------------------------------------------------------- -

        mandatory = ['LTL','rTL','CTL','gTL']
        if not self.isavailable(mandatory):
            pp.pprint(self.available)
            raise ValueError('General_TL.__init__: could not resolve %r' % 
                        [kw for kw in mandatory if not self.isavailable(kw)])
        
        #- ------------------------------------------------------------------- -
        
        self.solved = False
        
        if self.L is not None:
            if self.xs is None:
                if self.maxdx is None:
                    self.maxdx = -2*np.pi*self.f['vr'](0)*c0/self.w * self.maxdeg/360.
                     
                if self.maxdx < 0.:
                    self.xs = np.linspace(0., self.L, 
                                          num=max(2,-round(self.L/self.maxdx)))

                elif self.maxdx == 0.:
                    raise NotImplementedError(
                        'General_TL return function (maxdx == 0.) is not implemented')
                else:
                    self.xs = [k * self.maxdx 
                               for k in range(int(self.L/self.maxdx)+1)]
        
            self.solveVI()

    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self):
        
        def strvar(var):
            def value(y):
                if isinstance(y, complex):
                    return '%.12g%+.12gj' % (y.real, y.imag)
                else:
                    return '%.12g' % y
                
            L = 1. if self.L is None else self.L
            if var in self.available:
                s = '  %-10s = [%s] ' % (var, self.available[var])
                scale = SI_scale(self.scaler[var])
                unit = self.scaler[var]+self.units[var]
                try:
                    fs = self.f[var](0.)/scale, self.f[var](L)/scale
                except:
                    fs = 0, 0
                    print('error',var,self.f[var],L)
                    
                # print(var,'fs = %r' % (fs,))
                if self.available[var] == 'constant':
                    s += '%s [%s]\n' % (value(fs[0]), unit)
                elif self.available[var] == '2-point':
                    s += '(%s, %s) [%s]\n' % (value(fs[0]), value(fs[1]), unit)
                else: # N-point or function
                    s += '%s ... %s [%s]\n' % (value(fs[0]), value(fs[1]), unit)
            else:
                s = '  %-10s = not defined (%r)\n' % (var,self.d[var])
            return s
        
        s  = 'General_TL(\n'
        s += '  fMHz       = %10.3f MHz\n' % self.fMHz
        if self.L is not None:
            s += '  L          = %10.3f m\n' % self.L
        else:
            s += '  L          = None\n'
        s += '\n'                    
        for var in sorted(self.d):
            if var in self.kw:
                s += strvar(var)
        s += '\n'
        for var in sorted(self.d):
            if var not in self.kw:
                s += strvar(var)
        s += '\n'        
        for var in sorted(self.available):
            if not var in self.d:
                s += strvar(var)
        s += '\n'        
        s += '  constant   = %r\n\n  ' % self.constant
        s += '\n  '.join(self.solpath)
        s += ')'
        
        return s
    
    #===========================================================================
    #
    # i s a v a i l a b l e
    #
    def isavailable(self, v):
        if isinstance(v,(list,tuple,dict)):
            return all([vk in self.available for vk in v])
        else:
            return (v in self.available)
         
    #===========================================================================
    #
    # i s c o n s t a n t
    #
    def isconstant(self,*v):
        return all([self.available[vk] == 'constant' for vk in v])
    
    #===========================================================================
    #
    # e v a l e x p r
    #
    def evalexpr(self, expr, variables = None):
        const = True
        intvars = ''
        if variables is None:
            # use sets to avoid duplicates
            variables = set(re.findall("\$([A-Za-z][A-Za-z0-9]*)", expr))
            # print(variables)
        for l, var in sorted(zip([len(v) for v in variables],variables),
                             reverse=True):
            # replace longest variable names first to avoid replacing 
            #   e.g. $rho before $rhoO
            if self.isconstant(var):
                intvar = "_f_%s_" % var
                intvars += ', ' + intvar + "= self.f['%s'](0)"%var
                expr = expr.replace("$%s" % var, intvar )
            else:
                # print(var,available[var])
                intvar = "_f_%s_" % var
                intvars += ', ' + intvar + "= self.f['%s']"%var
                expr = expr.replace("$%s" % var, "%s(x)" % intvar)
                const = False
                
        expr = 'lambda x' + intvars + ': ' + expr
            
        if self.curvar in self.trace:
            self.LOGGER.warning('evalexpr: %s --> %s' % (', '.join(variables), expr))
        else:
            self.LOGGER.debug('evalexpr: %s --> %s' % (', '.join(variables), expr))
            
        if const:
            try:
                return eval(expr)(0)
            except FloatingPointError:
                # here we should flag an error condition (probably a divide by 0.)
#                 for v in variables:
#                     print(v, self.available[v],self.f[v])
#                     print(' --> ', self.f[v](0))
                return None
            except:
                print(expr)
                raise RuntimeError('')
        else:
            return eval(expr)

    #===========================================================================
    #
    # r e s o l v e v a r
    #
    def resolvevar(self, var, exprs):
        if var not in self.available:
            for expr in exprs if isinstance(exprs,(list,tuple)) else [exprs]:
                variables = set(re.findall("\$([A-Za-z][A-Za-z0-9_]*)", expr))
                tr = var in self.trace
                if tr:  
                    self.LOGGER.warning('resolvevar: tracing %r:%r  %r' % 
                        (var, expr, [(v, self.isavailable(v)) for v in variables]))
                if self.isavailable([v for v in variables]):                    
                    val = self.evalexpr(expr,variables)
                    if val is not None:
                        self.LOGGER.debug('resolvevar: found %r  from  %r' %(var, expr))
                        self.solpath.append('%s = %s' % (var,expr))
                        if tr:
                            self.LOGGER.warning('resolvevar: tracing %r --> %r' %
                                            (var,val))
                        self.prepfun(var, val)
                        return True
                    elif tr:
                        self.LOGGER.warning('resolvevar: tracing %r --> failed' %
                                            var)
        return False

    #===========================================================================
    #
    # s e a r c h v a r
    #
    def searchvar(self, var, searchlist=[], level=[0]):
        ##
        ## attempt to recursively solve the equations for missing parmeters
        ## advantage : possibly feedback on what is missing
        ##
        ## DOES NOT WORK YET
        ##
        level[0] += 1
        lstr = '  '*level[0]
        input('<return>')
        self.LOGGER.debug('%s searchvar: entered at level %d with %r and %r' %
                          (lstr,level[0],var,searchlist))
        
        if self.isavailable(var):
            self.LOGGER.debug('%s searchvar: %r is already found' % (lstr,var))
            level[0] -= 1
            return True
        
        if var in searchlist:
            self.LOGGER.debug('%s searchvar: circular search for %r' % (lstr,var))
            level[0] -= 1
            return False
        
        # try for "primitives" like rho_I, rho_O, rho
        if var in self.d and not var in self.EQNS:
            self.LOGGER.debug('%s searchvar: no equation for %r' % (lstr,var))
            level[0] -= 1
            return False
            
        
        searchlist.append(var)
        for k, eqn in enumerate(self.EQNS[var],1):
            self.LOGGER.debug('%s searchvar: looking for %r using (eqn %d of %d) %r' % 
                              (lstr, searchlist,k,len(self.EQNS[var]), eqn))
            variables = re.findall("\$([A-Za-z][A-Za-z0-9_]*)", eqn)
            found = self.isavailable(variables)
            if not found:
                for var2 in variables:
                    found =  self.searchvar(var2, searchlist)
                    if not found:
                        break
                    
            if found:
                self.LOGGER.debug('%s searchvar: found %r from %r' % (lstr,var,eqn))
                self.prepfun(var, self.evalexpr(eqn,variables))
                searchlist.pop(-1)
                level[0] -= 1
                return True
            
            else:
                self.LOGGER.debug('%s searchvar: could not solve %r from %r'
                                  ' because missing %r' %
                                  (lstr,var,eqn,searchlist))
                
                searchlist.pop(-1)
                
            self.LOGGER.debug('%s searchvar: done %d of %d eqns for %r' %
                              (lstr,k,len(self.EQNS[var]),var))
            
        level[0] -= 1
        return False

    #===========================================================================
    #
    # r e s o l v e v a r s
    #
    def resolvevars(self):
        
        if False:
            level = [0]
            searchlist = []
            self.LOGGER.debug('resolvevars: use recursive algorithm')
            for var in ['LTL','CTL','rTL','gTL']:
                found = self.searchvar(var, searchlist=searchlist, level=level)
                if not found:
                    break
                
            self.LOGGER.debug('resolvevars: level %d search %r' % 
                              (level[0], searchlist))
            if found:
                self.LOGGER.debug('resolvevar: succeeded')
            else:
                self.LOGGER.debug('resolvevar: failed')
                
            return
        
        self.LOGGER.info('resolvevars: use iterative algorithm')
        found = '1st loop'
        while found:
            self.Nloop += 1
            found = []
            for var, exprs in self.EQNS.items():
                self.curvar = var
                if self.resolvevar(var, exprs):
                    found.append(var)
                    break # restart the expression loop 

            self.LOGGER.info('resolvevars: resolving loop #%d found %s' % 
                    (self.Nloop, ', '.join(found) if found else '<nothing>'))
            
            # print(found)
            
    #===========================================================================
    #
    # p r e p f u n
    #
    def prepfun(self, var, val, scl=1.):

        self.curvar = var
        
        if var in self.trace:
            self.LOGGER.warning('prepfun: tracing %r --> %r x %r %s' % (
                var,val,scl,self.units[var]))
            
        if isinstance(val,str):
            self.LOGGER.info('prepfun: %r = evalexpr(%r)' % (var,val))
            val = self.evalexpr(val)
            
        if val is None:
            return
        
        elif hasattr(val, '__call__'):
            self.constant = False
            self.available[var] = 'function'
            if scl == 1.:
                self.f[var] = val
            else:
                self.f[var] = lambda x, s=scl: val(x)*s
        
        elif isinstance(val, (list,tuple)):
            if len(val) is not 2:
                raise ValueError(
                    'General_TL: %r : need a list/tuple of 2 floats or '
                    'a list/tuple of 2 list/tuple of floats' % var)
                
            if all([isinstance(v, (list,tuple)) for v in val]):
                isconst = all([v == val[1][0] for v in val[1][1:]])
                if isconst:
                    self.LOGGER.warning('prepfun: tracing %r --> %r (const)' % (var,val))
                    return self.prepfun(var, val[1][0], scl)
                self.constant &= isconst
                self.available[var] = '%d-point' % len(val[0])
#                 print('L = ', np.array(val[0])*SI_scale(self.scaler['L']))
#                 print('f = ', np.real(val[1])*scl)
                self.f[var] = interp1d(np.array(val[0])*SI_scale(self.scaler['L']), 
                                       np.real(val[1])*scl, kind='linear')
                
            elif all([isnummeric(v) for v in val]):
                # assume linear dependance
                isconst = (val[0] == val[1])
                if isconst:
                    return self.prepfun(var, val[0], scl)
                self.constant &= isconst
                self.available[var] = '2-point'
                self.f[var] = (
                    lambda x, 
                           v=val[0].real*scl, 
                           rc=scl*(val[1]-val[0]).real / (
                               1. if self.L is None else self.L):
                           v + x * rc
                    )
                
            else:
                raise ValueError(
                    'General_TL: %r : need a list/tuple of 2 floats or '
                    'a list/tuple of 2 list/tuple of floats' % var)
        
        elif isnummeric(val):
            self.available[var] = 'constant'
            try:
                self.f[var] = lambda x, v=val*scl: v * np.ones(np.array(x).shape)
            except:
                print(var, val, scl)
                raise
            # print(var, self.d[var], self.f[var])
        
        else:
            raise ValueError()
        
        if var in self.trace:
            self.LOGGER.warning('prepfun: tracing %r = [%r] %r' % (
                var, self.available[var], self.f[var](0)))
    
    #===========================================================================
    #
    # T L p r o p s 
    #
    def TLprops(self, x, verbose = False):

        self.Z0TLx = self.f['Z0TL'](x)
        self.rTLx = self.f['rTL'](x)
        self.gTLx = self.f['gTL'](x)
        jwLr = 1j * self.w * self.Z0TLx/(self.f['vr'](x)*c0) + self.rTLx
        jwCg = 1j * self.w / (self.Z0TLx * self.f['vr'](x)*c0) + self.gTLx
        
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
    def eqnVI(self, VI4, x):
        """
        VI4 = [V.real, V.imag, I.real, I.imag]
        """
        # jwLr, jwCg = self.TLprops(x,verbose=verbose)
        
        dVC = self.f['rjwL'](x) * (VI4[2]+1j*VI4[3])
        dIC = self.f['gjwC'](x) * (VI4[0]+1j*VI4[1])
        
        return [dVC.real, dVC.imag, dIC.real, dIC.imag]

    #===========================================================================
    #
    # s o l v e V I
    #
    def solveVI(self, V0 = None, I0 = None):
        """
        solveVI ...
        
        Note: in the telegrapher's equation the currents are flowing out of the
              'sink' port (V0,I0) and into the 'source' port (would be 
              V1=self.U[-1], I1= self.I[-1].
        """
                    
        if V0 is None and I0 is None:
            
            
            if self.xs[0] == self.xs[-1]:
                
                # special case of a 0-length TL section (0j + ... -> cast to complex)
                
                self.U10 = 0j + np.ones(np.array(self.xs).shape)
                self.I10 = 0j + np.zeros(np.array(self.xs).shape)
                
                self.U01 = 0j + np.zeros(np.array(self.xs).shape)
                self.I01 = 0j + np.ones(np.array(self.xs).shape)
                
            else:
                
                sol10 = odeint(self.eqnVI, [1.,0.,0.,0.], self.xs, **self.odeparams)
                self.U10 = sol10[:,0]+1j*sol10[:,1]
                self.I10 = sol10[:,2]+1j*sol10[:,3]
                
                sol01 = odeint(self.eqnVI, [0.,0.,1.,0.], self.xs, **self.odeparams)
                self.U01 = sol01[:,0]+1j*sol01[:,1]
                self.I01 = sol01[:,2]+1j*sol01[:,3]
            
            self.VIt = np.array([[self.U10[-1], self.U01[-1]],
                                 [self.I10[-1], self.I01[-1]]])
    
            self.S = S_from_VI(self.VIt, self.Zbase)
            
            self.solved = True
            
        else:
            if I0 is None:
                I0 = 0.
                
            if V0 is None:
                V0 = 0.
                
            if not self.solved:
                # solve ODE again ...
                VI0 = [V0.real, V0.imag, I0.real, I0.imag]
                
                if self.xs[0] == self.xs[-1]:
                    # special case of a 0-length TL section
                    self.U = (V0 + 0j) * np.ones(np.array(self.xs).shape)
                    self.I = (I0 + 0j) * np.ones(np.array(self.xs).shape)
                
                else:
                    sol = odeint(self.eqnVI, VI0, self.xs, **self.odeparams)
                    self.U = sol[:,0]+1j*sol[:,1]
                    self.I = sol[:,2]+1j*sol[:,3]
                    
            else:
                # much quicker ...
                self.U = V0 * self.U10 + I0 * self.U01
                self.I = V0 * self.I10 + I0 * self.I01
            
            return self.U, self.I
            
    #===========================================================================
    #
    # g e t _ S Z 
    #
    def get_SZ(self, **kwargs):
        """
        Zbase : reference impedance [Ohm] (default: self.Zbase)
        Portnames : (default: ['1', '2']
        """
        Zbase = kwargs.pop('Zbase', self.Zbase)
        Portnames = kwargs.pop('Portnames',['1','2'])

        if kwargs:
            raise ValueError('General_TL.get_SZ: unknown keywords: %r' %
                             [kw for kw in kwargs])
            
        SZ = sc.Scatter(fMHz=self.fMHz, Zbase=Zbase, Portnames=Portnames,
                        S = S_from_VI(self.VIt,Zbase)
                        #TODO: there are plenty of other parameters
                       )
        return SZ
    
    #===========================================================================
    #
    # g e t _ u n i t
    #
    def get_unit(self, var = None):
        if var is None:
            return self.get_unit(self.units)

        elif hasattr(var, '__iter__'):
            return dict((v, self.scaler[v]+self.units[v]) for v  in var)
            
        else:
            return self.scaler[var]+self.units[var]
        
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

    #===========================================================================
    #
    # f u l l s o l u t i o n
    #
    def fullsolution(self, V0, I0):
        
        def findmax(x, y):
            # if the maxima do not occurs at either end, use a quadratic fit
            # based on the neighboring points and find its maximum and location
            km = np.argmax(y)
            if km == 0 or km == (len(x)-1):
                ymax = y[km]
                xmax = x[km]
            else:
                x0, y0 = x[km], y[km]
                ab = np.dot(np.linalg.inv(
                    [[(x[km-1]**2-x0**2), (x[km-1]-x0)],
                     [(x[km+1]**2-x0**2), (x[km+1]-x0)]]),
                    [[y[km-1]-y0], [y[km+1]-y0]])
                a, b = ab[0][0], ab[1][0]
                c = y0 - (a * x0 + b) * x0
                xmax = - b /(2 * a)
                ymax = c + xmax * (b + xmax * a)
            return xmax, ymax
        
        Vs, Is = self.solveVI(V0, I0)
        sol = {'x': self.xs, 'V': Vs, 'I': Is}
        
        sol['xVmax'], sol['Vmax'] = findmax(self.xs, np.abs(Vs))
        sol['xImax'], sol['Imax'] = findmax(self.xs, np.abs(Is))
        
        for var in ['gTL', 'rTL', 'rTLO', 'rTLI', 'ID', 'OD', 'Z0TL', 'LTL',
                    'CTL', 'epsr', 'mur', 'vr', 'etar', 'sigma', 'tand'
                   ]:
            if self.isavailable(var):
                try:
                    sol[var] = self.f[var](0. if self.isconstant(var) else self.xs)
                except:
                    print(var)
                    raise
        
        sol['pc'] = 0.5*sol['rTL'] * np.abs(Is)**2
        sol['pm'] = 0.5*sol['gTL'] * np.abs(Vs)**2   
        sol['pt'] = sol['pc'] + sol['pm']
        
        sol['wLTL'] = self.w * sol['LTL']
        sol['wCTL'] = self.w * sol['CTL']
        
        if 'rTLO' in sol:
            if np.max(sol['rTL']) > 0.:
                sol['pcO'] = sol['rTLO'] / sol['rTL'] * sol['pc']
                sol['xmaxpcO'], sol['maxpcO'] = findmax(self.xs,sol['pcO'])
            else:
                sol['pcO'] = 0. * sol['x']
                sol['xmaxpcO'], sol['maxpcO'] = 0., 0.
            if 'OD' in sol:
                sol['pcSO'] = sol['pcO']/(np.pi * sol['OD'])
                sol['xmaxpcSO'], sol['maxpcSO'] = findmax(self.xs,sol['pcSO'])
                
            
        if 'rTLI' in sol:
            if np.max(sol['rTL']) > 0.:
                sol['pcI'] = sol['rTLI'] / sol['rTL'] * sol['pc']
                sol['xmaxpcI'], sol['maxpcI'] = findmax(self.xs,sol['pcI'])
            else:
                sol['pcI'] = 0. * sol['x']
                sol['xmaxpcI'], sol['maxpcI'] = 0., 0.
            if 'ID' in sol:
                sol['pcSI'] = sol['pcI']/(np.pi * sol['ID'])
                sol['xmaxpcSI'], sol['maxpcSI'] = findmax(self.xs,sol['pcSI'])
            
        sol['Pf'] = 0.5 * np.real(Vs * np.conj(Is))
        sol['Pl'] = np.abs(sol['Pf'][-1] - sol['Pf'][0])
            
        return sol
    
#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    
    import itertools
    import os
    import glob
    from pyRFtk import TouchStoneClass3a as _ts
    from pyRFtk import scatter3a as _sc
    
    import dill
    
    fMHz, Zbase, L = 55.0, 50., 10.000
    
    OD, ID = 228.6, 99.3              # (default diameters and Z0TL)
    Z0TL = eta_0_coax * np.log(OD/ID) # (default resulting Z0TL)
    
    #-------------------------------------------------------------------- case 0
    if False:
        print(
        '# ======================================================================\n'
        '# case 0: check resolving paprameters\n'
        '#\n'
        )
        Kwargs = [{'Z0TL':50.},
                  {'ID':0.0993},
                  {'ID':99.3, 'OD':228.6},
                  {'Z0TL':50., 'rho':0.02},
                  {'Z0TL':50., 'rhoI':0.02, 'OD':228.6},
                  {'Z0TL':50., 'A': 0.0001, 'OD':228.6},
                  {'OD':228.6, 'ID':[[0., 0.5, 1.0],[50.0, 100.0, 150.0]], 'A': 0.0001},
                  {'Z0TL':50., 'rTL': 10.0449372359},
                  {},
                  {'Z0TL': 20.0, # Ohm
                   'qTLI': 1.498,  # /m
                   'qTLO': 1.227,  # /m
                   'rho': 0.02214,  # uOhm.m
                  },
                  {'rho': 0.02214,  # uOhm.m (Cu at 100 degC)
                   'Z0TL': 12.6,    # Ohm
                   'qTLI': 1.346,   # /m     note the feeder inner and outer are not round 
                   'qTLO': 1.172,   # /m     qTLc = rTLc/Rsc takes into account the current distribution
                   'vr': 1.0,       # eps_0
                  },
                  {'Z0TL': 21.00  ,
                   'qTLI': 1.489,   # /m    note the feeder inner and outer are not round 
                   'qTLO': 1.227,   # /m    qTLc = rTLc/Rsc takes into account the current distribution
                   'vr': 1.0000, 
                   'rho': 0.02214,   # uOhm.m (Cu at 100 degC)
                  },
                  {
                   'length': 1.0,
                   'ID': [265,140],
                   'Z0TL': 20.,
                   'vr': 1.0000, 
                   'rho': 0.02214, 
                  }
                 ]
        for kwargs in Kwargs:
            print(kwargs)
            try:
                L = kwargs.pop('length', None)
                myGTL = General_TL(L, fMHz=fMHz, Zbase=Zbase, 
                                   trace=[],
                                   **kwargs)
                print('success !')
                print(myGTL)
                
            except SystemError:
                print('fail')
                raise
                
            print('\n' + '='*80 +'\n')  
        
#         SZ = _sc.Scatter(fMHz=fMHz, Zbase=Zbase)
#         print(SZ.TLproperties( A=0.0001, Z=50.))
#         for v, u in sorted(myGTL.get_unit().items()):
#             print('%10s [%s]' % (v,u))
            
    #-------------------------------------------------------------------- case 1
    if False:
        print(
        '# ======================================================================\n'
        '# case 1: compare with scatter3a with and without losses rho = 2E-2uOhm.m\n'
        '#\n'
        )
        L = 5.0 # m
        Zbase= 20.
        rhos = 0., 2E-2 # uOhm.m
        OD, Z = 228.6, 50.# mm, Ohm
        for rho in rhos:
            print('-'*80,'\nGeneral_TL\n')
            myGTL = General_TL(L, fMHz=fMHz, Zbase=Zbase,
                               rho=rho,
                               # tand=1E-4,
                               OD=OD, 
                               Z=Z,
                               # maxdx=0.01
                               # odeparams={'rtol':1E-12, 'atol':1E-12},
                               # trace=['A']
                              )
            
            print(myGTL, )
            GSZ = myGTL.get_SZ(Zbase=Zbase)
            GSZ.pfmt = ('RI','%+14.10f%+14.10fj')
            print(GSZ)
            
            print('-'*80,'\nScatter\n')
            SZ = sc.Scatter(fMHz=fMHz, Zbase=Zbase, pfmt=GSZ.pfmt)
            A = sc.CoaxAttenuation(fMHz,                           # MKSA units
                                   Z * SI_scale(myGTL.scaler['Z0TL']),
                                   OD * SI_scale(myGTL.scaler['OD']),
                                   rho_inner=rho * SI_scale(myGTL.scaler['rhoI']), 
                                   rho_outer=rho * SI_scale(myGTL.scaler['rhoO']))
            
            print('Z = ',Z * SI_scale(myGTL.scaler['Z0TL']))
            print('OD = ',OD * SI_scale(myGTL.scaler['OD']))
            print('rho_inner = ',rho * SI_scale(myGTL.scaler['rhoI']))
            print('rho_outer = ',rho * SI_scale(myGTL.scaler['rhoO']))
            print('Scatter.CoaxAttenuation =', A,'\n')
            
            SZ.trlAZV(['1','2'], L, Z=Z, A=A)
            print(SZ)
            
            print('-'*80,'\nerror\n')
            errs = 2*np.abs(GSZ.S - SZ.S)/np.abs(GSZ.S + SZ.S)
            print('max error %.3g%%\n' % np.max(np.max(errs*100)))
            # print(isinstance(myGTL, General_TL))
    
    if False:
        print(
        '# ======================================================================\n'
        '# case 2: create touchstone file of lossy (rho = 2E-8 Ohm.m)\n'
        '#         OD 0.2286 ID 0.050 -> 0.150 (frequency 35 -> 60 MHz)\n'
        '#         TL length 1.0m and 10.0m\n'
        )
        Ls = [1.0, 10.0] # m
        OD, IDs = 228.6, [(99.3, 99.3), (50.0, 150.0)] # mm
        rhos = 0., 2E-2 # uOhm.m
        
        fMHzs = np.linspace(35., 60., 51)
        
        for ID, rho, L in itertools.product(IDs,rhos,Ls):
            tsGTL = _ts.TouchStone(tformat='MHZ RI S R %f' % Zbase)
            fname = "GTL_OD=%gmm_ID=%gmm_%gmm_L=%gm_rho=%gOhm.m.s2p" % (
                     OD, ID[0], ID[1], L, rho*1e-6)
            print(fname)
            for fMHz in fMHzs:
                myGTL = General_TL(L, fMHz=fMHz, Zbase=Zbase,
                                   rho=rho,
                                   # tand=1E-4,
                                   OD=OD, 
                                   ID=ID,
                                   # maxdx=0.01
                                   # odeparams={'rtol':1E-12, 'atol':1E-12},
                                   trace=['ID'] if fMHz == 35.0 else [],
                                  )
                tsGTL.Add_Scatter(myGTL.get_SZ())
                
            print('writing %r' % fname)
            
            tsGTL.WriteFile(fname,'Created by General_TL.py\n' + str(myGTL))
    
    #===========================================================================
    #
    # case 3 : compare with MWS
    #
    
    if False:
        print(
        '# ======================================================================\n'
        '#\n'
        '# case 3: compare with MWS\n'
        '#\n'
        )
        os.chdir('../FDe_181217/test cases/')
        for fname in glob.glob('GTL_*.s2p'):
            tsCTL = _ts.TouchStone(filepath = fname)
            tsMWS = _ts.TouchStone(filepath = fname.replace('GTL_','MWS_'))
            
            if True:
                pl.figure(fname, figsize = (13,10))
                for k1 in range(2):
                    for k2 in range(2):
                        pl.subplot(2,2,k1*2+k2+1)
                        fMHzs = tsCTL.Freqs('MHz')
                        Sij = tsCTL.Datas(mtype='S', elfmt='RI', elems=(k1,k2), zref=50.)
                        pl.plot(fMHzs,np.real(Sij),'ro',ms=5,
                                label='Re[S$_{GTL,%d%d}$]' % (k1+1,k2+1))
                        pl.plot(fMHzs,np.imag(Sij),'mo',ms=5,
                                label='Im[S$_{GTL,%d%d}$]' % (k1+1,k2+1))
                        
                        m1, m2 = k1, k2
#                         if 'rho=0Ohm' not in fname:
#                             # MWS appears to swap the port 1 and 2
#                             m1, m2 = 1 - k1, 1- k2
                            
                        fMHzs = tsMWS.Freqs('MHz')
                        Sij = tsMWS.Datas(mtype='S', elfmt='RI', elems=(m1,m2), zref=50.)
                        pl.plot(fMHzs,np.real(Sij),'b',
                                label='Re[S$_{MWS,%d%d}$]' % (m1+1,m2+1))
                        pl.plot(fMHzs,np.imag(Sij),'c',
                                label='Im[S$_{MWS,%d%d}$]' % (m1+1,m2+1))
                
                        pl.xlabel('f [MHz]')
                        pl.ylabel('S$_{%d%d}$' % (k1+1,k2+1))
                        pl.title('S$_{%d%d}$' % (k1+1,k2+1))
                        pl.grid()
                        pl.legend(loc='best')
                        
                pl.suptitle(fname[4:-4].replace('_',' '),fontsize=15)
                # pl.tight_layout(h_pad=1.0, w_pad=1.0, rect=[0.,0.,0.,0.])
                pl.savefig(fname[4:-4]+'.png')
            
            pl.figure(fname[4:-4] + " power")
            
            if 'rho=0Ohm' in fname:
                ys = 10000
            elif 'L=10m' in fname:
                ys = 1000
            else:
                ys = 10000

            
            fMHzs = tsCTL.Freqs('MHz')
            Sij = tsCTL.Datas(mtype='S', elfmt='RI', 
                              elems=[(0,0),(0,1),(1,0),(1,1)], 
                              zref=50.)
            
            D11 = Sij[0] * np.conj(Sij[0]) + Sij[2] * np.conj(Sij[2])
            D22 = Sij[1] * np.conj(Sij[1]) + Sij[3] * np.conj(Sij[3])
            D12 = Sij[0] * np.conj(Sij[1]) + Sij[2] * np.conj(Sij[3])
            
            pl.plot(fMHzs, ys*(1 - np.real(D11)),'mo',label='GTL,ij=11')
#             pl.plot(fMHzs, np.imag(D11),'m.',label='Im[(S$^t$.S$^*$)$_{11}$]')
            pl.plot(fMHzs, ys*(1 - np.real(D22)),'co',label='GTL,ij=22')
#             pl.plot(fMHzs, np.imag(D22),'c.',label='Im[(S$^t$.S$^*$)$_{22}$]')
#             pl.plot(fMHzs, np.real(D12),'ko',label='Re[(S$^t$.S$^*$)$_{12}$]')
#             pl.plot(fMHzs, np.imag(D12),'k.',label='Im[(S$^t$.S$^*$)$_{12}$]')
            
            
            
            fMHzs = tsMWS.Freqs('MHz')
            Sij = tsMWS.Datas(mtype='S', elfmt='RI', 
                              elems=[(0,0),(0,1),(1,0),(1,1)], 
                              zref=50.)
            
            D11 = Sij[0] * np.conj(Sij[0]) + Sij[2] * np.conj(Sij[2])
            D22 = Sij[1] * np.conj(Sij[1]) + Sij[3] * np.conj(Sij[3])
            D12 = Sij[0] * np.conj(Sij[1]) + Sij[2] * np.conj(Sij[3])
            
            pl.plot(fMHzs, ys*(1 - np.real(D11)),'r',label='MWS,ij=11')
#             pl.plot(fMHzs, np.imag(D11),'r--',label='Im[(S$^t$.S$^*$)$_{11}$]')
            pl.plot(fMHzs, ys*(1 - np.real(D22)),'b',label='MWS,ij=22')
#             pl.plot(fMHzs, np.imag(D22),'b--',label='Im[(S$^t$.S$^*$)$_{22}$]')
#             pl.plot(fMHzs, np.real(D12),'g',label='Re[(S$^t$.S$^*$)$_{12}$]')
#             pl.plot(fMHzs, np.imag(D12),'g--',label='Im[(S$^t$.S$^*$)$_{12}$]')
            
            pl.xlabel('f [MHz]')
            pl.ylabel('%d * ( 1 - Re[(S$^t$.S$^*$)$_{ij}$] )' % ys)
            pl.title('%s * ( 1 - Re[(S$^t$.S$^*$)$_{ij}$] )' % ys)
            pl.grid()
            pl.legend(loc='best')
            pl.ylim(bottom=-0.5 ,top=4.0)

            pl.suptitle(fname[4:-4].replace('_',' '))
            pl.tight_layout(rect=[0.,0.,1.,0.95])
            pl.savefig(fname[4:-4]+'_power.png')

    #===========================================================================
    #
    # case 4 : pickle (dill) test
    #
    
    if True:
        print(
        '# ======================================================================\n'
        '#\n'
        '# case 4: pickle (dill) test\n'
        '#\n'
        )
        myGTL = General_TL(fMHz    =     60.000,                  # MHz
                           length  =     10.000,                  # m
                           OD      = [228.6, 228.6],              # mm
                           ID      = [ 50.0, 150.0],              # mm
                           rho     = 0.0224,                      # uOhm.m
                         )        

    pl.show()
    