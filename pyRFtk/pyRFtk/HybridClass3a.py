################################################################################
#                                                                              #
# Copyright 2015, 2016, 2017                                                   #
#                                                                              #
#                   Laboratory for Plasma Physics                              #
#                   Royal Military Academy                                     #
#                   Brussels, Belgium                                          #
#                                                                              #
#                   Fusion for Energy                                          #
#                                                                              #
#                   EFDA / EUROfusion                                          #
#                                                                              #
# Author : frederic.durodie@rma.ac.be                                          #
#                          @telenet.be                                          #
#                          @ccfe.ac.uk                                         #
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
Created on 17 Oct 2017

@author: frederic.durodie@gmail.com

+-------------+----------------------------------------------------------------+
| Date        | Comment                                                        |
+=============+================================================================+
| yyyy-MMM-dd | Comment                                                        |
+-------------+----------------------------------------------------------------+

"""
import numpy as np
from scipy.constants import speed_of_light as c0

from pyRFtk import scatter3a as sc
from pyRFtk import TouchStoneClass3a as ts

#===============================================================================

class Hybrid():
    def __init__(self, **kwargs):
        
        self.fMHz0 = kwargs.pop('fMHz0', None)
        
        self.OD = kwargs.pop('OD',0.230)
        self.Z0 = kwargs.pop('Z0', 30.)

        self.rho = kwargs.pop('rho', 0.)
        self.rho_i = kwargs.pop('rho_i', self.rho)
        self.rho_o = kwargs.pop('rho_o', self.rho)

        self.type = kwargs.pop('type', '2branch')
        self.fmt = kwargs.pop('fmt','MHZ S RI R %.3f' % self.Z0)
        
        if kwargs:
            raise ValueError('Hybrid: unknown arguments %r' % kwargs)
        
         
    #===========================================================================
    
    def __str__(self):
        s  = 'Hybrid: \n'
        s += ' f0 = %r MHz \n' % self.fMHz0
        s += ' type = %r \n' % self.type
        s += ' Z0 = %r Ohm\n' % self.Z0
        s += ' OD = %r m\n' % self.OD
        s += ' rho_inner = %r micro.Ohm.cm\n' % (self.rho_i*1E8)
        s += ' rho_outer = %r micro.Ohm.cm\n' % (self.rho_o*1E8)
        return s
        
    #===========================================================================
    
    def GetSZ(self,fMHz):
        
        return {'ideal': self.Hideal,
                'coupled':self.Hcoupled,
                '1branch':self.H1branch,
                '2branch':self.H2branch
               }[self.type](fMHz)

    #===========================================================================
    
    def Hideal(self, fMHz):
        """
        """
        #FIXME: check matrix
        SZ = sc.Scatter(fMHz=fMHz, Zbase=self.Z0,
                        Portnames=['F','Q','T','I'],
                        S = np.array([[ 0+0j,-1+0j, 0+1j, 0+0j],
                                      [-1+0j, 0+0j, 0+0j, 0+1j],
                                      [ 0+1j, 0+0j, 0+0j,-1+0j],
                                      [ 0+0j, 0+1j,-1+0j, 0+0j]])/np.sqrt(2)
                       )
        return SZ
    
    #===========================================================================
    
    def Hcoupled(self, fMHz):
        #FIXME: put coupler formulas
        return self.Hideal(fMHz)
    
    #===========================================================================
    
    def H2branch(self, fMHz):
        """
          F                               Q
              FC=======CF   QC=======CQ
          FI             FCQ             TQ
           |            _FCQ_            |
           |              ||             |
           |              ||             |
           |              ||             |
           |            _ICT_
          IF             ICT             QT
              IC=======CI   TC=======CT
          I                               T
        """        
        ZL = self.Z0/np.sqrt(2)
        ZH = self.Z0/(np.sqrt(2)-1)
        Lqw =  c0 / (4E6 * self.fMHz0)    
        
        SZ = sc.Scatter(fMHz=fMHz, Zbase=self.Z0)
        AL = sc.CoaxAttenuation(fMHz, ZL, self.OD, self.rho_i, self.rho_o)
        AH = sc.CoaxAttenuation(fMHz, ZH, self.OD, self.rho_i, self.rho_o)
        
        SZ.AddTL(['FC','CF'], Lqw, A=AL, Z=ZL)
        SZ.AddTL(['QC','CQ'], Lqw, A=AL, Z=ZL)
        SZ.AddTL(['IC','CI'], Lqw, A=AL, Z=ZL)
        SZ.AddTL(['TC','CT'], Lqw, A=AL, Z=ZL)
        SZ.AddTL(['_FCQ_','_ICT_'], Lqw, A=AL, Z=ZL)
        
        SZ.AddTL(['FI','IF'], Lqw, A=AH, Z=ZH)
        SZ.AddTL(['TQ','QT'], Lqw, A=AH, Z=ZH)
            
        SZ.joinports('FI','FC','F')
        SZ.joinports('IC','IF','I')
        SZ.joinports('CQ','TQ','Q')
        SZ.joinports('CT','QT','T')
        
        SZ.joinports('CF','QC','FCQ')
        SZ.joinports('CI','TC','ICT')
        
        SZ.connectports('_FCQ_', 'FCQ')
        SZ.connectports('_ICT_', 'ICT')
        
        SZ.sortports(['F','Q','T','I'])
        
        return SZ

    #===========================================================================
    
    def H1branch(self, fMHz):
        """
          F                Q
              FQ=======QF
          FI              TQ
           |               |
           |               |
           |               |
           |               |
           |               |
          IF              QT
              IT=======TI 
          I                 T
        """        
        ZL = self.Z0/np.sqrt(2)
        Lqw =  c0 / (4E6 * self.fMHz0)    
        
        SZ = sc.Scatter(fMHz=fMHz, Zbase=self.Z0)
        A0 = sc.CoaxAttenuation(fMHz, self.Z0, self.OD, self.rho_i, self.rho_o)
        AL = sc.CoaxAttenuation(fMHz, ZL, self.OD, self.rho_i, self.rho_o)
        
        SZ.AddTL(['FQ','QF'], Lqw, A=AL, Z=ZL)
        SZ.AddTL(['IT','TI'], Lqw, A=AL, Z=ZL)
        
        SZ.AddTL(['FI','IF'], Lqw, A=A0, Z=self.Z0)
        SZ.AddTL(['TQ','QT'], Lqw, A=A0, Z=self.Z0)
            
        SZ.joinports('FI','FQ','F')
        SZ.joinports('II','IF','I')
        SZ.joinports('QF','TQ','Q')
        SZ.joinports('TI','QT','T')
                
        SZ.sortports(['F','Q','T','I'])
        
        return SZ
    
    #===========================================================================
    
    def asTSF(self, fMHzs, fpath=None):
        """
        """        
        TS = ts.TouchStone(tformat=self.fmt)
        for fMHz in fMHzs:
            SZ = self.GetSZ(fMHz)
            TS.Add_Scatter(SZ)

        if fpath:
                TS.WriteFile(fpath,str(self))
        
        return TS
    
#===============================================================================

if __name__ == '__main__':

    import matplotlib.pyplot as pl

    dB = lambda x: 20*np.log10(np.abs(x))
    
    Hyb = Hybrid(type='2branch',fMHz0=55.,Z0=30.,rho=0.,OD=0.23)
    print(Hyb)
    
    fMHzs = np.linspace(30,80,1001)
    H = Hyb.asTSF(fMHzs,'hybrid_2branch_f0_55MHz_OD_230mm_rho_0E-8_Ohm.m(2).s4p')

    Sk1s = H.Datas(freqs=fMHzs, unit='MHz', mtype='S', elfmt='RI', 
                   elems=[(0,0),(1,0),(2,0),(3,0)], zref=30)
        
    pl.figure()
    pl.plot(fMHzs, dB(Sk1s[0]), 'r', label='S_11', lw=2)
    pl.plot(fMHzs, dB(Sk1s[1]), 'g', label='S_21', lw=2)
    pl.plot(fMHzs, dB(Sk1s[2]), 'b', label='S_31', lw=2)
    pl.plot(fMHzs, dB(Sk1s[3]), 'c', label='S_41', lw=2)
    
    pl.grid()
    pl.legend(loc='best')
    pl.ylim(-60,0)
    pl.title('S-matix coeffcients [dB]')
    pl.xlabel('frequency [fMHz]')
    
    
    pl.show()
    
    
    
              
    