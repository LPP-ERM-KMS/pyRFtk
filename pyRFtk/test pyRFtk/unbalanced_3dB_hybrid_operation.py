"""
Created on 12 Aug 2020

@author: frederic

check unbalanced hybrid operation
"""
__updated__ = "2020-08-12 15:13:08"

import numpy as np
import matplotlib.pylab as pl

import pyRFtk.scatter3a as _sc

fMHz     = 51.0

Zbase    = 30.0
H3dB_kdB = 2.72
H3dB_LC  = 1.685
H3dB_LP  = 0.500

SZ0 = _sc.HybridCoupler4Port(fMHz, Zbase, H3dB_kdB, H3dB_LC, H3dB_LP)
SZ0.addsignal('A2')
SZ0.termport('A2', 0.)
SZ0.addsignal('B1')
    

P_A1 = 1.15
P_B2 = 1.15
phi_deg = 80.

V_A1 = np.sqrt(2*Zbase*P_A1)
V_B2 = np.sqrt(2*Zbase*P_B2) * np.exp(1j * phi_deg * np.pi / 180)

VSWR_A1 = []
VSWR_B2 = []
PWR_B1 = []
PWR_A2 = []

VSWRs = np.linspace(1.001,3,21)
for VSWR in VSWRs:
    rho = (VSWR - 1)/(VSWR + 1)

    SZ = SZ0.copy()
    SZ.termport('B1', rho*np.exp(1j))
    
    res = SZ.calcsignals(
            {'A1':'G', 'B2':'G', 'A2':'P-', 'B1':'P-'},
            {'A1': V_A1, 'B2': V_B2},
          )
    VSWR_A1.append((1/np.abs(res['A1'])+1)/(1/np.abs(res['A1'])-1))
    VSWR_B2.append((1/np.abs(res['B2'])+1)/(1/np.abs(res['B2'])-1))
    PWR_A2.append(res['A2'])
    PWR_B1.append(res['B1'])

pl.figure("3dB unbalanced operation", figsize=(8,14))
pl.subplot(2,1,1)
pl.plot(VSWRs, VSWR_A1, 'r', label= 'A1')
pl.plot(VSWRs, VSWR_B2, 'b', label= 'B2')
pl.title('Amplifier VSWRs for A1=%.3fMW and B2=%.3fMW' % (P_A1, P_B2))
pl.xlabel('CSTL VSWR')
pl.ylabel('VSWR')
pl.grid()
pl.legend(loc='best')
pl.xlim(left=1., right=3.)
pl.ylim(bottom=1., top=5.)

pl.subplot(2,1,2)
pl.plot(VSWRs, PWR_A2,'r', label='Cmb Load')
pl.plot(VSWRs, PWR_B1,'g', label='CSTL')
pl.title('Power Combiner Load and CSTL [MW]')
pl.xlabel('CSTL VSWR')
pl.ylabel('Power [MW]')
pl.grid()
pl.legend(loc='best')
pl.xlim(left=1., right=3.)
pl.ylim(bottom=0., top=3.)

# pl.tight_layout()
pl.show()