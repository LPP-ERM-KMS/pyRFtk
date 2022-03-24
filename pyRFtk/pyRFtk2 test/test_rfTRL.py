"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 16 Feb 2021

@author: frederic
"""
__updated__ = "2021-10-06 17:02:08"

import numpy as np
import matplotlib.pyplot as pl
import sys

from pyRFtk2 import rfTRL

#===============================================================================
# test zero frequency response
aTL = rfTRL(Z0TL=[50.,50.], L=1) # should be Zbase and Z0TL 50 Ohm
print(aTL.getS(47.5e6),'\n')
print(aTL)
sys.exit()

#===============================================================================
aTL = rfTRL(sNp='../test data/test_HFSSDesign1_50_Ohm.s2p', Z0TL=30)
print(aTL.sNpdata['Ss'][:,1,1].shape)
print(aTL.sNpdata['Ss'][1,:,:].shape)

# ys.exit(0)

#===============================================================================

Z1, Z2 = 30, 50
f0, E, Zbase = 30E6, {'1':1, '2': -(Z2-Z1)/(Z2+Z1)}, Z1
QWL = 3e8/f0/4
ZQW = np.sqrt(Z1*Z2)
ZDX = 0.05
aTL = rfTRL(L=2 + QWL, dx=288, 
            Z0TL=[[0.00, 1-ZDX, 1+ZDX, 1 + QWL - ZDX, 1 + QWL + ZDX, 2 + QWL],
                  [  Z1,    Z1,   ZQW,           ZQW,            Z2,      Z2]], 
            OD=0.230, rho=2E-8)

aTL = rfTRL(L=4 + 2* QWL, Zbase = 30, dx=0.005, 
            Z0TL=[[0.00, 1-ZDX, 1+ZDX, 1 + QWL - ZDX, 1 + QWL + ZDX, 3 + QWL - ZDX,
                    3 + QWL + ZDX, 3 + 2 * QWL - ZDX, 3 + 2 * QWL + ZDX, 4 + 2 * QWL],
                  [  Z1,    Z1,   ZQW,           ZQW,            Z2,            Z2,
                              ZQW,               ZQW,                Z1,          Z1]], 
            OD=0.230, rho=2E-8)

print(f'ports = {aTL.ports}')
print(f'constant = {aTL.constant}')
print(f'S =\n {aTL.getS(f0, Zbase=Zbase)}')

print(aTL)

# aTL = rfTRL(L=2, Z0TL=[30,30.000001])
# print(f'ports = {aTL.ports}')
# print(f'constant = {aTL.constant}')
# print(f'S =\n {aTL.getS(f0, Zbase=Zbase)}')

xs, (Usw, Isw, E) = aTL.VISWs(f0, E, Zbase)
vmax, loc = aTL.maxV(f0, E, Zbase)
print(f'max = {vmax:.3f}V at {loc}')

Nr = 4
pl.figure('test_rfTL',figsize=(7, 2.5*Nr))
pl.subplot(Nr,1,1)
pl.plot(xs, aTL.TLP.Z0TL(2*np.pi*f0,xs))
pl.grid()
#pl.xlabel('x [m]')
pl.title('Z0TL [Ohm]')

pl.subplot(Nr,1,2)
pl.plot(xs, aTL.TLP.OD(2*np.pi*f0,xs)*np.ones(xs.shape),'r', label='outer')
pl.plot(xs, aTL.TLP.ID(2*np.pi*f0,xs),'b', label='inner')
pl.legend(loc='best')
pl.grid()
# pl.xlabel('x [m]')
pl.title('ID , OD [m]')

pl.subplot(Nr,1,3)
pl.plot(xs,np.abs(Usw))
pl.grid()
pl.axhline(vmax,color='k',alpha=0.5,ls='--')
# pl.xlabel('x [m]')
pl.title('Usw [kV]')

pl.subplot(Nr, 1, 4)
pwr = 0.5 * np.real(Usw * np.conj(Isw))
print(f'pwr.shape = {pwr.shape}')
pl.plot(xs, pwr*1e3)
pl.grid()
pl.xlabel('x [m]')
pl.title('power (at 0m > 0. flows out of the TL) [kW]')

pl.tight_layout()


if True:
    
    bTL = rfTRL(L=20, Z0TL=40, OD=0.230, rho=2E-8)
    bTL.set(L=7.5)
    xs, (Usw, Isw, E) = aTL.VISWs(f0, E, Zbase)
    #pl.plot(xs,np.abs(Usw),'r')
    print()

    print(bTL)

pl.show()
