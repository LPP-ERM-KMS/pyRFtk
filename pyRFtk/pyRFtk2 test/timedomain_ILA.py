"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 17 Aug 2021

@author: frederic

Attempt to simulate the ILA RDLs 
"""
__updated__ = "2021-10-07 09:41:52"

import os

# set the number of threads for the OpenBLAS libs
#   (needs to happen before importing numpy)

os.environ['OPENBLAS_NUM_THREADS']='{:d}'.format(2)

import numpy as np
import matplotlib.pyplot as pl

from pyRFtk2 import circuit
# from pyRFtk2 import junction
from pyRFtk2 import rfTRL
from pyRFtk2 import rfRLC

# set the frequency sampling
dfMHz = 1
maxfMHz = 8191.
fMHzs = np.arange(dfMHz,maxfMHz+dfMHz/2,dfMHz).tolist()
# print(fMHzs)

Lnwa = rfTRL(Zbase=50,Z0TL=50,ports=['nwa','feed'],L=0.5)

def ILA():
    # the simpliied ILA circuit should look like
                               
    #          :      :     :      : 
    # Gnd ---- : -||- : -+  :      : 
    #          :      :  +- : ---- : ------- : ----+-------------- Gnd
    # Gnd ---- : -||- : -+  :      :               |
    #   straps    Cs     T   Low Z   30 Ohm        +
    
    # build the circuit,
    ct = circuit(Zbase=50)
    
    lS1, ZS1, l1a, Z1a, l1b, Z1b, C1 = 0.30, 45, 0.25, 30, 0.15, 20, 80e-12
    lS2, ZS2, l2a, Z2a, l2b, Z2b, C2 = 0.35, 45, 0.30, 30, 0.15, 20, 80e-12
    
    S1  = rfTRL(ports=['Gnd','S'], Zbase=50., Z0TL=ZS1, L=lS1)
    F1a = rfTRL(ports=['S' ,'Fa'], Zbase=50., Z0TL=Z1a, L=l1a)
    F1b = rfTRL(ports=['Fa','Fb'], Zbase=50., Z0TL=Z1b, L=l1b)
    C1  = rfRLC(ports=['Fb','T' ], Zbase=50., Cs=C1, Ls=20e-9)
    
    S2  = rfTRL(ports=['Gnd','S'], Zbase=50., Z0TL=ZS2, L=lS2)
    F2a = rfTRL(ports=['S' ,'Fa'], Zbase=50., Z0TL=Z2a, L=l2a)
    F2b = rfTRL(ports=['Fa','Fb'], Zbase=50., Z0TL=Z2b, L=l2b)
    C2  = rfRLC(ports=['Fb','T' ], Zbase=50., Cs=C2, Ls=20e-9)
    
    L1 = rfTRL(ports=['T','1'], Zbase=50., Z0TL=9., L=1.6)
    L2 = rfTRL(ports=['1','A'], Zbase=50., Z0TL=30., L=3.2)
    
    ct.addblock('S1', S1)
    ct.terminate('S1.Gnd', Z=0)
    ct.addblock('F1a', F1a)
    ct.connect('S1.S','F1a.S')
    ct.addblock('F1b', F1b)
    ct.connect('F1a.Fa','F1b.Fa')
    ct.addblock('C1',C1)
    ct.connect('F1b.Fb','C1.Fb')
    
    ct.addblock('S2', S2)
    ct.terminate('S2.Gnd', Z=0)
    ct.addblock('F2a', F2a)
    ct.connect('S2.S','F2a.S')
    ct.addblock('F2b', F2b)
    ct.connect('F2a.Fa','F2b.Fa')
    ct.addblock('C2',C2)
    ct.connect('F2b.Fb','C2.Fb')
    
    ct.addblock('L1', L1)
    ct.connect('C1.T','C2.T','L1.T')
    
    ct.addblock('L2',L2)
    ct.connect('L1.1','L2.1')
    
    print(ct)
    return ct

def simple_test():
    ct = circuit(Zbase=50.)
    L1 = rfTRL(Zbase=50, Z0TL=60, L=1)

    ct.addblock('L1',L1,ports=['sc','1'])
    ct.terminate('L1.sc', Z=0)
    
    C1 = rfRLC(Cs=300e-12)
    ct.addblock('C1', C1)
    ct.connect('C1.s','L1.1')
    
    ct.addblock('nwa', Lnwa)
    ct.connect('C1.p','nwa.feed')
    print(ct)
    return ct

# compute the reflection coeff
ct = simple_test()

S11s = []
for fMHz in fMHzs:
    S11s.append(ct.getS(fMHz*1E6)[0,0])
    
# insert the zero frequency point
fMHzs.insert(0,0)
if np.abs(S11s[0] - 1) > np.abs(S11s[0] + 1):
    S11s.insert(0,-1+0j)
else:
    S11s.insert(0,+1+0j)


# plot the RC
nrow=1
fig, axs = pl.subplots(nrow, 1, squeeze=True, figsize=(6,1 + 3*nrow))
if not hasattr(axs,'__iter__'):
    axs = [axs]
    
if nrow == 3:
    pl.sca(axs[1])
    pl.plot(fMHzs,np.abs(S11s))
    pl.grid()
    pl.title('Circuit RC')
    pl.ylabel('|S$_{1,1}$|')
    pl.sca(axs[2])
    pl.plot(fMHzs,np.angle(S11s))
    pl.grid()
    pl.ylabel('$\phi$ S$_{1,1}$')
    pl.xlabel('frequency [MHz]')

print('... done circuit.')

# prepare ifft 
dt = 1e-6/fMHzs[-1]
maxt = 1e-6/dfMHz
ts = np.linspace(0,maxt,len(fMHzs))

sig = np.fft.irfft(S11s, len(fMHzs))

# plot the time domain signal
pl.sca(axs[0])
pl.plot(ts*3e8/2,sig)
pl.grid()
pl.title('time domain')
pl.ylabel('signal')
#pl.xlabel('time [$\mu$s]')
pl.xlabel('dist @ c0 [m]')
# pl.xlim(left=0, right=10)

pl.tight_layout()
pl.show()

