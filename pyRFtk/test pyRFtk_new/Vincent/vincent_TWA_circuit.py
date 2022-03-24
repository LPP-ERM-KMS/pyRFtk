"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 30 Sep 2021

@author: frederic

Test/evaluate Vincent's TWA circuit

"""
__updated__ = "2021-10-08 14:41:25"

import numpy as np
from matplotlib import pyplot as pl

from pyRFtk2 import rfObject, rfRLC, circuit

import WEST_TWA_7s_4_No_FS_protruding_match as match

#===============================================================================
# print the solution at the nodes for a given frequency and impedance base

def PrintSol(fMHz, Efeed=1., Zbase=match.Zit):
    
    s = f'\nCase: {case} -> {tspath}\n\n'
    s += f'Solution at {fMHz} MHz on {Zbase} Ohm\n'
    tSol = CT.Solution(fMHz *1E6,E={'feed':Efeed}, Zbase=Zbase)
    for w, vs in sorted(tSol.items()):
        s += (f'{w:10s}'
              f'V={vs[0].real:7.3f}{vs[0].imag:+7.3f}j V,  '
              f'I={vs[1].real:7.3f}{vs[1].imag:+7.3f}j A,  '
              f'V+={vs[2].real:7.3f}{vs[2].imag:+7.3f}j V,  '
              f'V-={vs[3].real:7.3f}{vs[3].imag:+7.3f}j V \n')
    
    Vfeed, Ifeed = tSol['feed'][0:2]
    Pin = 0.5 * np.real(Vfeed * np.conj(Ifeed))
    Pout = - 0.5 * np.real(tSol['load'][0] * np.conj(tSol['load'][1]))
    Pfwd = np.abs(tSol["feed"][2])**2/(2*Zbase)
    Prfl = np.abs(tSol["feed"][3])**2/(2*Zbase)
    
    s += '\n'
    s += f'Pin  = {Pin*1E3:7.2f}mW    =   0.5 Re(Vfeed.Ifeed*)\n'
    s += f'Pfwd = {Pfwd*1E3:7.2f}mW    =   0.5 |Vf|**2 / Zbase\n'
    s += f'Prfl = {Prfl*1E3:7.2f}mW    =   0.5 |Vr|**2 / Zbase\n'
    s += f'Pout = {Pout*1E3:7.2f}mW    = - 0.5 Re(Vload.Iload*)\n'
    s += f'Prad = {(Pfwd - Prfl - Pout)*1E3:10.6f}mW   = Pfwd - Prfl - Pout\n'
    s += '\n'

    s += f'# TWA input forward voltage waves on {Zbase} Ohm\n'
    s += 'Vf = [ \n'
    for k in range(1,8):
        Vf = tSol[f'TWA.{k}'][2]
        s += f'   {Vf.real:+.10f}{Vf.imag:+.10f}i , # TWA.{k}\n'
    s += ']\n'
    
    print(s)
    with open(f'TWA_{case}.txt','w') as f:
        f.write(s)
        
    return tSol

#===============================================================================

#
# read the touchstone
#
cases = {
    'new':'n/WEST_TWA_7s_4_No_FS_protruding_sweep.s7p',
    'new1':'n/WEST_TWA_7s_4_No_FS_protruding.s7p',
    'orig':'WEST_TWA_7s_4_No_FS_protruding.s7p'
}

fig1 = pl.figure('all cases')

for case, tspath in cases.items():
    TWA = rfObject(touchstone=tspath)
    fMHzs = TWA.fs/1E6
    
    # build the circuit
    CT = circuit(Zbase=match.Zit)
    CT.addblock('TWA', TWA)
    for k in range(1,8):
        CT.addblock(f'C{k}',rfRLC(Cs=match.__dict__[f'C{k}']*1e-12))
        CT.terminate(f'C{k}.p',Z=0)
        if 1 < k < 7:
            CT.connect(f'C{k}.s', f'TWA.{k}')
        elif k == 7:
            CT.connect(f'C{k}.s', f'TWA.{k}','load')
            CT.terminate('load',Z=match.Zit)
        else:
            CT.connect(f'C{k}.s', f'TWA.{k}','feed')
    
    
    # Plot the reflection coefficient vs frequency
    pl.figure(f'{case} -> {tspath}')
    S11s = CT.getS(fMHzs*1e6)[:,0,0]
    mrk = 'b*' if len(S11s) == 1 else 'bo'
    pl.plot(fMHzs, 20*np.log10(S11s).real, mrk)
    pl.grid()
    pl.xlabel('frequency [MHz]')
    pl.ylabel('S$_{11}$ [dB]')
    pl.title(f'S$_{{11}}$ for Z$_{{it}}$ = {match.Zit:.3f} Ohm')
    pl.suptitle(f'{case} -> {tspath}', fontsize=16)
    
    pl.tight_layout(rect=[0,0,1,0.95])
    
    pl.savefig(f'TWA_{case}.png')
    
    pl.figure(fig1.number)
    print(S11s.shape, len(S11s))
    mrk = '-' if len(S11s) > 1 else 'o'
    pl.plot(fMHzs, 20*np.log10(S11s).real, mrk, label=case)
    pl.grid()
    pl.xlabel('frequency [MHz]')
    pl.ylabel('S$_{11}$ [dB]')
    pl.title(f'S$_{{11}}$ for Z$_{{it}}$ = {match.Zit:.3f} Ohm')
    pl.legend(loc='best')
    pl.tight_layout(rect=[0,0,1,1])
    pl.savefig(f'TWA_all_cases.png')

    fMHz = 53
    
    # print the solution at Zbase = Zit
    tSol = PrintSol(fMHz,Zbase=match.Zit)
    
    # get Vfeed and Ifeed to get the Efeed for another Zbase
    Vfeed, Ifeed = tSol['feed'][0:2]
    
    # print the solution at Zbase = port impedance TWA
    Zbase = 8.59
    Efeed = (Vfeed + Zbase * Ifeed) / 2
    tSol2 = PrintSol(fMHz, Efeed=Efeed, Zbase=Zbase)

    with open(f'TWA_{case}.txt','a') as f:
        f.write('\nCircuit object\n')
        f.write(str(CT))


