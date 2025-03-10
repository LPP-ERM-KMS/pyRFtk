import csv
from deco import concurrent
from deco import synchronized
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import spatial, optimize
from pyRFtk import rfCircuit, rfTRL, rfRLC, rfObject, rfCoupler
from pyRFtk import plotVSWs
from TomasCircuit import ArthurMod as TomasCircuit
from alive_progress import alive_bar
from Algos import *
import multiprocessing


MHz_ = 1e6
pF_ = 1e-12
resolution = 6
maxcs = 1000e-12
mincs = 7e-12
mincp = 35e-12
maxcp = 1000e-12

FREQ = 25*MHz_

plt.rcParams.update({'font.size': 20})


CsVals = np.linspace(mincs,maxcs,resolution)#capacitor is defined in steps
StartCsVals = np.linspace(300e-12,800e-12,resolution)#capacitor is defined in steps
#leftmost = (np.abs(CsVals-120e-12)).argmin()
#rightmost = (np.abs(CsVals-170e-12)).argmin()
#CsVals = CsVals[leftmost:rightmost]

CpVals = np.linspace(mincp,maxcp,resolution)#capacitor is defined in steps
StartCpVals = np.linspace(400e-12,maxcp,resolution)#capacitor is defined in steps
#leftmost = (np.abs(CpVals-100e-12)).argmin()
#rightmost = (np.abs(CpVals-750e-12)).argmin()
#CpVals = CpVals[leftmost:rightmost]

U = []
V = []

def MatchAble(algorithm,ct,CpVal,CsVal,FREQ,ProbeIndexA=None,ProbeIndexB=None,ProbeIndexC=None,phasefactor=None):
    i = 0
    matched = False
    while ((i < 5000) and not matched):
        
        if CsVal > maxcs or CsVal < mincs:
            break
        if CpVal > maxcp or CpVal < mincp:
            break

        ct.set('Cs.Cs',CsVal)
        ct.set('Cp.Cs',CpVal)

        Solution = ct.Solution(f=FREQ, E={'Source': np.sqrt(2*50*5*1E3)}, nodes=['V1toV0','V3toV2'])

        Vf = Solution['V3toV2.V3'][2]
        Vr = Solution['V3toV2.V3'][3]
        Gamma = Vr/Vf

        if abs(Vr/Vf) <= 0.1:
            matched = True
        if phasefactor:
            EpsG,EpsB = algorithm(Solution,FREQ,ProbeIndexA,ProbeIndexB,ProbeIndexC,phasefactor)
        else:
            EpsG,EpsB = algorithm(Solution,FREQ,ProbeIndexA,ProbeIndexB,ProbeIndexC)
        
        CsVal += EpsG*1*pF_
        CpVal += EpsB*1*pF_

        i += 1

    return i,matched


Gammas = []
ratios = []
Steps = []
avg_steps = []

ct = TomasCircuit()

CaVal = 150*pF_
ct.set('Ca.Cs',CaVal)
num = 0
tot = 0

SPfactors1 = np.linspace(4,8,8)
SPfactors2 = np.linspace(8,63,25)
SPfactors = np.concatenate((SPfactors1,SPfactors2))

with alive_bar(len(SPfactors)*resolution*resolution) as bar:
    for SPfactor in SPfactors:
        avg_step = 0
        num =0
        tot =0
        for j,CsVal in enumerate(StartCsVals):
            for k,CpVal in enumerate(StartCpVals):
                ct.set('Cs.Cs',CsVal)
                ct.set('Cp.Cs',CpVal)
                Solution = ct.Solution(f=FREQ, E={'Source': np.sqrt(2*50*5*1E3)}, nodes=['V0toV1','V3toV2'])
                steps,matched = MatchAble(DirCoupler,ct,CpVal,CsVal,FREQ,0,2,3,SPfactor)
                tot+=1
                avg_step += steps
                if matched:
                    num+=1
                bar()
        ratios.append(num/tot)
        avg_steps.append(avg_step/tot)
        Steps.append(avg_steps)
        print(f"ratio:{ratios[-1]}")
        print(f"average number of steps:{Steps[-1]}")

#COLOR_POINTS = "#9e32a8"
COLOR_POINTS = "#3399e6"
COLOR_CURRENT = "#9e32a8"
#COLOR_CURRENT = "#3399e6"

fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(SPfactors, avg_steps, color=COLOR_POINTS, lw=3)
ax2.plot(SPfactors, ratios, color=COLOR_CURRENT, lw=4)

ax1.set_xlabel("S=P (pF)")
ax1.set_ylabel("average number of steps", color=COLOR_POINTS, fontsize=20)
ax1.tick_params(axis="y", labelcolor=COLOR_POINTS)

ax2.set_ylabel("ratio of matchability", color=COLOR_CURRENT, fontsize=20)
ax2.tick_params(axis="y", labelcolor=COLOR_CURRENT)

plt.show()
