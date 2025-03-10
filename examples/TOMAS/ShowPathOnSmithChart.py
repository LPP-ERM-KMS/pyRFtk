import csv
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import spatial, optimize
from pyRFtk import rfCircuit, rfTRL, rfRLC, rfObject, rfCoupler
from pyRFtk import plotVSWs
from TomasCircuit import TOMASICCircuit
from alive_progress import alive_bar
from Algos import *
from mpl_smithchart import SmithAxes

logging.basicConfig(filename='/home/arthur/PhD/Projects/TOMAS_ICRH_Auto_Matching/Simulations/Algo/Algo.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
logging.info('starting the software')

plt.rcParams.update({'font.size': 20})

MHz_ = 1e6
pF_ = 1e-12

linewidth=3
resolution = 150
maxcp = 1000e-12
maxcs = 1000e-12
mincp = 7e-12
mincs = 25e-12

#y -axis:
CsVals = np.linspace(mincs,maxcs,resolution)#capacitor is defined in steps
csmin = 160
csmax = 240
cpmin = 300
cpmax = 900
leftmost = (np.abs(CsVals-csmin*1e-12)).argmin()
rightmost = (np.abs(CsVals-csmax*1e-12)).argmin()
CsVals = CsVals[leftmost:rightmost]
CsVals = np.linspace(csmin*1e-12,csmax*1e-12,resolution)#capacitor is defined in steps

#x -axis:
CpVals = np.linspace(mincp,maxcp,resolution)#capacitor is defined in steps
leftmost = (np.abs(CpVals-cpmin*1e-12)).argmin()
rightmost = (np.abs(CpVals-cpmax*1e-12)).argmin()
CpVals = CpVals[leftmost:rightmost]
CpVals = np.linspace(cpmin*1e-12,cpmax*1e-12,resolution)#capacitor is defined in steps

U = []
V = []
def PathLength(X):
    #X is an array of vectors
    pathlength = 0
    for i,point in enumerate(X[:-1]):
        pathlength += np.linalg.norm(point - X[i+1])
    return pathlength
def add_arrow(line, position=None, direction='right', size=35, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )
def MatchPath(algorithm,ct,CpVal,CsVal,FREQ,ProbeIndexA=None,ProbeIndexB=None,ProbeIndexC=None,phasefactor=None):
    global admittances
    global gammas
    admittances = []
    gammas = []
    i = 0
    matched = False
    Path = []
    while ((i < 30000) and not matched):
        
        Path.append(np.array([CpVal,CsVal])/pF_)

        if CsVal > maxcs or CsVal < mincs:
            break
        if CpVal > maxcp or CpVal < mincp:
            break

        ct.set('Cs.Cs',CsVal)
        ct.set('Cp.Cs',CpVal)

        Solution = ct.Solution(f=FREQ, E={'Source': np.sqrt(2*50*5*1E3)}, nodes=['V1toV0','V3toV2'])

        Vf = Solution['V3toV2.V3'][2]
        Vr = Solution['V3toV2.V3'][3]
        #Gamma = Vr/Vf
        #gammas.append(Gamma)
        #phase shift
        c = 299792458
        rho = np.abs(Vr)/np.abs(Vf)
        phi =  np.angle(Vr) - np.angle(Vf)
        Beta = 2*np.pi*FREQ/(c)
        BetaL3 = Beta*2.35
        phi = -1*phi - 2*BetaL3 #phase transform from V3 to load, pyRFtk phi = - my definition of phi
        Gamma = rho*np.exp(1j*(phi))
        gammas.append(Gamma)

        admittances.append((1-Gamma)/(1+Gamma))

        if np.abs(Vr)/np.abs(Vf) <= 0.05:
            matched = True

        EpsG,EpsB = algorithm(Solution,FREQ,ProbeIndexA,ProbeIndexB,ProbeIndexC)

        SPFactor = 1
        if np.abs(Gamma)<0.1:
            SPFactor = 1

        if not phasefactor: phasefactor = 0 #here to test capacitor dependences on Y

        CsVal += EpsG*SPFactor*pF_ #- 0.1*EpsB*1*pF_ # try 
        CpVal += EpsB*SPFactor*pF_

        i += 1

    Path.append(np.array([CpVal,CsVal])/pF_)
    
    return np.array(Path)

FREQ = 25*MHz_

Gammas = []

ct = TOMASICCircuit()

CaVal = 150*pF_
ct.set('Ca.Cs',CaVal)
Matched = []

with alive_bar(len(CsVals)*len(CpVals),title="Making colormap",length=20,bar="filling",spinner="dots_waves2") as bar:
    for j,CsVal in enumerate(CsVals):
        for k,CpVal in enumerate(CpVals):
            ct.set('Cs.Cs',CsVal)
            ct.set('Cp.Cs',CpVal)
            Solution = ct.Solution(f=FREQ, E={'Source': np.sqrt(2*50*5*1E3)}, nodes=['V3toV2'])
            Vf = Solution['V3toV2.V3'][2]
            Vr = Solution['V3toV2.V3'][3]
            Gammas.append(Vr/Vf)
            bar()

X, Y = np.meshgrid(CpVals/pF_,CsVals/pF_) 
Gammas = np.array(Gammas)
Z = np.abs(Gammas).reshape(len(CsVals),len(CpVals)) #put into shape with x length CsVals and y length CpVals
levels = np.linspace(0.0, 1.0, 100)
CS = plt.imshow(Z, cmap="hot",extent=[cpmin,cpmax,csmin,csmax],origin='lower',aspect=(cpmax-cpmin)/(csmax-csmin))
colorbar = plt.colorbar(CS,ticks=np.linspace(0,1,11))
colorbar.set_label("     $\mid\Gamma\mid$",rotation=0)
plt.ylabel("Cs (pF)")
plt.xlabel("Cp (pF)")
plt.ylim(csmin,csmax)
plt.xlim(cpmin,cpmax)

plt.title(f"Ca = {round(CaVal/pF_,2)}pF, Freq= {FREQ/MHz_}MHz")

CpInit = 850*pF_
CsInit = 200*pF_

Path = MatchPath(Algo2V,ct,CpInit,CsInit,FREQ,0,3)
PathPlot = plt.plot(Path[:, 0], Path[:, 1],label="TEXTOR",color="purple",linewidth=linewidth)[0]
print(f"2V Steps: {len(Path)}")
print(f"2V Path length (pF): {PathLength(Path)}pF")
add_arrow(PathPlot)

Path = MatchPath(Algo3V,ct,CpInit,CsInit,FREQ,0,2,3)
PathPlot = plt.plot(Path[:, 0], Path[:, 1],label="3V Algorithm",color="green",linestyle=(5, (5, 3)),linewidth=linewidth)[0]
print(f"3V Steps: {len(Path)}")
print(f"3V Path length (pF): {PathLength(Path)}pF")
add_arrow(PathPlot)

Path = MatchPath(DirCoupler,ct,CpInit,CsInit,FREQ,0,2,3)
PathPlot = plt.plot(Path[:, 0], Path[:, 1],label="Dir coupler Algorithm",color="black",linestyle=(3, (5, 3)),linewidth=linewidth)[0]
print(f"Dir Coupler Steps: {len(Path)}")
print(f"Dir Path length (pF): {PathLength(Path)}pF")
add_arrow(PathPlot)

Path = MatchPath(Algo4V,ct,CpInit,CsInit,FREQ,0,2,3)
PathPlot = plt.plot(Path[:, 0], Path[:, 1],label="4V Algorithm",color="blue",linestyle=(1, (5, 3)),linewidth=linewidth)[0]
print(f"4V Steps: {len(Path)}")
print(f"4V Path length (pF): {PathLength(Path)}pF")
add_arrow(PathPlot)

plt.legend()
#plt.savefig(f'figures/CapSpaceS{CsInit}P{CpInit}.pdf')
plt.show()

ax = plt.subplot(1, 1, 1, projection='smith')

Path = MatchPath(Algo2V,ct,CpInit,CsInit,FREQ,0,3)
plt.plot(gammas,label='TEXTOR',color="purple",linewidth=linewidth)

Path = MatchPath(Algo3V,ct,CpInit,CsInit,FREQ,0,2,3,-1)
plt.plot(gammas,label='3V Algorithm',color="green",linestyle=(5, (5, 3)),linewidth=linewidth)

Path = MatchPath(Algo4V,ct,CpInit,CsInit,FREQ,0,2,3)
plt.plot(gammas,label='4V Algorithm',color="blue",linestyle=(3, (5, 3)),linewidth=linewidth)

Path = MatchPath(DirCoupler,ct,CpInit,CsInit,FREQ,0,2,3)
plt.plot(gammas,label='Dir Coupler Algorithm',color="black",linestyle=(1, (5, 3)),linewidth=linewidth)

plt.legend(loc="upper right")

#plt.savefig(f'figures/SmithS{CsInit}P{CpInit}.pdf')

plt.show()
