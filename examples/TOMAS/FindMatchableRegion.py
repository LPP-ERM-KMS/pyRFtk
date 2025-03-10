import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import spatial, optimize
from pyRFtk import rfCircuit, rfTRL, rfRLC, rfObject, rfCoupler
from pyRFtk import plotVSWs
from scipy.stats import qmc
from TomasCircuit import ArthurMod
from TomasCircuit import AntennaAndCaps
from alive_progress import alive_bar
from Algos import *

"""
Explores the domain and looks at the size of the blue blob and the maximal current running through the capacitors inside the blob
"""

plt.rcParams.update({'font.size': 30})

MHz_ = 1e6
pF_ = 1e-12

global mincs 
mincs = 25e-12
global maxcs 
maxcs = 1000e-12
global mincp 
mincp = 7e-12
global maxcp 
maxcp = 1000e-12

def MinimizeableFunction(CVals,CaVal,FREQ=25*MHz_):
    CpVal = CVals[0]
    CsVal = CVals[1]

    ct.set('Ca.Cs',CaVal)
    ct.set('Cs.Cs',CsVal)
    ct.set('Cp.Cs',CpVal)
    Solution = ct.Solution(f=FREQ, E={'Source': np.sqrt(2*50*5*1E3)}, nodes=['V1toV0','V3toV2'])

    Vf = abs(Solution['V3toV2.V3'][2])
    Vr = abs(Solution['V3toV2.V3'][3])

    Gamma = Vr/Vf

    return Gamma

def Match(algorithm,ct,FREQ):
    matched = False
    CsVals = np.linspace(mincs,maxcs,5000)#capacitor is defined in steps
    CpVals = np.linspace(mincp,maxcp,5000)#capacitor is defined in steps
    j=0
    while not matched and j < 15:
        CsVal = np.random.choice(CsVals)
        CpVal = np.random.choice(CpVals)
        i=0
        while ((i < 10000) and not matched):
            
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

            if abs(Vr)/abs(Vf) <= 0.06:
                matched = True
                break
            ProbeIndexA,ProbeIndexB,ProbeIndexC = 0,1,3
            EpsG,EpsB = algorithm(Solution,FREQ,ProbeIndexA,ProbeIndexB,ProbeIndexC)
            
            SPFactor=40*pF_
            if abs(Gamma) < 0.1:
                SPFactor=1*pF_
            CsVal += SPFactor*EpsG
            CpVal += SPFactor*EpsB

            i += 1
        j+=1
    #Finetuning
    options = {'maxiter':30}
    result = optimize.minimize(MinimizeableFunction, np.array([CpVal,CsVal]),args=(CaVal,FREQ),method='Nelder-Mead',options=options)
    Gamma = result.fun
    CVals = result.x
    #If not found yet:
    k=0
    while ((Gamma > 0.1) and (k<11)):
        options = {'maxiter':80}
        CsVal = np.random.choice(CsVals)
        CpVal = np.random.choice(CpVals)
        result = optimize.minimize(MinimizeableFunction, np.array([CpVal,CsVal]),args=(CaVal,FREQ),method='Nelder-Mead',options=options)
        Gamma = result.fun
        CVals = result.x
        k+=1
    return Gamma,CVals



CaVals = np.linspace(35e-12,600e-12,60)#capacitor is defined in steps
leftmost = (np.abs(CaVals-35e-12)).argmin()
rightmost = (np.abs(CaVals-400-12)).argmin()
CaVals = CaVals[leftmost:rightmost]

FREQ = 28*MHz_
SquareLength = 10e-12

TotalPoints = []
MaxI = []

f = open(f'data{int(FREQ/MHz_)}MHz.csv', 'w')
# create the csv writer
writer = csv.writer(f)

writer.writerow(["CaVal","Middle CsVal","Middle CpVal","points","ICa","ICp","ICs","MaxI"])

ct = ArthurMod()
antennact = AntennaAndCaps()

with alive_bar(len(CaVals)*int(5000*SquareLength/1000e-12)*int(5000*SquareLength/1000e-12),title="Finding #Matched/#Total",length=20,bar="filling",spinner="dots_waves2") as bar: #410,402
    for CaVal in CaVals:
        CsVals = np.linspace(mincs,maxcs,5000)#capacitor is defined in steps
        CpVals = np.linspace(mincp,maxcp,5000)#capacitor is defined in steps

        l_bounds = [7e-12,25e-12]
        u_bounds = [1000e-12,1000e-12]


        #Try to match the system
        Gamma,CVals = Match(DirCoupler,ct,FREQ)

        points = 0
        Iratios = []
        rICas=[]
        rICps=[]
        rICss=[]

        if Gamma < 0.1:
            print(Gamma)
            CpVal,CsVal = CVals[0],CVals[1]
            #antennact.set('Ca.Cs',CaVal)
            #antennact.set('Cs.Cs',CsVal)
            #antennact.set('Cp.Cs',CpVal)
            #print(antennact.getS(FREQ))

            #Define square around point

            leftmost = (np.abs(CpVals-(CpVal-SquareLength/2))).argmin()
            rightmost = (np.abs(CpVals-(CpVal+SquareLength/2))).argmin()
            CpValRegion = CpVals[leftmost:rightmost]

            leftmost = (np.abs(CsVals-(CsVal-SquareLength/2))).argmin()
            rightmost = (np.abs(CsVals-(CsVal+SquareLength/2))).argmin()
            CsValRegion = CsVals[leftmost:rightmost]
            
            
            for CsVal in CsValRegion:
                for CpVal in CpValRegion:
                    CVals = [CpVal,CsVal]
                    Gamma = MinimizeableFunction(CVals,CaVal,FREQ)
                    if Gamma < 0.1:
                        Solution = ct.Solution(f=FREQ, E={'Source': np.sqrt(2*50*1.0*1E3)}, nodes=['V1toV0','V3toV2','Cs','Cp','Ca'])
                        rICa = (1/np.sqrt(2))*np.abs(Solution['Ca.cap'][1])/(138*(1-Gamma))
                        rICp = (1/np.sqrt(2))*np.abs(Solution['Cp.Cp'][1])/(70*(1-Gamma))
                        rICs = (1/np.sqrt(2))*np.abs(Solution['Cs.cap'][1])/(140*(1-Gamma))
                        rICas.append(rICa)
                        rICps.append(rICp)
                        rICss.append(rICs)
                        points += 1
                    bar()
            MaxI.append(np.array([max(rICas),max(rICps),max(rICss)]))
        else:
            print(f"no matchable region found for Ca={CaVal/pF_}pF")
            MaxI.append([0,0,0])

        TotalPoints.append(points)
        writer.writerow([CaVal,CsVal,CpVal,points,MaxI[-1][0],MaxI[-1][1],MaxI[-1][2],max(MaxI[-1])])
        print(f"Ca:{CaVal},Cs:{CsVal},Cp:{CpVal},points:{points},MaxI:{MaxI[-1]}")
        
f.close()
COLOR_POINTS = "#9e32a8"
COLOR_CURRENT = "#3399e6"

fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

MaxI=np.array(MaxI)
ax1.plot(CaVals/pF_, TotalPoints, color=COLOR_POINTS, lw=3)
ax2.plot(CaVals/pF_, MaxI[:,0],linestyle='dashed', color='green', lw=4,label='rICa')
ax2.plot(CaVals/pF_, MaxI[:,1],linestyle='dashed', color='red', lw=4,label='rICp')
ax2.plot(CaVals/pF_, MaxI[:,2],linestyle='dashed', color='blue', lw=4,label='rICs')
ReducedMaxI = []
for MaxIRatio in MaxI:
    ReducedMaxI.append(max(MaxIRatio))
ax2.plot(CaVals/pF_, ReducedMaxI, color=COLOR_CURRENT, lw=4,label='Max(rICa,rICp,rICs)')


ax1.set_xlabel("Ca (pF)")
ax1.set_ylabel("Matchable region", color=COLOR_POINTS)
ax1.tick_params(axis="y", labelcolor=COLOR_POINTS)

ax2.set_ylabel("Current ratio", color=COLOR_CURRENT)
ax2.tick_params(axis="y", labelcolor=COLOR_CURRENT)

plt.title(f"TOMAS ICRH circuit at 1kW {FREQ/MHz_}MHz")
plt.legend()
plt.savefig(f"{FREQ/MHz_}MHzCaScan.pdf")
plt.show()
