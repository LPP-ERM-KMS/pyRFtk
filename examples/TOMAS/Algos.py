import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import spatial, optimize
from pyRFtk import rfCircuit, rfTRL, rfRLC, rfObject, rfCoupler
from pyRFtk import plotVSWs
from TomasCircuit import TOMASICCircuit as TomasCircuit


def DirCoupler(Solution,FREQ,probeindexA=None,probeindexB=None,probeindexC=None,SPfactor=1):

    Vf = Solution['IntoV3.In'][2]
    Vr = Solution['IntoV3.In'][3]

    #phase shift
    c = 299792458
    rho = np.abs(Vr)/np.abs(Vf)
    phi =  np.angle(Vr) - np.angle(Vf)

    Gamma = rho*np.exp(1j*(phi)) 

    Beta = 2*np.pi*FREQ/(c)
    BetaL3 = Beta*2.585
    phi = -1*phi - 2*BetaL3 #phase transform from V3 to load, pyRFtk phi = - my definition of phi

    Gamma = rho*np.exp(1j*(phi))

    u = Gamma.real
    v = Gamma.imag

    EpsG = u**2 + v**2 + u
    EpsB = v

    return -1*SPfactor*EpsG,-1*SPfactor*EpsB 

def Algo3V(Solution,FREQ,probeindexA,probeindexB,probeindexC,SPFactor=1):

    Ai = probeindexA
    Bi = probeindexB
    Ci = probeindexC

    # Get voltages at positions in Measurepoints:
    MeasurePoints = np.array([0.235,0.895,1.69,2.35])
    V = np.zeros(len(MeasurePoints))

    #constants:
    Beta = 2*np.pi*FREQ/(3*(10**8))
    S = np.sin(2*Beta*MeasurePoints) #array
    C = np.cos(2*Beta*MeasurePoints) #array
    SABBC = (S[Ai] - S[Bi])/(S[Bi] - S[Ci])
    CABBC = (C[Ai] - C[Bi])/(C[Bi] - C[Ci])

    #Make array of voltages on the various points:
    V[0] = abs(Solution['V1toV0.V0'][0])
    V[1] = abs(Solution['V1toV0.V1'][0])
    V[2] = abs(Solution['V3toV2.V2'][0])
    V[3] = abs(Solution['V3toV2.V3'][0]) 

    Vf = abs(Solution['V3toV2.V3'][2])
    Vr = abs(Solution['V3toV2.V3'][3])

    Gamma = Vr/Vf

    Vs = (V/Vf)**2 - 1

    udenom = 2*(C[Bi]-C[Ci])*(CABBC - SABBC)
    vdenom = 2*(S[Bi]-S[Ci])*(CABBC - SABBC)

    u = ((Vs[Ai] - Vs[Bi]) - (Vs[Bi] - Vs[Ci])*SABBC)/udenom
    v = ((Vs[Ai] - Vs[Bi]) - (Vs[Bi] - Vs[Ci])*CABBC)/vdenom
    
    EpsG = u**2 + v**2 + u
    EpsB = v

    return -1*EpsG,-1*EpsB

def Algo4V(Solution,FREQ,ProbeIndexA=None,ProbeIndexB=None,ProbeIndexC=None):
    ######################################
    # Calculate new values of Cs and Cp  #
    ######################################

    # Get voltages at positions in Measurepoints:
    MeasurePoints = np.array([0.235,0.895,1.69,2.35])
    V = np.zeros(len(MeasurePoints))

    #constants:
    lam = FREQ/(3*(10**8))

    beta = 2*np.pi*lam
    S = np.sin(2*beta*MeasurePoints) #array
    C = np.cos(2*beta*MeasurePoints) #array

    BigS = (S[0] - S[1])/(S[2] - S[3])
    BigC = (C[0] - C[1])/(C[2] - C[3])

    V[0] = abs(Solution['V1toV0.V0'][0])
    V[1] = abs(Solution['V1toV0.V1'][0])
    V[2] = abs(Solution['V3toV2.V2'][0])
    V[3] = abs(Solution['V3toV2.V3'][0]) 

    Vf = abs(Solution['V3toV2.V3'][2])
    Vr = abs(Solution['V3toV2.V3'][3])

    Vs = (V**2)/(Vf**2) - 1

    u = (1/2)*((Vs[0] - Vs[1]) - (Vs[2] - Vs[3])*BigS)/((C[2] - C[3])*(BigC - BigS))
    v = (1/2)*((Vs[0] - Vs[1]) - (Vs[2] - Vs[3])*BigC)/((S[2] - S[3])*(BigC - BigS))

    EpsB = v
    EpsG = u**2 + v**2 + u

    return -1*EpsG,-1*EpsB

def Algo2V(Solution,FREQ,ProbeIndexA,ProbeIndexB,ProbeIndexC=None):
    ######################################
    # Calculate new values of Cs and Cp  #
    ######################################

    A = ProbeIndexA
    B = ProbeIndexB

    # Get voltages at positions in Measurepoints:
    MeasurePoints = np.array([0.235,0.895,1.69,2.35])
    V = np.zeros(len(MeasurePoints))

    #constants:
    beta = 2*np.pi*FREQ/(3*(10**8))
    S = np.sin(2*beta*MeasurePoints) #array
    C = np.cos(2*beta*MeasurePoints) #array
    BigS = (S[0] - S[1])/(S[2] - S[3])
    BigC = (C[0] - C[1])/(C[2] - C[3])


    #Make array of voltages on the various points:
    V[0] = np.abs(Solution['V1toV0.V0'][0])
    V[1] = np.abs(Solution['V1toV0.V1'][0])
    V[2] = np.abs(Solution['V3toV2.V2'][0])
    V[3] = np.abs(Solution['V3toV2.V3'][0])

    Vf = abs(Solution['V3toV2.V3'][2])
    Vr = abs(Solution['V3toV2.V3'][3])

    Gamma = np.abs(Vr/Vf)

    EpsB = (V[A] - Vf)*S[B] - (V[B] - Vf)*S[A]
    EpsG = (V[A] - Vf)*C[B] - (V[B] - Vf)*C[A]

    x = np.array([EpsG,EpsB])
    x = x/np.linalg.norm(x)
    EpsG = x[0]
    EpsB = x[1]

    return -1*EpsG,EpsB
