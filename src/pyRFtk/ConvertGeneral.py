__updated__ = "2022-02-23 09:58:17"

import numpy as np

from .config import logit, logident
from ._check_3D_shape_ import _check_3D_shape_ 

#===============================================================================
#
#  c o n v e r t _ g e n e r a l
#
def ConvertGeneral(Z2, S1, Z1, type1='V', type2='V'):
    """CovertGeneral(Z2, S1, Z1, type1="V"¸ type2="V")
    
    Converts (an array of N frequencies of) scattering matrice(s) of type1 and
    port impedance(s) Z1 to type2 and port impedance(s) Z2

    Z1, Z2 : port impedance(s)
        scalar : 
            all port impedances are the same for all ports and frequencies
        1D-list/array : 
            length must correspond to the number of ports; ports have different
            impedances but it is the same for all frequencies
        2D-list/array :
            the shape corresponds to (nF, nP)
    
    S1 input S-matri-x/ces
        shape: 
            scalar -> nFreqs = 1, nPorts=1 -> (1,1,1)
            1D-list -> nFreqs=#, nPorts=1  -> (#,1,1)
            2D-list -> nFreqs=1, nPorts=#  -> (1,#,#)
            3D-list -> nFreqs=*, nPorts=#  -> (*,#,#)
            
    type1, type2 : S-matrix types
    
        "P" : power wave Zk.imag==0
            ak=(Vk+Zk.Ik)/(2.Zk**0.5) 
            bk=(Vk-Zk.Ik)/(2.Zk**0.5)
            
        "V" : voltage waveZk.imag==0
            ak=(Vk+Zk.Ik)/2, 
            bk=(Vk-Zk.Ik)/2
            
        "G" : generalized s-matrix [Orfanidis] waveZk.imag!=0
            ak=(Vk+Zk.Ik)/(2.Zk.real**0.5) 
            bk=(Vk-Zk.Ik)/(2.Zk.real**0.5)
        
    
    S2 : converted S-matri-x/ces
        return value has the shape of S1
    """
    
    _debug_ = logit['DEBUG']
    _debug_ and logident('>', printargs=True)
        
    # check S1 ... a single scalar s         -> [ [[s]] ] 1 freq 1x1 S-matrix
    #              a list of 1 element [s]   -> [ [[s]] ] 1 freq 1x1 S-matrix
    #              a 2D list/array     [[S]] -> [ [[S]] ] 1 freq NxN S-matrix
    #              a 3D list/array    [ [[S_1]] ... [[S_n]] ] n freqs NxN S-matrices
    
    L, S1 = _check_3D_shape_(S1)
            
    nF, N = S1.shape[:2]
    
    O = np.ones((nF, N))                  # (nF,N)

    def coeffs(Z, typ):    
        
        # check Z ... a single scalar z   -> [ [z ... z] * nF ] nF freqs N ports
        #             a 1D list/array [Z] -> [ [z_1 ... z_N] * nF ] nF freqs N ports
        #             a 2D list/array [[Z]] -> nF frequencies N ports
        
        Zd = np.array(Z)
        
        if len(Zd.shape) == 0:
            Zd = np.array([[Zd] * N] * nF)
            
        elif len(Zd.shape) == 1 and Zd.shape[0] == N:
            Zd = np.array(Zd.tolist() * nF).reshape(nF,N)
            
        elif len(Zd.shape) != 2 or Zd.shape[0] != nF or Zd.shape[1] != N:
            raise ValueError(
                'convert_general: Z1 or Z2 shape not expected ... %r' % (
                    np.array(Z).shape))
                    
        
        if typ == 'V':
            iZd = 1/Zd                                       # (nF,N)
            QaV, QaI = O/2,  Zd/2                            # (nF,N)
            QbV, QbI = O/2, -Zd/2                            # (nF,N)
            QVa, QVb = O, O                                  # (nF,N)
            QIa, QIb = iZd, -iZd                             # (nF,N)
            
        elif typ == 'G':
            iRd = 1/np.sqrt(np.real(Zd))                     # (nF,N)
            QaV, QaI = iRd/2,  Zd * iRd / 2                  # (nF,N)
            QbV, QbI = iRd/2, -np.conj(Zd) * iRd / 2         # (nF,N)
            QVa, QVb = np.conj(Zd) * iRd, Zd * iRd           # (nF,N)
            QIa, QIb = iRd, -iRd                             # (nF,N)
        
        elif typ == 'P':
            iRd = 1/np.sqrt(Zd)                              # (nF,N)
            QaV, QaI = iRd/2,  Zd * iRd / 2                  # (nF,N)
            QbV, QbI = iRd/2, -Zd * iRd / 2                  # (nF,N) 
            QVa, QVb = Zd * iRd, Zd * iRd                    # (nF,N) 
            QIa, QIb = iRd, -iRd                             # (nF,N)
        
        else:
            raise ValueError('type must be "V", "P" or "G" got %r' % typ)
                 
        return QaV, QaI, QbV, QbI, QVa, QVb, QIa, QIb
    
    *_, SVa, SVb, SIa, SIb = coeffs(Z1, type1)
    TaV, TaI, TbV, TbI, *_ = coeffs(Z2, type2)
    
    # there is no neat way to do this without looping over the frequencies ...
    S2 = []
    for S1k, SVak, SVbk, SIak, SIbk, TaVk, TbVk, TaIk, TbIk in zip(
        S1, SVa, SVb, SIa, SIb, TaV, TbV, TaI, TbI):
        
        V = np.diag(SVak) + np.diag(SVbk) @ S1k
        I = np.diag(SIak) + np.diag(SIbk) @ S1k
        A2 = np.diag(TaVk) @ V + np.diag(TaIk) @ I
        B2 = np.diag(TbVk) @ V + np.diag(TbIk) @ I
        S2.append(B2 @ np.linalg.inv(A2))
    
    S2 = np.array(S2)
    
    if L == 0:
        _debug_ and logident('< L==0')
        return S2[0,0,0]
    
    elif L == 1:
        _debug_ and logident('< L==1')
        return [S2[0,0,0]]
    
    elif L == 2:
        _debug_ and logident('< L==2')
        return S2[0,:,:]
             
    _debug_ and logident('< L==3')
    return S2

    
