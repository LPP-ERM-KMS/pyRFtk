__updated__ = "2020-12-18 17:56:43"

import numpy as np
from . import _check_3D_shape_

#===============================================================================
#
# S _ f r o m _ Y 
#
def S_from_Y(Y, Zbase=50.):
    """
    returns the scattering matrix S defined on Zbase [Ohm] from an 
    admittance matrix Y
    
    always returns an numpy.ndarray unless the Z is a scalar
    """
    L, Y = _check_3D_shape_(Y)
    
    Y0 = np.eye(Y.shape[1])/Zbase
    S = [ (Y0 - Yk) @ np.linalg.inv(Y0 + Yk)  for Yk in Y]
    
    S = S[0,0,0] if L == 0 else S[:,0,0] if L == 1 else S[0,:,:] if L==2 else S 
    
    return S

