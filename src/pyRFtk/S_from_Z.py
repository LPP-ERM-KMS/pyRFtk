__updated__ = "2021-11-25 10:20:38"

from ._check_3D_shape_ import _check_3D_shape_
import numpy as np

#===============================================================================
#
# S _ f r o m _ Z 
#
def S_from_Z(Z, Zbase=50.):
    """
    returns the scattering matrix S defined on Zbase [Ohm] from an 
    impedance matrix Z
    
    always returns an numpy.ndarray unless the Z is a scalar
    """
    L, Z = _check_3D_shape_(Z)
    
    Z0 = Zbase*np.eye(Z.shape[1])
    S = np.array([ (Zk - Z0) @ np.linalg.inv(Zk + Z0)  for Zk in Z])
    
    S = S[0,0,0] if L == 0 else S[:,0,0] if L == 1 else S[0,:,:] if L==2 else S 
    
    return S

