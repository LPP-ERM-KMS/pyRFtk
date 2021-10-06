"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 18 Dec 2020

@author: frederic
"""
__updated__ = "2020-12-18 17:44:08"

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
    S = [ (Zk - Z0) @ np.linalg.inv(Zk + Z0)  for Zk in Z]
    
    S = S[0,0,0] if L == 0 else S[:,0,0] if L == 1 else S[0,:,:] if L==2 else S 
    
    return S

