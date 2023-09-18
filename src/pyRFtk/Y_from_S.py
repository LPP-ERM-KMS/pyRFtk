"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 18 Dec 2020

@author: frederic
"""
__updated__ = "2022-03-30 13:41:41"

from ._check_3D_shape_ import _check_3D_shape_
import numpy as np

#===============================================================================
#
# Y _ f r o m _ S 
#
def Y_from_S(S, Zbase=50.):
    """
    returns the scattering matrix S defined on Zbase [Ohm] from an 
    admittance matrix Y
    
    always returns an numpy.ndarray unless the Z is a scalar
    """
    L, S = _check_3D_shape_(S)
    
    I = np.eye(S.shape[1])
    Z = np.array([np.linalg.inv(I + Sk)  @ (I - Sk) / Zbase for Sk in S])
    
    Z = Z[0,0,0] if L == 0 else Z[:,0,0] if L == 1 else Z[0,:,:] if L==2 else Z 
    
    return Z

