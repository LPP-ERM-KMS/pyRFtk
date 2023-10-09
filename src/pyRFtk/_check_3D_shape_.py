__updated__ = "2020-12-18 17:43:41"

import numpy as np

#===============================================================================
#
#  _c h e c k _ 3 D _ s h a p e _
#
def _check_3D_shape_(M3):
    
    M3= np.array(M3)
    L = len(M3.shape)
    
    if L == 0 or L == 1 :
        M3 = M3.reshape(M3.shape[0] if L else 1, 1, 1)
        
    elif L == 2 and M3.shape[0] == M3.shape[1]:
        M3 = M3.reshape(1,M3.shape[0],M3.shape[1])
        
    elif L !=3 or M3.shape[1] != M3.shape[2]:
        #
        # expected:
        #    a scalar (float or complex) possibly as numpy.array(<scalar>)
        #    a 1D list or array which is reshaped to (len(list), 1, 1)
        #    a square 2D list or array which is reshaped to (1, N, N), N = len(list)
        #    a list or aray of square 2D lists / arrays of the shape (M, N, N)
        #
        raise ValueError('_check_3d_shape_: Unexpected shape: %r' % (M3.shape))

    return L, M3
    
