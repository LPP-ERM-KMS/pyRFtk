__updated__ = "2021-02-17 11:18:02"

import numpy as np

##===============================================================================
#
# S _ f r o m _ V I 
#
def S_from_VI(M, Z0=30.):
    """
    returns the S on reference imedance Z0 from the VI transformation matrix M :

    .. code-block:: 
                      +-------------------+
                      |                   |
                I2    |  [V2]       [V1]  |         I1
            ----->----+  [  ] = M . [  ]  +--------->-----
                 ^    |  [I2]       [I1]  |         ^
             V2  |    |                   |         | V1
                 |    +-------------------+         |
                 
    note S is defined for an oposite sign of I1 shown above (i.e. standard)
    """
    if not isinstance(M, np.ndarray):
        M = np.array(M)
        
    if M.shape != (2,2):
        raise ValueError('S_from_VI is only possible for 2x2 M matrices')
    
    S2 = np.array([[      1            ,     + Z0         ],
                   [ M[0,0]-Z0*M[1,0]  , M[0,1]-Z0*M[1,1] ]])
    
    S1 = np.array([[      1            ,     - Z0         ],
                   [ M[0,0]+Z0*M[1,0]  , M[0,1]+Z0*M[1,1] ]])
    
    S = np.dot(S2,np.linalg.inv(S1))
    
    return S
    
