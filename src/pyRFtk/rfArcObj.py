__updated__ = '2022-04-13 13:17:12'

if __name__ == '__main__':
    import sys
    print('running the test module "../pyRFtk2 test/pyRFtk2_tests.py"')
    sys.path.insert(0, '../pyRFtk2 test')
    import pyRFtk2_tests                       # @UnresolvedImport @UnusedImport
    sys.exit(0)

degSign = u'\N{DEGREE SIGN}'
    
import numpy as np

from . import rfBase
from .config import logit, logident

#===============================================================================
#
# r f A r c O b j
#
class rfArcObj(rfBase):
    """
    """
    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, *args, **kwargs):
        """rfArcObj(
            [Larc],   # if supplied the arc inductance in [H]
            [Larc=],  # if Larc arg not supplied then as Larc arg; default to +np.inf [H]
            [Id=],    # Identifier; default to rfArcObj_(::ALPHANUM::<6>)
            
            [S=],     # the S-matrix; default to np.zeros((len(ports),len(ports)))
            [ports=], # portnames; default to [f'{k}' for k in range(1,len(S)+1)]
            [Zbase=]  # reference impedance; default to 50 [Ohm]
           )
            
            1-port rfArcObj
        """
        
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', True)
        
        if not hasattr(self,'args'):
            self.args = args
            
        if not hasattr(self,'kwargs'):
            self.kwargs = kwargs.copy()
            
        _debug_ and logident(f'{kwargs}')
        
        super().__init__(**kwargs)
        
        _debug_ and logident(f'{kwargs}')
        
        Larc, args = (args[0], args[1:]) if len(args) else (+np.inf, ())
        self.Larc = kwargs.pop('Larc', Larc)
        if args:
            _debug_ and logident(f'{len(args)} unprocessed args')
        
        self.f, self.S = None, None
        
        self.attrs = ['Larc']
        
        _debug_ and logident('<')
        
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, full=0):
        s  = super().__str__()
        if full:
            s += f'\n| Larc = {self.Larc * 1e9} [nH]\n^'
        return s
   
    #===========================================================================
    #
    # g e t S
    #
    def get1S(self, f):
        
        jwL = 2j * np.pi * f  * self.Larc
        Ss = (jwL - self.Zbase) / (jwL + self.Zbase)
        self.f, self.S = f, Ss
        return Ss
