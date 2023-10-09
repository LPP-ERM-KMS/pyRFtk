__updated__ = '2020-03-24 08:10:11'

import time

_ticT0_ = 0.

def tic():
    global _ticT0_
    _ticT0_ = time.time()
    return _ticT0_
    
def toc(t0 = None):
    return time.time() - (t0 if t0 else _ticT0_)

