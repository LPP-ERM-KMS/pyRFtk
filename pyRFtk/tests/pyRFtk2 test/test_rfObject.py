"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 19 Aug 2021

@author: frederic
"""
__updated__ = "2021-08-19 13:13:41"


import numpy as np
import matplotlib.pyplot as pl
import sys
import os

from pyRFtk2 import rfObject

print(os.path.abspath(os.path.curdir))
O1 = rfObject(touchstone='../test data/StrapOrientation_v5p0_7_reference_R2_HFSSDesign1.s24p')
print(O1.getS(37.3e6))
print(O1)