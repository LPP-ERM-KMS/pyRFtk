"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 23 Dec 2020

@author: frederic
"""
__updated__ = "2021-01-08 08:32:21"

import pyRFtk2 as _rf

import pathlib as path

#===============================================================================

import logging
from pyRFtk2 import tLogger, setLogLevel
# tLogger.setLevel(logging.DEBUG)
setLogLevel(logging.DEBUG)

#===============================================================================

def find(ID, tvars):
    for k, obj in tvars.items():
        if isinstance(obj,_rf.rfBase):
          print(k, obj.ID)

#===============================================================================

a = _rf.rfBase(ID='1',sNp = '../test data/test_mixed_HFSSDesign1_unnormalized.s3p', Zbase=20)
b = _rf.rfBase(sNp = '../test data/test_mixed_HFSSDesign1_50Ohm.s3p',Zbase=20)
c = _rf.rfBase(funit='MHz',fs=[])
print(a)
print(b)
print(c)
find(b.ID, vars())