"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 18 Dec 2020

@author: frederic
"""
__updated__ = "2020-12-18 18:10:18"

from pyRFtk2.CommonLib import ReadTSF

R = ReadTSF('../test data/SRI.s6p')
print(R)