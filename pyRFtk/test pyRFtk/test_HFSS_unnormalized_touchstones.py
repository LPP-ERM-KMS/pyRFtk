"""
Created on 17 Jan 2020

@author: frederic
"""

__updated__ = "2020-01-20 15:00:14"

import os

import pyRFtk.scatter3a as _sc
import pyRFtk.TouchStoneClass3a as _ts
from Utilities.printMatrices import printMA

pfmt = '%9.6f %+9.4f'
DATADIR = '../test data'
FILES = ['test_HFSSDesign1_50_Ohm.s2p',
         'test_HFSSDesign1.s2p',
         'TITAN_TWA_6s_v5_TWA_model_ports_conform_tap_30ohm_45MHz.s8p',
         'TITAN_TWA_6s_v5_TWA_model_ports_conform_tap_45MHz.s8p',
         'test_mixed_HFSSDesign1_50Ohm.s3p',
         'test_mixed_HFSSDesign1_unnormalized.s3p',
#          '4pj-L4PJ=403mm-lossy_normalized_50ohm_1.s4p',
#          '4pj-L4PJ=403mm-lossy_unnormalized_1.s4p',
        ]

NF = 4

path1 = os.path.join(DATADIR,FILES[NF])

ts1 = _ts.TouchStone(filepath=path1)

print(ts1.FormatStr())
fMHzs = ts1.Freqs('MHz')
SZ1 = ts1.Get_Scatter(fMHzs[0], 'MHz')
S1 = _sc.convert_general(50.0, SZ1.S, SZ1.Zbase, 'std', 'std')

printMA(SZ1.S, pfmt=pfmt)
printMA(S1, pfmt=pfmt)

path2 = os.path.join(DATADIR,FILES[NF+1])

print(path2)

ts2 = _ts.TouchStone(filepath=path2)

print(ts2.FormatStr())
fMHzs = ts2.Freqs('MHz')
SZ2 = ts2.Get_Scatter(fMHzs[0], 'MHz')

printMA(SZ2.S, pfmt=pfmt)
printMA(SZ2.S-S1, pfmt=pfmt)
