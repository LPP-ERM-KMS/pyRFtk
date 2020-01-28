"""
Created on 23 Apr 2015

@author: frederic.durodie@gmail.com

+-------------+----------------------------------------------------------------+
| Date        | Comment                                                        |
+=============+================================================================+
|             |                                                                |
+-------------+----------------------------------------------------------------+

returns the copper resistivity (Ohm.m)  vs temperature (degree C)
"""

__updated__ = '2020-01-17 12:12:00'

def rhoCuT(TC=25) :
    """
    Pure anhealed Copper resistivity according IDM :
     'Pure_Cu_-_Electrical_Resistivity_24L79N_v1_1.doc'
    """
    return  (- 0.305
             + 6.8855E-03*(TC+273)
             - 0.6725E-06*(TC+273)**2
             + 0.8559E-09*(TC+273)**3) * 1e-8
             
if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as pl
    
    TCs = np.linspace(20,1000, num=99)
    pl.plot(TCs, rhoCuT(TCs)*1E8)
    pl.grid()
    pl.title('OFHC Cu resistivity vs Temperature [24L79N v1.1]')
    pl.xlabel('Temperature [$^\circ$C]')
    pl.ylabel('Resitivity [$\mu \Omega$.cm]')
    pl.tight_layout()
    pl.show()