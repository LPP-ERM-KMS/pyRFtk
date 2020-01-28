#
# NOT FUNCTIONAL
#
"""
Created on 23 Nov 2017

@author: frederic.durodie@gmail.com

TL_class objects : simple homogeneous TL with properties

+-------------+----------------------------------------------------------------+
| Date        | Comment                                                        |
+=============+================================================================+
| 2017-Nov-23 | Created                                                        |
+-------------+----------------------------------------------------------------+

"""
#===============================================================================
#
# i m p o r t s
#
import numpy as np
from re import findall

#===============================================================================
#
# r c P a r a m s
#
rcParams = {'OD'    : 0.230, # m
            'Z'     : 30.,   # Ohm
            'V'     : 1.,    # c0
            'A'     : 0.,    # 1/m
            'ro'    : 0.,    # Ohm.m
            'ri'    : 0.,    # Ohm.m
            'units' : {'frequency'             : 'MHz',
                       'impedance'             : 'Ohm',
                       'conductance'           : 'Mho',
                       'resistivity'           : 'muOhm/m',
                       'conductivity'          : 'muMho/m',
                       'capacitance'           : 'pF',
                       'inductance'            : 'nH',
                       'length'                : 'm',
                       'specific_conductivity' : 'Mho/m',
                       'specific_resistivity'  : 'uOhm.m',
                       'specific_capacitance'  : 'pF/m',
                       'specific_inductance'   : 'nH/m',
                       'voltage'               : 'kV',
                       'current'               : 'kA',
                       'power'                 : 'MW',
                    },
           }

#===============================================================================
#
# u n i t   a n d   s c a l e   h a n d l i n g
#
scales = {'a' :1e-18, 'f' :1e-15, 'p' :1e-12, 'n' :1e-09, 'u' :1e-06, 'm' :1e-03,
          'c' :1e-02, 'd' :1e-01, ''  :1e+00, 'da':1e+01, 'h' :1e+02,
          'k' :1e+03, 'M' :1e+06, 'G' :1e+09, 'T' :1e+12, 'P' :1e+15, 'E' :1e+18}

scale_unit = lambda _u : findall('(p|n|u|m|c|da|d|h|k|M|G|T){0,1}'
                                 '(Ohm\.m|Ohm\/m|Ohm'
                                 '|Mho\/m|Mho'
                                 '|Hz|m|F\/m|F|H\/m|H'
                                 '|V|A|W'
                                 '){1}',_u)[0]

scale = lambda _u : scales[scale_unit(_u)[0]]

#===============================================================================
#
# T L o b j e c t 
#
class TLobject():
    """TLobject:
    """
    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, *args, **kwargs):
        """TLobject:__init__:
        """
        units = rcParams['units']
        
        self.length = 0.
        if args:
            self.length = args.pop() * scale(units['length'])
        
        self.frequency = 0.
        if args:
            self.frequency = args.pop() * scale(['frequency'])
        
        if args:
            raise ValueError('TLobject.__init__: only length and frequency '
                             'allowed as positional arguments')
            
        if 'length' in kwargs and not(self.length):
            self.length = kwargs.pop('length',0.) * scale(units['length'])
            
        if 'frequency' in kwargs and not(self.frequency):
            self.length = kwargs.pop('frequency',0.) * scale(units['frequency'])
    
        if args:
            raise ValueError('TLobject.__init__: only length and frequency '
                             'allowed as positional arguments')

    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, **kwargs):
        """TLobject:__str__:
        """
        pass
    
#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    for u, v in rcParams['units'].items():
        try:
            scaler, tunit = scale_unit(v)
            tscale = scales[scaler]
            if not scaler:
                scaler = ' '
            print('%-25s : %10.0e(%s)  %1s' % (u, tscale, scaler, tunit))
        except:
            print(u, v)
