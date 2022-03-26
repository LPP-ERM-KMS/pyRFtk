################################################################################
#                                                                              #
# Copyright 2018-2022                                                          #
#                                                                              #
#                   Laboratory for Plasma Physics                              #
#                   Royal Military Academy                                     #
#                   Brussels, Belgium                                          #
#                                                                              #
#                   ITER Organisation                                          #
#                                                                              #
# Author : frederic.durodie@rma.ac.be                                          #
#                          @gmail.com                                          #
#                          @ccfe.ac.uk                                         #
#                          @telenet.be                                         #
#                          .lpprma@telenet.be                                  #
#                                                                              #
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by   #
# the European Commission - subsequent versions of the EUPL (the "Licence");   #
#                                                                              #
# You may not use this work except in compliance with the Licence.             #
# You may obtain a copy of the Licence at:                                     #
#                                                                              #
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-11-12                  #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the Licence is distributed on an "AS IS" basis,            #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the Licence for the specific language governing permissions and          #
# limitations under the Licence.                                               #
#                                                                              #
################################################################################

__updated__ = '2022-03-25 16:57:22'

"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 14 Oct 2021

@author: frederic
"""

import numpy as np
import matplotlib.pyplot as pl

from pyRFtk2.config import setLogLevel, logit, tLogger, ident
from pyRFtk2.Utilities import str_dict

#===========================================================================
#
# p l o t V S W s 
#
def plotVSWs(VSWs, maxlev=4, Id = None,  **kwargs):
    
    figkwargs = dict([ (kw, val) for kw, val in kwargs.items() 
                     if kw in ['figsize', 'num']])
    
    if 'num' not in figkwargs:
        if Id is None and len(VSWs)==1:
            Id = next(iter(VSWs))
            
    if Id: 
        if 'num' not in figkwargs:
            figkwargs['num'] = Id
        else:
            raise ValueError(
                'pyRFtk2.commonLib.plotVSWs: only one of "ID" or "num" kwargs allowed')
        
    tfig, ax = pl.subplots(**figkwargs)
    
    plotnodes = kwargs.pop('plotnodes', False)
    
    ankwargs = dict(
        xytext=(0,30),
        textcoords='offset points', 
        ha='center', 
        va='bottom', 
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=1), 
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0') 
    )
    
    def plotit(BLK, VSW, lev=0):
        _debug_ = logit['DEBUG']
        _debug_ and tLogger.debug(ident(
            f'> [plotit] BLK={BLK}, type(VSW)={type(VSW)},lev= {lev}', +1))
        if isinstance(VSW, dict):
            for blk, vsw in VSW.items():
                plotit(blk, vsw, lev + 1)
        elif  isinstance(VSW, tuple):
            xs, absV = VSW
            absV = np.abs(absV)
            # note: matplotlib does not show legend labels starting with _
            lbl = (('' if BLK[0] != '_' else ' ') + BLK) if lev < maxlev else '_'
            _debug_ and tLogger.debug(ident(
                f'label= {lbl}; '
                f'BLK[0] != "_": {not(BLK[0] == "_")}, '
                f'lev < maxlev: {lev < maxlev}'
            ))
            if any([x != xs[0] for x in xs]) or any([v != absV[0] for v in absV]):
                _debug_ and tLogger.debug(ident(f'vsw'))
                pl.plot(xs, absV, '.-', label=lbl)
            elif plotnodes:
                _debug_ and tLogger.debug(ident(f'node'))
                if isinstance(xs[0], (float,int)):
                    pl.plot(xs, absV, 'd', label=lbl)
                    
        _debug_ and tLogger.debug(ident(f'< [plotit]',-1))

    plotit(Id, VSWs)
                
    pl.figure(tfig.number)
    pl.legend(loc='best')
    pl.xlabel('x [m]')
    pl.ylabel('U [kV]')
    pl.title(f'{Id} Voltage standing waves')
    pl.grid(True)
    
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    def findclosest(xp, yp, ks):
    
        _debug_ = logit['DEBUG']
        _debug_ and tLogger.debug(ident(
            f'> [findclosest] xp= {xp}, yp={yp}',
            1
        ))
        
        #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        
        def find(BLK, VSW, curdist2=np.Inf, curBLK='', curUp=yp, curX=xp):
            _debug_ and tLogger.debug(ident(
                f'> [find] BLK= {BLK}, '
                f'VSW= {[_ for _ in VSW] if isinstance(VSW, dict) else type(VSW)}, '
                f'curBLK={curBLK}, curUp= {curUp}',
                1
            ))
            if isinstance(VSW, dict):
                _debug_ and tLogger.debug(ident(f'VSW = dict'))
                
                for blk_, vsw_ in VSW.items():
                    _debug_ and tLogger.debug(ident(
                        f'blk_= {blk_}, vsw_= {type(vsw_)}'
                    ))
                    ndist2, nblk, tUp, tX = find(
                        blk_, vsw_, curdist2, curBLK, curUp, curX)
                    
                    if ndist2 < curdist2:
                        
                        # curdist, curBLK, curUp = ndist, f'{BLK}.{nblk.split(".",1)[-1]}', tUp
                        curdist2, curBLK, curUp, curX = ndist2, f'{BLK}.{nblk}', tUp, tX
    
            elif isinstance(VSW, tuple):
                _debug_ and tLogger.debug(ident(f'VSW = tuple[{len(VSW)}]'))
                xs, Us = VSW
                Us = np.abs(Us)
                
                if len(xs) > 1:
                    _debug_ and tLogger.debug(ident(f'len(VSW[0])= {len(VSW[0])}'))
                    x1, U1 = xs[0], Us[0]
                    for x2, U2 in zip(xs[1:], Us[1:]):
                        d2 = (x1 - x2)**2 + (ks * (U1 - U2))**2
                        if d2 > 0:
                            q = min(max(
                                (x1**2 - x1 * x2 - (x1 - x2) * xp +
                                 ks**2 * (U1**2 - U1 * U2 - (U1 - U2) * yp)) / d2,
                                0.),1.)
                        else:
                            q = 0.
                        xq, yq = x1 + q * (x2 - x1), U1 + q * (U2 - U1)
                        ndist2 = (xq - xp)**2 + (ks * (yq - yp))**2
                        x1, U1 = x2, U2
                        if ndist2 < curdist2:
                            curdist2, curBLK, curUp, curX = ndist2, BLK, yq, xq
                else:
                    _debug_ and tLogger.debug(ident(f'(node)'))
                    if isinstance(xs[0], (float, int)):
                        ndist2 = (xs[0] - xp)**2 + (ks * (Us[0] - yp))**2
                        if ndist2 < curdist2:
                            curdist2, curBLK, curUp, curX = ndist2, BLK, Us[0], xs[0]
                                
            _debug_ and tLogger.debug(ident(
                f'< [find] curdist2= {curdist2}, curBLK= {curBLK}, '
                f'curUp= {curUp}, curX= {curX}',
                -1
            ))
            return curdist2, curBLK, curUp, curX
        
        #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    
        _, blk, Up, tX = find('<top>', VSWs)
    
        _debug_ and tLogger.debug(ident(
            f'< [findclosest] Up= {Up}, ID= {blk.split(".",1)[-1]}',
            -1
        ))
        
        return Up, f'{blk.split(".",1)[-1]}', tX
    
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    
    annotation = ax.annotate(f"<ID>\n<Up>kV", xy=(0., 0.), **ankwargs 
                )
    annotation.set_visible(False)
    
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    
    def hover(event):
        nonlocal annotation
        if event.inaxes == ax:
            annotation.set_visible(False)
            xi, yi = event.xdata, event.ydata
            xsc = pl.gca().get_xlim()
            ysc = pl.gca().get_ylim()
            ks = (xsc[1]-xsc[0])/(ysc[1]-ysc[0])
            # k0 = xsc[0] - ysc[0] * ks
            Up, ID,tX = findclosest(xi, yi, ks)
            if ID:
                annotation = ax.annotate(
                    f"{ID}\n{Up:.3f}kV",  xy= (tX,Up), **ankwargs)
                annotation.set_visible(True)
                pl.draw()
    
    tfig.canvas.mpl_connect("motion_notify_event", hover)
    
    return

#===============================================================================
#
# s c a l e V S W
#
def scaleVSW(VSW, scale):
    for Id, val in VSW.items():
        if isinstance(val, dict):
            scaleVSW(val, scale)
        elif hasattr(val, '__len__') and len(val) == 2:
            if hasattr(val[1], '__iter__'):
                for k in range(len(val[1])):
                    val[1][k] *= scale
            else:
                VSW[Id] *= scale
        else:
            raise ValueError('pyRFtk.CommonLib.plotVSWs.scaleVSW: internal error')
        
#===============================================================================
#
# s t r V S W
#
def strVSW(VSW, indent=0):
    s =''
    for key, val  in VSW.items():
        s += f'{" "*indent}{key}\n'
        if isinstance(val, dict):
            s += strVSW(val,indent+1)
        elif isinstance(val, tuple) and len(val) == 2:
            if hasattr(val[0], '__iter__') and hasattr(val[1], '__iter__'):
                if len(val[0]) == len(val[1]):
                    for _, (x, y) in enumerate(zip(val[0], val[1]),1):
                        try:
                            s += f'{" "*(indent+2)}{_:5d}:{np.abs(x):10.3f},{np.abs(y):10.3f}\n'
                        except:
                            if x is None:
                                s += f'{" "*(indent+2)}{_:5d}:{" < None > ":10s},{np.abs(y):10.3f}\n'
                            else:
                                s += f'! #={_}, x={x}, y={y}\n'
                else:
                    s += f'! unequal array/list lengths x:{len(val[0])} <> y:{len(val[0])}\n'
            elif not (hasattr(val[0],'__iter__') or hasattr(val[1],'__iter__')):
                s += f'{" "*(indent+2)}{"node":5s}:{np.abs(val[0]):10.3f},{np.abs(val[1]):10.3f}\n'
            else:
                s += f'! tuple error ...\n'
        else:
            s += f'! unexpected value type {type(val)} expected tuple\n'
    return s

#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    import pickle
    from pprint import pprint
    
    with open('../../pyRFtk2 test/Launcher_VSWs.bin','rb') as f:
        VSWs = pickle.load(f)
    setLogLevel('DEBUG')
    VSWs1 = VSWs['<top>']['VTL_1']
    plotVSWs(VSWs1, maxlev=2)
    print(str_dict(VSWs1))
    pl.show()


