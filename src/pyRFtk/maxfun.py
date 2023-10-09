__updated__ = "2022-01-19 11:30:52"

import numpy as np
import matplotlib.pyplot as pl

from .config import logit, tLogger, ident

VERBOSE = False

import warnings

def maxfun(xs, ys):
    debug = logit['DEBUG']
    debug and tLogger.debug(ident(
        f'> [maxfun]',
        1
    ))
    
    xymax = [xs[-1], ys[-1]] if ys[-1] > ys[0] else [xs[0],ys[0]]
    
    if VERBOSE:
        pl.figure('maxfun')
        
    dys = np.sign(np.diff(ys))
    ddys = np.diff(dys)
    idxmaxs = np.where(ddys < 0)[0].tolist()
    
    if VERBOSE:
        pl.plot(xs,ys,'.-')
        for idx in idxmaxs:
            pl.plot(xs[idx],ys[idx],'go')
            
    cases = []
    
    if VERBOSE:
        if len(idxmaxs):
            if idxmaxs[0] != 0 and dys[0] < 0:
                cases.append((xs[0],ys[0]))
                pl.plot(xs[0],ys[0],'gs')
    
        elif len(dys):
            if dys[0] < 0:
                cases.append((xs[0],ys[0]))
                pl.plot(xs[0],ys[0],'cs')
        else:
            cases.append((xs[0],ys[0]))
            pl.plot(xs[0],ys[0],'rs')
                
    for idx in idxmaxs:
        cases.append([ys[idx],ys[idx+1],ys[idx+2]])
        
        # note polyfit can cast a numpy.RankWarning:
        #  this probably happens when the TL fixed step dx parameter causes
        #  the last but one and last points to be too close to each other
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = np.polyfit([xs[idx],xs[idx+1],xs[idx+2]],
                           [ys[idx],ys[idx+1],ys[idx+2]],
                           deg=2)
        
        xy = (-p[1]/(2*p[0]), p[2]-p[1]**2/(4*p[0]))
        if  xs[idx] <= xy[0] <= xs[idx+2]:
            if xy[1] > xymax[1]:
                xymax = xy
        else:
            debug and tLogger.debug(ident('Warning result outside window'))
            xymax = 0, 0
            for xm, ym in zip(xs[idx:idx+3],xs[idx:idx+3]):
                if ym > xymax[1]:
                    xymax = xm, ym
            
        if VERBOSE:
            pl.plot(xy[0],xy[1],'g^')
    
    if VERBOSE:
        if len(idxmaxs):
            if idxmaxs[-1] < (len(xs) - 1) and dys[-1] > 0:
                cases.append((xs[-1],ys[-1]))
                pl.plot(xs[-1],ys[-1],'cs')
        
        elif len(dys):
            if dys[-1] > 0:
                cases.append((xs[-1],ys[-1]))
                pl.plot(xs[-1],ys[-1],'rs')

    debug and tLogger.debug(ident(f'< [maxfun]', -1))
    return xymax
