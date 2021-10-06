"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 17 Feb 2021

@author: frederic
"""
__updated__ = "2021-02-18 10:50:29"

import numpy as np
import matplotlib.pyplot as pl

VERBOSE = False

def maxfun(xs, ys):
        
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
        p = np.polyfit([xs[idx],xs[idx+1],xs[idx+2]],
                       [ys[idx],ys[idx+1],ys[idx+2]],
                       deg=2)
        xy = (-p[1]/(2*p[0]), p[2]-p[1]**2/(4*p[0]))
        if xy[1] > xymax[1]:
            xymax = xy
            
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

    return xymax