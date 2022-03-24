"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 4 Nov 2021

@author: frederic
"""
__updated__ = "2021-12-08 14:28:13"

import sys

def whoami(pname=None, dframe=1):
    frame = sys._getframe(dframe)
    fname = frame.f_code.co_name
    lnum = frame.f_lineno
    try:
        cname = frame.f_locals["self"].__class__.__name__
    except KeyError:
        cname = '<None>'
    if pname is None:
        pname = __package__.rsplit('.')[0]
    return f'{pname}.{cname}.{fname}[{lnum}]'
