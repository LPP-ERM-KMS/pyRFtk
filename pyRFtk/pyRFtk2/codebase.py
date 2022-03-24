"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 10 Nov 2021

@author: frederic
"""
__updated__ = "2021-11-15 09:31:27"

import os

def list1dir(path, level=" "):
    N = 0
    # curdir = os.getcwd()
    # os.chdir(path)    
    for p in sorted(os.listdir(path)):
        if os.path.isdir(p) and p not in ['obsolete', '__pycache__']:
            print(f'{level}{os.path.basename(p)}:')
            N += list1dir(os.path.join(path,p), level + "    ")
        elif p[-3:] == '.py' and p not in ['codebase.py']:
            with open(os.path.join(path,p),'r') as f:
                n = len(f.readlines())
            N += n
            print(f'{level}{os.path.basename(p):{50 - len(level)}s} {n:5d}')
    # os.chdir(curdir)
    return N

print(list1dir('.'))