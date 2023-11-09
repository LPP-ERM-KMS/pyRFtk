__updated__ = "2023-08-24 11:11:43"

import os

def list1dir(path, level=" "):
    N, N1 = 0, 0
    # curdir = os.getcwd()
    # os.chdir(path)    
    for p in sorted(os.listdir(path)):
        if os.path.isdir(p) and p not in ['obsolete', '__pycache__']:
            print(f'{level}{os.path.basename(p)}:')
            n, n1 = list1dir(os.path.join(path,p), level + "    ")
            N += n
            N1 += n1
        elif p[-3:] == '.py' and p not in ['codebase.py']:
            with open(os.path.join(path,p),'r') as f:
                # n = len(f.readlines())
                n, n1 = 0, 0
                for ll in f.readlines():
                    lll = ll.strip()
                    if len(lll) > 0 and lll[0] != '#':
                        n1 += 1
                    n += 1
            N += n
            N1 += n1
            print(f'{level}{os.path.basename(p):{50 - len(level)}s} {n:5d} / {n1:5d}')
    # os.chdir(curdir)
    return N, N1

print(list1dir('.'))
