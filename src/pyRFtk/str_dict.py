__updated__ = "2021-12-16 15:50:48"

def str_dict(adict):
    
    try:
        str_dict.N
    except AttributeError:
        str_dict.N = 0
        str_dict.s = ''
    
    if isinstance(adict, dict):
        str_dict.s +=  '{\n'
        str_dict.N += 1
        for key, val in adict.items():
            str_dict.s += '    '*str_dict.N + ' '+key + ': '
            str_dict(val)
        str_dict.N -= 1
        str_dict.s += '    '*str_dict.N + '},\n'
    else:
        if isinstance(adict, tuple) and len(adict)==2:
            if len(adict[0]) == 1:
                str_dict.s += f'node [{"N/A" if adict[0][0] is None else "%7.4f"%adict[0][0]} m],\n'
            else:
                str_dict.s += (
                    f'line [{len(adict[0])} pts, '
                    f'{adict[0][0]:7.4f}m to  {adict[0][-1]:7.4f}m],\n')
        else:
            str_dict.s += f'?? {type(adict)},\n'
    
    return str_dict.s
        
if __name__ == '__main__':
    import pickle
    with open('../../pyRFtk test/fpj_vsw.bin','rb') as f:
        vsws = pickle.load(f)
#    vsws = {'a': 1, 'B': {'c': 2, 'D': {'e':1}}}
    s = str_dict(vsws)
    print(s)
    print(vsws)
