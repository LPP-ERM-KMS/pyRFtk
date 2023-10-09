__updated__ = '2022-05-17 12:57:05'

if __name__ == '__main__':
    import sys
    print('please run test_rfGTL...')
    sys.exit(1)

import numpy as np            # @UnusedImport but it is used in the eval strings
from pprint import pprint
import ast
import os
import warnings
from copy import deepcopy


from . import rfCircuit, rfTRL, rfRLC, rfObject, rfArcObj
from .whoami import whoami 
from .config import logit, tLogger, ident, _newID, logident

#===============================================================================
#
# r f G T L  
#
class rfGTL(rfCircuit):
    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, path2model, objkey=None, variables={}, **kwargs):
               
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', printargs=False)
        
        if not hasattr(self,'args'):
            self.args = (path2model,)
            
        if not hasattr(self,'kwargs'):
            self.kwargs = dict(objkey=objkey, 
                               variables=variables.copy(),
                               **kwargs.copy())
        
        Id = kwargs.pop('Id','rfGTL_' + _newID())
        _debug_ and logident(f'Id = {Id}')
        super().__init__(**dict(kwargs,Id=Id))
        
        _debug_ and logident(f'self.Id <- {self.Id}')
        
        self.path2model = path2model
        self.variables = variables.copy()
        # obsolete -- self.kwargs = kwargs.copy()
        
        if objkey == None:
            try:
                objkey = os.path.splitext(path2model)[1].upper().replace('.MOD','')
            except:
                objkey = 'model'
                
        self.objkey = objkey
            
        with open(path2model,'r') as f:
            self.model = ast.literal_eval(f.read())
        
        try:
            self.GTL = self.model[self.objkey]['GTL']
        except KeyError:
            for key in self.model:
                print(f' {key}')
            raise
        
        self.glovars = self.model.get('variables',{})
        self.locvars = self.model[objkey].get('variables', {})
            
        self.glovars.update(variables)
        self.Portnames = self.model[objkey].get('Portnames',[])
        self.vars = dict(self.locvars)
        # pprint(self.vars)
        self.vars.update(self.glovars)
              
        self.path2sNp = None
        # do not set self.sNp = None because getS checks on hasatrr(obj, 'sNp')
        if 'sNp' in self.model[objkey]:
            self.path2sNp = self.model[objkey]['sNp']
            self.sNp = rfObject(touchstone=os.path.join(
                                        os.path.dirname(path2model),
                                        self.path2sNp))
            if self.Portnames:
                self.sNp.ports = self.Portnames
                
        #----------------------------------------------------- resolve variables
        found = True # prime the while loop
        while found:
            found = {}
            for var, val in self.vars.items():
                if isinstance(val, str):
                    try:
                        found[var] = eval(val, globals(), self.vars)
                    except (NameError, SyntaxError, TypeError):
                        pass # probably could not resolve yet
            if found:
                self.vars.update(found)    
                
        positions = self.model[objkey].get('relpos',{})
        for p in positions:
            if p not in self.Portnames:
                raise ValueError(
                    f'{whoami(__package__)}: port name {p} is not in list of'  # @UndefinedVariable
                    f' declared ports {self.Portnames}'
                )
            else:
                if isinstance(positions[p], str):
                    positions[p] = eval(positions[p], globals(), self.vars)
                
        #
        portlist = {}
        # obsolete -- xposlist = {}
        for blk, tkwargs in self.GTL.items():
             
            tkwargs = self._remap_(blk, tkwargs, [('ports', 'Portnames', True)])
                          
            blkPortnames = tkwargs['ports']
            terminated = False
            for kp, p in enumerate(blkPortnames):
                #------------------------------------ check for terminated ports
                if isinstance(p, dict):
                    # terminate kwargs
                    terminated = (p, '[**]')
                    blkPortnames[kp] = '[**]'
                    continue
                elif isinstance(p, str):
                    if p == '[oc]':
                        terminated = {'Y':0}, '[oc]'
                        continue
                    elif p == '[sc]':
                        terminated = {'Z':0}, '[sc]'
                        continue
                #---------------------------- check if port was already declared
                if p not in portlist:
                    portlist[p] = []
                portlist[p].append(blk)
            
            #---------------------------------------------- substitute variables
            for kw, val in tkwargs.items(): #FIXME: 'ports' is a tkwarg and should not be considered
                if isinstance(val, str):
                    try:
                        tkwargs[kw] = eval(val, globals(), self.vars)
                    except ():
                        raise RuntimeError(
                            f'{whoami(__package__)}: could not eval substitution for "{kw}"'  # @UndefinedVariable
                            f' = "{val}" in GTL block "{blk}"'
                        )
            #----------------------------------------------------- convert units  
            conv = {
                'ID'  : 1e-3, 
                'OD'  : 1e-3,
                'CTL' : 1e-12,
                'LTL' : 1e-9,
                'rho' : 1e-6,
                'Cs'  : 1e-12,
                'Cp'  : 1e-12,
                'Ls'  : 1e-9,
                'Lp'  : 1e-9,
                'Larc': 1e-9,
            }
            for kw in conv:
                if kw in tkwargs:
                    if isinstance(tkwargs[kw], (list, tuple)):
                        tkwargs[kw] = [v * conv[kw] for v in tkwargs[kw]]
                    else:
                        tkwargs[kw] *= conv[kw]
                        
            tkwargs = self._remap_(blk, tkwargs, [('L', 'length', False),
                                                  ('Z0TL', 'Z', False),
                                                  ('xpos', 'relpos', False)])
            
            if 'xpos' in tkwargs:
                xpos = tkwargs.pop('xpos')
                positions[f'{blkPortnames[0]}'] = xpos
            else:
                xpos = -10
                
            plotkws = tkwargs.pop('plotkws',{}) # future feature
            
            if any([kw in tkwargs for kw in ['Ls','Cs','Rs','Lp','Cp','Rp']]):
                # rfRLC object                       
                # obsolete -- xposlist[blk] = [xpos, xpos]
                RFobj = rfRLC(Zbase=self.Zbase,**tkwargs)
            elif len(kwargs) == 1 and 'Larc' in kwargs:
                # rfArcObject
                # obsolete -- xposlist[blk] = [xpos, xpos]
                RFobj = rfArcObj(kwargs['Larc'])
            else:
                # rfTRL object
                # obsolete -- xposlist[blk] = [xpos, xpos + tkwargs['L']]
                try:
                    RFobj = rfTRL(Zbase=self.Zbase,**tkwargs)
                except:
                    print(blk)
                    pprint(tkwargs)
                    raise
                
            self.addblock(blk, RFobj, ports=blkPortnames, relpos=xpos)
            if terminated:
                self.terminate(f'{blk}.{terminated[1]}', **terminated[0])
        
        # connect all ports that belong to more than 1 block together
        nports = 0
        for _p, _blks in portlist.items():
            if len(_blks) > 1:
                self.connect(*[f'{_b}.{_p}' for _b in _blks])
            else:
                nports += 1 # count the free ports
                
        if nports != len(self.Portnames):
            raise ValueError(
                f'{whoami(__package__)}: number of declared ports ({len(self.Portnames)})'  # @UndefinedVariable
                f' and the resulting free ports ({nports}) mismatch')
        
        # resolve the internal port positions
        _debug_ and logident('finding port positions')
        
        # for _kp, _p in enumerate(self.Portnames):
        #     _blk = self.blocks[portlist[_p][0]]
        #     _blk_relpos = _blk['xpos'] + xpos
        #     _xpos = _blk['object'].xpos[_blk['ports'].index(_p)] + _blk_relpos
        #     self.xpos.append(_xpos)
        
        for p in positions:
            for blk in portlist[p]:
                tblk = self.blocks[blk]
                tports = tblk['ports']
                k1 = tports.index(p)
                if k1 == 0:
                    tblk['xpos'] = positions[p]
                else:
                    rfobj_xpos = tblk['object'].xpos
                    tblk['xpos'] = positions[p] - rfobj_xpos[1] + rfobj_xpos[0]
        
        found = True # prime the loop
        setblks = []
        while found:
            found = False
            for blkId, tblock in self.blocks.items():
                if blkId in setblks:
                    continue
                
                obj = tblock['object']
                for kp, (p, xp) in enumerate(zip(tblock['ports'], obj.xpos)):
                    if p in positions:
                        found = True
                        setblks.append(blkId)
                        tblock['xpos'] = positions[p] - xp
                        # set other ports related to that blk
                        for p1, xp1 in zip(tblock['ports'], obj.xpos):
                            if p1 != p and p1 not in ['[**]','[oc]','[sc]']:
                                if p1 not in positions:
                                    positions[p1] = tblock['xpos'] + xp1
                                else:
                                    err = (positions[p1] - xp1) - (positions[p] - xp)
                                    if np.abs(err) > 1E-5:
                                        msg = ('rfGTL: inconsistent positions '
                                               f'{err*1e3}mm for {blkId}.{p} and '
                                               f'{blkId}.{p1}')
                                        _debug_ and logident(msg)
                                        if np.abs(err) > 1E-2:
                                            warnings.warn(msg)
                        break
                
                        
            continue
            print('woo wooo wooooo ...')
            # below here is not executed
            for p, blks in portlist.items():
                if p not in positions:
                    for blk in blks:
                        tblk = self.blocks[blk]
                        rfobj, ports = tblk['object'], tblk['ports']
                        rfobj_xpos = rfobj.xpos
                        if p in ports:
                            for p2 in [p1 for p1 in ports  if p1 != p]:
                                # if (p1 != p and p1 not in ['[**]','[oc]','[sc]'])]:
                                if p2 in positions:
                                    kp = ports.index(p)
                                    found = True
                                    if kp == 0:
                                        tblk['xpos'] = (
                                            positions[p2] - rfobj_xpos[1] + rfobj_xpos[0])
                                        positions[p] = tblk['xpos']
                                    else:
                                        tblk['xpos'] = positions[p2]
                                        positions[p] = tblk['xpos'] + rfobj_xpos[1] - rfobj_xpos[0]
                                    
        # connect free ports to a port name without the block to which it belongs
        for _p in self.Portnames:
            if _p in portlist and len(portlist[_p]) == 1:
                self.connect(f'{portlist[_p][0]}.{_p}', _p)
                
            else:
                raise ValueError(
                    f'{whoami(__package__)}: declared port name "{_p}" not found'  # @UndefinedVariable
                    ' in the list of resulting free ports'
                )
                
        # FIXME: somehow self.ports and self.Portname are doubling up
        #        and it is confusing ... (as long as they are the same
        #        it does not matter)
        #        note that self.Portnames is set by the "Portnames" entry in the
        #        GTL dict
        
        if self.ports != self.Portnames:
            raise RuntimeError(
                f'{whoami(__package__)}: self.ports and self.Portnames differ !'  # @UndefinedVariable
            )
                    
        # found = True # prime the loop
        # xpos = {}
        # while found:
        #     found = False
        
        # finally check the portnames for the sNp if present
        try:
            self.sNp.Portnames = self.Portnames
        except AttributeError:
            pass
        
        _debug_ and logident('prime rfGTL object calling getSgtl')
        S = self.getSgtl(45e6)
        
        _debug_ and logident('<')
        
    #===========================================================================
    #
    # c o p y
    #
    def copy(self):
        return self.__deepcopy__()
    
        # this has become obsolete
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', printargs=True)
        other = super().copy()
        other.model = self.model.copy()
        other.GTL = self.GTL.copy()
        other.glovars = self.glovars.copy()
        other.locvars = self.locvars.copy()
        other.vars = self.vars.copy()
        other.path2sNp = self.path2sNp.copy()
        _debug_ and logident('<')
        return other
    
    def __copy__(self):
        return self.__deepcopy__()
    
    def __deepcopy__(self, memo={}):
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', printargs=True)
        
        # other = type(self)(self.path2model, self.objkey, self.variables, **self.kwargs) # because rfGTL object has a path2model
        other = type(self)(*self.args, **self.kwargs) # because rfGTL object has a path2model
        for attr, val in self.__dict__.items():
            try:
                _debug_ and logident(f'copy {attr}')
                other.__dict__[attr] = deepcopy(val)
            except:
                print(f'{whoami(__package__)}: failed to deepcopy {attr}')  # @UndefinedVariable
                raise
        
        _debug_ and logident('<')
        return other
        
    #===========================================================================
    #
    # _ r e m a p _
    #
    def _remap_(self, blk, aDict, kwPairList):
        
        for kwPref, kwAlt, mandatory in kwPairList:
            if kwPref not in aDict:
                try:
                    aDict[kwPref] = aDict.pop(kwAlt)
                except KeyError:
                    if mandatory:
                        raise ValueError(
                            f'pyRFtk2.rfGTL: "GTL" block "{blk}" must have one of'
                            f' "{kwPref}" or "{kwAlt}" keys'
                        )
            elif kwAlt in aDict:
                raise ValueError(
                    f'pyRFtk2.rfGTL: "GTL" block "{blk}" cannot have both the'
                    f' "{kwPref}" or "{kwAlt}" keys'
                )
                    
        return aDict
            
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, full=0):
        s = f'{type(self).__name__} Id="{self.Id}" at {hex(id(self))}\n'
        s += f'|  Portnames          : {", ".join(self.Portnames)}\n'
        
        ml = max([len(_p) for _p in self.ports])
        # print(ml, len(self.ports), len(self.xpos))
        for _p, _x in zip(self.ports, self.xpos):
            s += f'|   {_p:{ml+1}s}: {_x:.4f}m\n'

        s += f'|  Zbase              : {self.Zbase:.3f} Ohm\n'
        try:
            s += f'|  sNp                : {self.path2sNp}\n'
        except AttributeError:
            pass
        if self.glovars:
            s += '|  global variables   :\n'
            for kw, val in self.glovars.items():
                s += f'|  |  {kw:<15s} : {val}\n'
            s += '|  ^\n'
        if self.locvars:
            s += '|  local variables    :\n'
            for kw, val in self.locvars.items():
                s += f'|  |  {kw:<15s} : {val} '
                if kw in self.glovars:
                    s += f'-> {self.glovars[kw]}'
                if isinstance(val,str):
                    s += f'-> {self.vars[kw]}'
                s += '\n'
            s += '|  ^\n'
        if full:
            for sk in super().__str__(full-1).split('\n'):
                s += f'|  {sk}\n'
        s += '^'
        return s

    #===========================================================================
    #
    # g e t S g t l
    #
    def getSgtl(self, fs, Zbase=None, params={}, flags={}):
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', printargs=True)
        if Zbase == None:
            Zbase = self.Zbase
        # print('using gtl')
        S = super().getS(
            fs, Zbase=Zbase, params=params, flags=dict(flags, sNp=False))
        
        _debug_ and logident('<')
        return S
    
    #===========================================================================
    #
    # g e t S
    #
    def getS(self, fs, Zbase=None, params={}, flags={}):
        
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', printargs=True)
        
        if Zbase == None:
            Zbase = self.Zbase
            
        # we need to call getSgtl at least once here because the onderlying 
        # circuit object does not know about the touchstone object to evaluate
        # the voltages at the circuit's nodes ... the extS of the circuit parent
        # will not be called by self.getS (which only fetches the touchstone 
        # value) and update the various excitation port equation numbers (self.E)
        
        #FIXME: params is not consistently used across the various call signatures
        
        # this should force the correct update of the circuit object's self.E
        if not hasattr(super(),'E') or not super().E:

            self.getSgtl(fs[0] if hasattr(fs,'__iter__') else fs, 
                         Zbase=Zbase, params=params)
        
        # use the touchstones if available
        if hasattr(self, 'sNp') and flags.get('sNp',True):
            _debug_ and logident(f'using sNp')

            S = self.sNp.getS(fs, Zbase=Zbase, params=params, flags=flags)
        
        else:
            _debug_ and logident(f'not using sNp')
            S = super().getS(fs, Zbase=Zbase, params=params, flags=flags)
        
        _debug_ and logident('<')
        return S
    
    #===========================================================================
    #                
    # S o l u t i o n
    #
    
##
## this is the old processGTL code from pyRFtk.circuit
## the code lines at the end requiring to import pyRFtk.scatter3a were commented 
   
#===============================================================================
#
# p r o c e s s G T L   
#
def processGTL(fMHz, Zbase, GTL_data, **kwargs):
    """
    a GTL data structure is as follows:
    
    {'Portnames': ordered list of port names (if a sNp touchstone file  is 
                  provided the order must match),
                  
     'sNp'      : optional touchstone file, # *** future extension ***
     
     'relpos'   : {'p1':position, ... } where the p1 are some or all of the 
                  portnames,
     
     'variables': { dict of substitutions },
     
     'defaults' : { dict of default AddBlockGTL kwargs }, # *** future extension ***
     
     'GTL       : {
         'TLid1': { # AddBlockGTL kwargs General TL section 1
         
             length: float
             
             Portnames: [ list of 2 portnames ], # portname = 
                                                 #     str | (str, dict) | dict
                                                 # dict = CT.Terminate kwargs
                                                 
             relpos: float # optional: position of 1st port
             
             ... General_TL kwargs of the TL properties ...
         
         }, # TLid1
         
         ...
         
         'TLidN': { AddBlockGTL kwargs General TL section N 
         
             ... TLid 1 ...
         }, # TLidN
         
     }, # GTL
    }
    """
    
    #---------------------------------------------------- r e s o l v e V a  r s   
    #
    def resolveVars(kwargs, variables):
        ekwargs = {}
        for kw, val in kwargs.items():
            if isinstance(val, str):
                try:
                    ekwargs[kw] = eval(val, globals(), variables)
                except:
                    print(f'val       = {val}')
                    print(f'variables = {variables}')
                    raise
            else:
                ekwargs[kw] = val

        return ekwargs
    #
    #- ----------------------------------------------------------------------- -
    
    CT = rfCircuit(fMHz=fMHz, Zbase=Zbase)
    variables = {**GTL_data.get('variables',{}), **kwargs.get('variables',{})}
    Portnames = GTL_data.get('Portnames', [])
    relpos = GTL_data.get('relpos', dict([(p, 0.) for p in Portnames[:-1]]))
    
    #--------------------------------------------------------- resolve variables
    found = True # prime the while loop
    while found:
        found = {}
        for var, val in variables.items():
            if isinstance(val, str):
                try:
                    found[var] = eval(val, globals(), variables)
                except (NameError, SyntaxError, TypeError):
                    pass # probably could not resolve yet
        if found:
            variables.update(found)    
            
    #------------------------------------------- check for expressions in relpos
    
    for p, v in relpos.items():
        if isinstance(v,str):
            relpos[p] = eval(v, globals(), variables)
    
    #----------------------------------------------------- iterate over GTL list
    GTLs = {}
    for TL, GTLkwargs in GTL_data['GTL'].items():
        
        GTLkws =    resolveVars(GTLkwargs, variables)
        terminations = []
        for k, port in enumerate(GTLkws['Portnames']):
            
            # detect if a port is terminated or not
            if isinstance(port, dict):
                # auto generated portname (we not need it later on)
                GTLkws['Portnames'][k] = '<T%05d>' % CT.newID()
                terminations.append((GTLkws['Portnames'][k], port))
                
            elif isinstance(port, tuple):
                # we are interested to keep the portname
                GTLkws['Portnames'][k] = port[0]
                terminations.append(port)
            
            # substitute variables/expressions
            for k, (port, Tkws) in enumerate(terminations):
                for kw, val in Tkws.items():
                    if isinstance(val, str):
                        Tkws[kw] = eval(val, globals(), variables)
                                
        GTLs[TL] = [GTLkws, terminations]
    
    #------------------------------------ try and resolve the relative positions
        
    found = True
    while found:
        found = False
        for TL in GTLs:
            for k, port in enumerate(GTLs[TL][0]['Portnames']):
                if 'relpos' not in GTLs[TL][0]:
                    if port in relpos:
                        if k: # the 2nd port: thus the relpos is moved back
                            GTLs[TL][0]['relpos'] = relpos[port] - GTLs[TL][0]['length']
                        else:
                            GTLs[TL][0]['relpos'] = relpos[port]
                        # print('%s : %.4f m' % (TL, GTLs[TL][0]['relpos']))
                        found = True
                    
                    else:
                        # we don't have a port position yet 
                        otherport = GTLs[TL][0]['Portnames'][1-k]
                        if otherport in relpos:
                            # but we got the other port
                            if k: # the 2nd port: thus otherport is port 1
                                relpos[port] = relpos[otherport] + GTLs[TL][0]['length']
                                GTLs[TL][0]['relpos'] = relpos[otherport]
                            else: # the 1st port thus other port is port 2
                                relpos[port] = relpos[otherport] - GTLs[TL][0]['length']
                                GTLs[TL][0]['relpos'] = relpos[port]
                            # print('port %s : %.4f m' % (port, relpos[port]))
                            # print('TL %s : %.4f m' % (TL, GTLs[TL][0]['relpos']))
                            found = True
                        # else: cannot resolve further
                else:
                    # check to see if this resolves furtur segments
                    if port not in relpos:
                        if k: # 2nd port
                            relpos[port] = GTLs[TL][0]['relpos'] + GTLs[TL][0]['length']
                        else:
                            relpos[port] = GTLs[TL][0]['relpos']
                        found = True
                        # print('port %s : %.4f m' % (port, relpos[port]))
                        
        # PP.pprint(relpos)
        
    #------------------------------ iterate over GTL again with the final GTLkws
    for TL, (GTLkws, terminations) in GTLs.items():
        
        CT.AddBlockGTL(TL, **GTLkws)
        
        for port, termination in terminations:
            CT.Terminate(TL + '/' + port, **termination)
                    
    #TODO: auto process rel positions from portnames and lengths

    #---------------------------------------------------- find all the portnames
    portnames = {}
    for TLid, TLkws in CT.TLs.items():
        for port in TLkws['ports']:
            if port not in portnames:
                portnames[port] = []
            portnames[port].append(TLid)
    
    internal, external, orphans = {}, {}, []
    for port, TLids in portnames.items():
                    
        if any([(TLid+'/'+port in CT.T) for TLid in TLids]):
            pass # already terminated
        
        elif port == '[sc]':
            # terminate the short circuit(s)
            for TLid in TLids:
                CT.Terminate(TLid+'/[sc]',Z=0)
        
        elif port == '[oc]':
            for TLid in TLids:
                CT.Terminate(TLid+'/[oc]',Y=0)
           
        elif len(TLids) > 1:
            # connect multiples
            CT.Connect(*(TLid+'/'+port for TLid in TLids))
                            
        else:
            # excite the remaining orphan ports  (needed to extract the 
            # S-matrix) and keep track of the internal/external portnames
            orphans.append(TLids[0]+'/'+port)
            CT.Excite(orphans[-1], 0.)
            internal[port] = orphans[-1]
            external[orphans[-1]] = port
                    
    #----------------------------------------------------- extract the SZ matrix
    S = CT.Get_Coeffs(orphans, orphans)
#     SZ =  sc.Scatter(fMHz=fMHz, Zbase=Zbase, 
#                       Portnames=[external[p] for p in orphans], 
#                       S=S[0])
    
    #------------------------------- check Portnames match and sort if available
#     if Portnames:
#         # verify
#         if len(Portnames) != len(orphans) or any(
#                      [internal[p] not in orphans for p in Portnames]):
#             warnings.warn('Portnames do not match')
#         else:
#             SZ.sortports(Portnames)
#              
#     return CT, SZ

