################################################################################
#                                                                              #
# Copyright 2015, 2016, 2017, 2018, 2019, 2020                                 #
#                                                                              #
#                   Laboratory for Plasma Physics                              #
#                   Royal Military Academy                                     #
#                   Brussels, Belgium                                          #
#                                                                              #
#                   Fusion for Energy                                          #
#                                                                              #
#                   EFDA / EUROfusion                                          #
#                                                                              #
#                   ITER Organisation                                          #
#                                                                              #
# Author : frederic.durodie@rma.ac.be                                          #
#                          @gmail.com                                          #
#                          @ccfe.ac.uk                                         #
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


"""
module TouchStoneClass
======================

Frederic Durodie 2009-04-21

+-----------+---------------------------------------------------------------------+
| date      | Modification                                                        |
+===========+=====================================================================+
|           |                                                                     |
| 19 Dec 19 | NEED TO CORRECT BUG FOR CONSEQUTIVE CALLS TO .Datas()               |
|           |                                                                     |
+-----------+---------------------------------------------------------------------+
| 03 Feb 20 | added LINE_ENDING                                                   |
|           |   self.LINE_ENDING = '\r\n' [default DOS line endings]              |
|           | see also parameter line_ending='\r\n' in WriteFile()                |
+-----------+---------------------------------------------------------------------+
| 29 Jan 20 | added COEFFS_PER_LINE                                               |
|           |   self.COEFFS_PER_LINE = 4 [default]                                |
|           | see also parameter coefs_per_line=4 in WriteFile()                  |
+-----------+---------------------------------------------------------------------+
| 15 Jan 20 | finally corrected reading (HFSS) of not normalized touchstone files |
+-----------+---------------------------------------------------------------------+
| 28 Aug 19 | added reading (HFSS) of not normalized touchstone files             |
+-----------+---------------------------------------------------------------------+
| 08 Aug 19 | 1) refactor internal representation:                                |
|           |                                                                     |
|           |    from list of S-array to numpy.array of S-arrays                  |
|           |                                                                     |
|           |    this should simplify and speed up the code                       |
|           |                                                                     |
|           |      self.datas[k] -> self.datas[k,:,:]                             |
|           |      sij = [d[i,j] for d in self.datas] -> sij = self.datas[:,i,j]  |
|           |                                                                     |
|           | 2) return arrays instead of lists                                   |
+-----------+---------------------------------------------------------------------+
| 11 Dec 18 | 1) corrected (* unexplainable) bugs in CombimeWithEven|Odd          |
|           |    *) apparently 6 is 6 was false while 6 == 6 is True ??           |
|           | 2) replaced internal calls to  .Add_freq -> .Add_Data               |
+-----------+---------------------------------------------------------------------+
| 06 Oct 18 | Added code to detect source of touchstone file when it is read in   |
|           | (HFSS / MWS) and extract the variables / parameters that are saved  |
|           | in the file: TouchStone.ts_source (str) and TouchStone.ts_variables |
|           | (dict([(<variable>:[<value>,<unit>]), ... ]))                       |
+-----------+---------------------------------------------------------------------+
| 25 Sep 18 | corrected bug in Get_Sccatter when reference impedance is modified  |
+-----------+---------------------------------------------------------------------+
| 25 Sep 18 | corrected bug in Get_Sccatter when the defaut portnames are used    |
+-----------+---------------------------------------------------------------------+
| 29 Apr 15 | added code to limit the range of interpolation if desired           |
+-----------+---------------------------------------------------------------------+
| 05 Apr 15 | rewritten __str__ and WriteFile so that one can also have a string  |
|           |   returned                                                          |
|           | read_raw_touchstone (is now obsolete) and is replaced by the pair   |
|           |   str2touchstone and ReadFile                                       |
|           | __init__ has the kwarg data which is a string which is assumed to   |
|           |   be the content of a touchstone file                               |
+-----------+---------------------------------------------------------------------+
| 11 Mar 15 | fork : imports scatter3a                                            |
+-----------+---------------------------------------------------------------------+
| 04 Feb 14 | convert to python 3.x                                               |
+-----------+---------------------------------------------------------------------+
| 02 Oct 13 | cleanup of 'errors' (thanks to Canopy (pyflake))                    |
+-----------+---------------------------------------------------------------------+
| 13 Feb 11 | added code to read MWS data (unfinished)                            |
+-----------+---------------------------------------------------------------------+
| 04 Jan 11 | completed the WriteFile method                                      |
|           | added __copy__() and __deepcopy__() methods                         |
|           | added convertto() method                                            |
+-----------+---------------------------------------------------------------------+
| 13 Dec 10 | added coded to assemble matrices from even and odd excitation for   |
|           | symmetric half matrices                                             |
|           | added !SHAPE, !PART and !NUMBERING directives / properties          |
|           | added comments ...                                                  |
+-----------+---------------------------------------------------------------------+
| 21 Sep 10 | corrected errors (due confusion between multiplying <ndarray>s or   |
|           | <ndmatrix>s in _convert_type_).                                     |
|           | added parameter zref in Datas                                       |
+-----------+---------------------------------------------------------------------+
| 17 Sep 10 | changed interpolation : now using interp1d (not yet ...)            |
+-----------+---------------------------------------------------------------------+
| 14 Sep 10 | added a feature called !MARKERS (as comment in a TS file) :         |
|           | Touchstone.Markers(unit=) return a list of frequencies set on the   |
|           | !MARKERS f1 f2 f3 ...                                               |
|           | This may be usefull to tag the data for special frequencies (e.g.   |
|           | Topica data frequencies for an interpolated TSF)                    |       
+-----------+---------------------------------------------------------------------+

"""

__updated__ = '2020-01-29 11:58:13'


#===============================================================================
#                        define versions
#===============================================================================

__version_info__ = tuple(__updated__.split()[0].split('-') + ['(array version)'])
__version__ = '.'.join(__version_info__)

_debug_ = False

#===============================================================================
#                        define error classes
#===============================================================================

class TouchStoneError(Exception):
    """
    Error class for TouchStone class objects (kept to the bare minimum)
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

#===============================================================================
#                        import section
#===============================================================================

import numpy as np

from numpy.linalg import inv
from scipy.interpolate import interp1d, splrep, splev
from scipy.constants import speed_of_light

import re, os, bisect, time, copy
import sqlite3

from functools import reduce
from pyRFtk import scatter3a as sc
from Utilities.ppdict import ppdict
from Utilities.printMatrices import printMA
pfmt = '%12.7f %+9.4f'

assert sc.__version__ >= '2012.05.02', TouchStoneError(
    'Scatter version >= 2012.05.02 required')

#===============================================================================
#  the Touchstone Class
#===============================================================================

class TouchStone:

    """
    if there is a valid filepath the whole touchstone file is read;
    else if file path is None an empty instance is created to which elements can be
    added in this case the tformat should be a valid touchstone format string;

    if a filepath and tformat are given the internal data representation is
    converted to  the supplied tformat string :

    tformat is a valid touchstone file options string (e.g. "MHz S RI R 50" )

    the TouchStone class object has following attributes

    Attributes
    ^^^^^^^^^^

    nports : int
        number of ports

    freqs : [<float>]
        list of frequencies

    fscale : float
        multiplier to scale the frequencies in *freqs* to Hz

    format : string
        one of 'MA', 'RI' or 'DB' but the internal representation is always complex
        (RI)

    mtype : string
        one of 'S', 'Y' or 'Z' which is the representation of *datas*

    datas : [ [ [complex]*nports ]*nports ]*len(freqs)
        an array nports x nports matrices representing 'S', 'Y' or 'Z' (see *mtype*)

    zref : float
        reference impedance (relevant for *mtype* == 'S')

    markers : [<float>]
        list of marker frequencies in the supplied frequency unit (see *fscale*)
        
    | self.shape     : a tuple clarifying the port layout (rows,columns) of a 2D map
    | self.part      : 'left', 'right', 'top' or 'bottom'
    | self.numbering : the numbering pattern of the port layout on the 2D map :
    |                 ('left'|'right','down'|'up') | ('down'|'up','left'|'right')
    | self.comments  : comments as a list of 1-liner strings that will be written at
    |                  the beginning of the .sNp file
    |                  !REM comment[0]
    |                  !REM comment[1]
    | self.ports     : port names as will be put in the touchstone file
    |                  !PORTS name1, name2, ... , nameN
                     
    """

    def __init__(self,filepath=None,tformat=None, data=None, **kwargs):
        
        self.COEFFS_PER_LINE = 4
        self.LINE_ENDING = '\r\n' # so the files a compatible with DOS
        
        # override/set portnames
        portnames = kwargs.pop('portnames', [])
        comments = kwargs.pop('comments','')
        
        if kwargs:
            raise ValueError('unrecognized kwarg(s): %r' % kwargs)
            
        # this information is additional to the basic touchstone file structure
        self.filepath = None
        self.markers = np.array([])      #  !MARKERS 10.0 30.0 50.0
        self.shape = None      #  !SHAPE 3 2        
        self.part = None       #  !PART Left
        self.numbering = None  #  !NUMBERING Left Up
        self.nports = 0
        self.freqs = np.zeros((0,), dtype=np.float64)      # empty list
        self.datas = np.zeros((0,0,0),dtype=np.complex128) # empty list
        if comments:
            self.comments = comments.split('\n')
        else:
            self.comments = []     #  !REM comments ...
        self.ports = []        #  !PORTS name1, name2, ...
        self.ts_source = ''
        self.ts_variables = {}

        # defaults
        self.funit  = 'GHZ'
        self.fscale = 1.e9
        self.mtype  = 'S'
        self.format = 'MA'
        self.zref   = 50.
        
        # for mixed impedance touchstones
        self.Zcs    = np.zeros((0,0),dtype=np.complex128)
        self.Gms    = np.zeros((0,0),dtype=np.complex128)
        
        if filepath:
            if data :
                raise ValueError('TouchStone.__init__ : filpath exor data')
            if os.path.isdir(filepath):
                # recieved a directory : so this could be a MWS generated matrix
                self.readMWSdata(filepath)
                self.filepath = filepath
                
            else:
                # if is a file : check the extension
                try:
                    # self.raw_read_touchstone(filepath)
                    self.ReadFile(filepath)
                    
                except IOError:
                    raise IOError('TouchStoneClass3: IOError on reading \n  %s'
                                   % os.path.abspath(filepath))
                self.filepath = filepath
                
        elif data:
            self.str2touchstone(data)
            
        if tformat:
            if filepath or data:
                self.convertto(tformat,inplace=True)
            else:
                self._get_tsf_fmt_(tformat) # sets self. -fscale -funit -zref -mtype

        # override/set portnames if requested and possible
        if portnames:
            if isinstance(portnames,list):
                if len(portnames) == self.nports:
                    self.ports = portnames[:]
                else:
                    print(len(portnames),self.nports)
                
            elif re.findall('^[^%]*(%[0-9]*\.{0,1}[0-9]*d)[^%]*$', portnames):
                # allow just one % format field
                self.ports = [portnames % (k+1) for k in range(self.nports)]
                   
#===============================================================================
#
# c o p y 
#

    def copy(self):
        """
        returns a deepcopy of of the TouchStone object
        """
        return copy.deepcopy(self) # make anyway a deep copy

#===============================================================================
#
# _ _ r e p r _ _
#

    def __repr__(self, data=True):
                
        """
        write a touchstone file
        """
        s = ''
        eoln = self.LINE_ENDING
        
        if not data:
            f = self.Freqs()
            fmin, fmax, dfs = np.min(f), np.max(f), np.diff(f)
            
            s += '! %d ports %d frequencies from %.f to %.f %s' % (
                self.nports, len(self.Freqs()),fmin,fmax,self.funit) + eoln
            
        s += '! TouchStone file (TouchStoneClass version %s)' % __version__ + eoln
        s += '! Date : %s' % time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + eoln
        
        if len(self.comments) > 0 :
            for line in self.comments :
                s += '!REM %s' % line + eoln

        if len(self.ports) > 0 :
            s += '!PORTS ' + ', '.join(self.ports) + eoln
            
        if len(self.markers) :
            s += '!MARKERS '+(('%.3f '*len(self.markers))%tuple(self.markers))+eoln
            
        if self.shape :
            s += '!SHAPE %d %d' % self.shape + eoln
            
        if self.part:
            if type(self.part) is tuple:
                s += '!PART '+ ('%s '*len(self.part)) % self.part + eoln
            else:
                s += '!PART %s' % self.part + eoln
                
        if self.numbering:
            if type(self.numbering) is tuple:
                s += '!NUMBERING '+('%s '*len(self.numbering))%self.numbering+eoln
            else:
                s += '!NUMBERING %s' % self.numbering+eoln
                        
        if self.ts_source:
            s += '! source: %s' % self.ts_source+eoln
        
        if self.ts_variables:
            v = ppdict(self.ts_variables, label='variables: ')
            for al in v.split('\n'):
                s += '! ' + al + eoln
            s += eoln
        
        s += '# %s' % self.FormatStr() + eoln

        if not data:
            return s
        
        freq_fmt = '%9.6f '
        freq_fmt_len = 10
        
        if True:
            # find the optimal frequency format that is able to represent the
            # smallest frequency step as well as the maximun frequency
            
            if len(self.freqs) > 1:
                for df in list(set(sorted(np.diff(self.freqs)))):
                    if df > 0.:
                        break
            else:
                df = self.freqs[0]
            
            m0, dm = 15, 3
            m = m0
            df0 = round(df,m)
            while m >= 0 and df0 == round(df,m):
                m -= dm
            m += dm
            freq_frac = '.%df' % m
            n = len(('%'+freq_frac) % np.max(self.freqs))
            freq_fmt = ('%%%d' % n) + freq_frac + ' '
            freq_fmt_len = n + 1
        
        CN = self.COEFFS_PER_LINE
        CN = self.nports if CN == 0 else CN
        for fk, d in zip(self.freqs, self.datas):
            s += freq_fmt % fk

            for kr, di in enumerate(d):
                for kc, dij in enumerate(di):
                    
                    if self.format == 'RI':
                        s += '   %+19.12g %+19.12g' % (np.real(dij),np.imag(dij))
                        
                    elif self.format == 'DB':
                        s += '   %+11.7f %+8.3f'% (20.*np.log10(np.abs(dij)),
                                                   np.angle(dij,1))
                    else: # can only be MA
                        s += '   %+17.7e %+8.3f' % (np.abs(dij),np.angle(dij,1))
                        
                    if (kc % CN is (CN-1)) and (kc is not self.nports-1):
                        if kc is (CN-1):
                            s += ' !    row %d' % (kr+1)
                        s += eoln+' '*freq_fmt_len
                s += eoln
                if kr is not self.nports-1:
                    s += ' '*freq_fmt_len
                    
        s += eoln # add a trailing return
        
        return s
     
#===============================================================================
#
# _ _ s t r _ _
#
    
    def __str__(self):
        return self.__repr__()

#===============================================================================
#
# i n f o
#
    def info(self):
        return self.__repr__(data=False)
                   
#===============================================================================
#
# r e a d M W S d a t a
#

    def readMWSdata(self,dirpath):
        """
        import MWS generated data :
        file patterns are amplitudes a<i>(1)<j>(1).sig and
                          phases p<i>(1)<j>(1).sig for portmode 1
        """

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++                                                        ++")
        print("++               TouchStone.readMWSdata :                 ++")
        print("++                                                        ++")
        print("++    R O U T I N E   N O T   V A L I D A T E D   . . .   ++")
        print("++                                                        ++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        def indices(fstr):
            return tuple(map(int,re.findall('[a|p]([0-9]+)\(1\)([0-9]+)\(1\)\.sig',
                                            fstr)[0]))

        def readdata(regex, freqs, fscale):
            """
            '^a([0-9]+\(1\))+.sig$'
            '^p([0-9]+\(1\))+.sig$'
            '^zline([0-9]+\(1\))+.sig$'
            """
            data = {}
            fcheck = (len(freqs) is not 0)

            tfiles = [x for x in os.listdir(dirpath) if len(re.findall(regex,x))]
            print(tfiles)
            
            if len(tfiles) is 0:
                print('hmm ... maybe the data is a sqlite database ...')
                print('... trying to open Storage.sdb ...')
                sdb = sqlite3.connect(os.path.join(dirpath,  # @UndefinedVariable
                                                   'Result/Storage.sdb'))
                cur = sdb.cursor()
                print('sqlite3.connection.cursor.arraysize :', cur.arraysize)
                
                print('\nselect * from sqlite_master;\n')
                cur.execute('select * from sqlite_master;')
                anss = cur.fetchall()
                for ka,ans in enumerate(anss):
                    for ka2, ans2 in enumerate(ans):
                        print('%4d %4d %-s' % (ka,ka2,str(ans2)))
                    print()
                
                print('\n.tables\n')
                cur.execute("""SELECT name FROM sqlite_master 
                               WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'
                               UNION ALL 
                               SELECT name FROM sqlite_temp_master 
                               WHERE type IN ('table','view') 
                               ORDER BY 1
                             """)
                while True:
                    anss = cur.fetchone()
                    if anss:
                        for ka,ans in enumerate(anss):
                            print('%4d %-s' % (ka,str(ans)))
                        print()
                    else:
                        break
                    
                print('\nselect * from SigHeader;\n')
                cur.execute("SELECT name, length, table_id FROM SigHeader WHERE name LIKE 'a%';")
                # cur.execute("SELECT name, length, table_id FROM SigHeader;")
                while True:
                    anss = cur.fetchone()
                    if anss:
                        for ka,ans in enumerate(anss):
                            print('%4d %-s' % (ka,str(ans)))
                        print()
                    else:
                        break

                cur.close()
                sdb.close()
                
            else:
                for fname in tfiles:
                    f = open(os.path.join(dirpath,fname),'r')
                    # this returns the tuple (i,j) for the filename 'a<i>(1)<j>(1).sig'
                    tkey = tuple(map(int,re.findall(regex,fname)[0]))
                    if tkey in data.keys():
                        # this can however never happen ...
                        raise TouchStoneError('TouchStone.readMWSdata() :'
                                              'Duplicate indices ??? %s' % str(tkey))
                    
                    if not fcheck: 
                        freqs = [] # fMHzs was not set yet
                    data[tkey] = []
                    kf = 0
                    for aline in f.readlines():
                        if aline[0] not in [dk for dk in '0123456789']:
                            try:
                                if aline.index('Scale:') is 0:
                                    if fscale is 0: # so it was not set yet
                                        fscale = float(aline.split()[1])
                                    else:
                                        if fscale != float(aline.split()[1]):
                                            raise TouchStoneError(
                                                'TouchStone.readMWSdata() :'
                                                'frequency scale not consistent (%s)'
                                                % os.path.basename(fname))
                            except ValueError:
                                # if 'Scale' is not in aline then a ValueError exception is generated
                                pass
                        else:
                            e = aline.split()
                            freq = float(e[0])
                            if fcheck:
                                if freq != freqs[kf]:
                                    raise TouchStoneError(
                                        'TouchStone.readMWSdata() :'
                                        'frequency data not consistent (%s)'
                                        % os.path.basename(fname))
                                kf += 1
                            else:
                                freqs.append(freq)
                            data[tkey].append(float(e[1]))
                
                for k, v in data.items():
                    data[k] = np.array(v)
                    
                return np.array(freqs), fscale, data

        self.freqs = np.zeros((0,), dtype=np.float64)
        self.fscale = 0
        
        self.freqs, self.fscale, adata = readdata('^a([0-9]+)\(1\)([0-9]+)\(1\)\.sig$',
                                                  self.freqs,self.fscale)
        self.freqs, self.fscale, pdata = readdata('^p([0-9]+)\(1\)([0-9]+)\(1\)\.sig$',
                                                  self.freqs,self.fscale)
        
        freqs1, fscale1 = [], 0
        freqs1, fscale1, zdata = readdata('^zline([0-9]+)\(1\)\.sig$',freqs1,fscale1)
        tkeys = adata.keys()
        tkeys.sort()
        if tkeys[-1][0] is not tkeys[-1][1]:
            raise TouchStoneError('TouchStone.readMWSdata() :'
                                  'matrix data not square (%s)' % str(tkeys[-1]))
        self.nports = tkeys[-1][0]

        # we assume that MW data is always linear S (but what was the base impedance ?)
        # well it is probably given in zline<i>(1).sig
        #
        # (Vm - Z0m Im)/sqrt(Z0m) = sum(n=1:N, Smn (Vn + Z0n In)/sqrt(Z0n) )
        # (Am + Bm)/sqrt(Z0m) - sqrt(Z0m) (Am - Bm) / Z0 = sum(n=1:N, Smn ((An + Bn)/sqrt(Z0n) + sqrt(Z0n) (An - Bn) / Z0)
        # (1/sqrt(Z0m) - sqrt(Z0m)/Z0) Am + (1/sqrt(Z0m) + sqrt(Z0m)/Z0) Bm = sum(n=1:N, Smn ((1/sqrt(Z0n) + sqrt(Z0n)/Z0) An + (1/sqrt(Z0n) - sqrt(Z0n)/Z0) Bn))
        # (D[(1/sqrt(Z0m) + sqrt(Z0m)/Z0) - S D[1/sqrt(Z0m) - sqrt(Z0m)/Z0]) B = (- D[1/sqrt(Z0m) - sqtr(Z0m)/Z0] + S D[(1/sqrt(Z0m) + sqrt(Z0m)/Z0)) A
        # B = {inv(D[(1/sqrt(Z0m) + sqrt(Z0m)/Z0) - S D[1/sqrt(Z0m) - sqrt(Z0m)/Z0]) (- D[1/sqrt(Z0m) - sqrt(Z0m)/Z0] + S D[(1/sqrt(Z0m) + sqrt(Z0m)/Z0))} A

        Z0 = self.zref
        
        # D = np.diag([zdata[k][0] for k in sorted(zdata.keys())])
        DP = np.diag([1/np.sqrt(zdata[k][0]) + np.sqrt(zdata[k][0])/Z0
                      for  k in sorted(zdata.keys())]) # np.eye(len(zdata.keys())) + D / Z0
        DM = np.diag([1/np.sqrt(zdata[k][0]) - np.sqrt(zdata[k][0])/Z0
                      for  k in sorted(zdata.keys())]) # np.eye(len(zdata.keys())) - D / Z0

        # now produce the S matrix for every fMHz

        self.datas = np.zeros((0, self.nports, self.nports), dtype=np.complex128)
        
        Sk = np.zeros((self.nports,self.nports),dtype=np.complex128)
        for kf1, fMHz in enumerate(self.freqs):
            for kr in range(self.nports):
                for kc in range(self.nports):
                    try :
                        amp = adata[(kr+1,kc+1)][kf1]
                        pha = pdata[(kr+1,kc+1)][kf1]
                        Sk[kr,kc] = amp * np.exp(1j * pha * np.pi / 180.)
                    except:
                        print(kf1,kr,kc)
                        raise
                
            Sk = np.linalg.inv( DP - Sk @ DM ) @ ( Sk @ DP - DM)
            Sk = np.array([[[sij] for sij in si] for si in Sk])
            self.datas = np.append(self.datas, Sk, axis=0)
                    
        try:
            self.funit = ['HZ','KHZ','MHZ','GHZ'][int(np.log10(self.fscale)/3)]
        except:
            raise TouchStoneError('TouchStone.readMWSdata() :'
                                  'could not convert fscale to funit (%s)'
                                  % str(self.fscale))
        
#===============================================================================

    def convertto(self,tformat,inplace=False):
        """
        Convert the data to the format given in the format string

        Arguments :
          tformat : a valid TouchStone format string
          inplace [False] : normally the conversion is done on a (deep)copy of the
          TouchStone object. If the inplace argument is set to True the conversion
          is performed on the object itself.

        Output :
            if inplace is False :
                the converted TouchStone object
            else
                None

            
        """
        
        # oldformat = self.format
        oldmtype  = self.mtype
        oldzref   = self.zref
        oldfscale = self.fscale
        # oldfunit  = self.funit
        
        if inplace:
            tsf = self
        else:
            tsf = self.copy()

        tsf._get_tsf_fmt_(tformat) # this updates self.format .mtype .zref .funit and .fscale
        
        for k, M in enumerate(self.datas):
            tsf.datas[k] = tsf._convert_type_(M,
                                              tsf.mtype,
                                              tsf.zref,
                                              fromtype=oldmtype,
                                              fromzref=oldzref)
        if tsf.fscale != oldfscale:
            tsf.freqs = tsf.freqs * oldfscale / tsf.fscale

        if not inplace:  
            return tsf
        
#===============================================================================

    def Scaler(self, unit=None):
        """
        returns the scaler to be applied (multiplied) to frequency given in
        unit in order to get it in the same units as the frequency in the 
        TouchStoneClass instance
        """
        if not unit:
            return 1
        else:
            try:
                s = ({'HZ':1e+0,'KHZ':1e+3,'MHZ':1e+6,'GHZ':1e+9}[unit.upper()]
                     / self.fscale)
            except:
                raise TouchStoneError('TouchStoneClass.Scaler() :'
                                      'unknown units ' + str(unit)) 
            return s
    
#===============================================================================

    def _convert_type_(self, M, mtype=None, zref=None, unit=None,
                       fromtype=None, fromzref=None, fromunit=None):
        """
        converts the matrix of the TouchStoneClass' type into one of
           mtype ('Z','S','Y')
        """
##        print "_convert_type:"
##        print "type(M)  : %s, (%d)" % (str(type(M)),len(M))
##        print "mtype    :", mtype
##        print "zref     :", zref
##        print "unit     :", unit
##        print "fromtype :", fromtype
##        print "fromzref :", fromzref
##        print "fromunit :", fromunit
        
        if not fromtype:
            fromtype0 = self.mtype
        else:
            fromtype0 = fromtype.upper()

        if not mtype:
            mtype0 = self.mtype
        else:
            mtype0 = mtype.upper()
            
        if not fromzref:
            fromzref0 = self.zref
        else:
            fromzref0 = fromzref
            
        if not zref:
            zref0 = self.zref
        else:
            zref0 = zref

        ## convert frequency units if required
            
        if not fromunit:
            fromfscale0 = self.fscale
            # fromfunit = self.funit
        else:
            # fromfunit = fromunit.upper()
            fromfscale0 = {'HZ':1e+0,'KHZ':1e+3,'MHZ':1e+6,'GHZ':1e+9}[fromunit]

        if not unit:
            fscale0 = self.fscale
            unit= self.funit
        else:
            unit = unit.upper()
            fscale0 = {'Hz':1e+0,'KHZ':1e+3,'MHZ':1e+6,'GHZ':1e+9}[unit]

        if fscale0 != fromfscale0:
            self.freqs = self.freqs * fromfscale0 / fscale0
            self.funit = unit
            
        assert (mtype0 in ['S','Y','Z']) and (fromtype0 in ['S','Y','Z'])

        assert (zref0 > 0) and (fromzref0 > 0)
        
        if fromtype0 == 'S':
            # need to return scattering matrix elements
            if mtype0 == 'S':
                if fromzref0 == zref0:
                    # nothing to do
                    return M
                else:
                    # convert the zref's
                    I = np.eye(self.nports)
                    # 2010-09-21 FDe : new code converts avoiding using Z
                    p = (fromzref0/zref0 + 1.)/2.
                    m = p - 1.
##                    print type(p * I + m * M)
##                    print type(inv(p * I + m * M))
                    return np.dot(inv(p * I + m * M),(m * I + p * M))
                    # old code uses Z (which does not always exists)
                    # Z = fromzref0*dot((I + M),inv(I - M))
                    # ZI = zref0 * I
                    # return (Z - ZI)*inv(Z + ZI)

            elif mtype0 == 'Z':
                # convert from S to Z
                I = np.eye(self.nports)
                return fromzref0*np.dot((I + M),inv(I - M))
                
            else: # can only be mtype0 == 'Y':
                # convert from S to Y
                I = np.eye(self.nports)
                return np.dot((I - M),inv(I + M))/fromzref0
            
        elif fromtype0 == 'Z':
            # need to return impedance matrix elements"
            if mtype0 == 'Z':
                # nothing to do
                return M
            elif mtype0 == 'S':
                # convert from Z to S
                ZI = zref0 * np.eye(self.nports)
                return np.dot((M - ZI),inv(M + ZI))
            else: # can only be mtype0 == 'Y':
                # convert from Z to Y
                return inv(M)
            
        else: # can only be fromtype0 == 'Y':
            # need to return admittance matrix elements
            if mtype0 == 'Y':
                # nothing to do
                return M
            elif mtype0 == 'Z':
                # convert from Y to Z
                return inv(M)
            elif mtype0 == 'S':
                # convert from Y to S
                ZI = zref0 * np.eye(self.nports)
                return np.dot((M + ZI),inv(M - ZI))
        
#===============================================================================
        
    def _convert_fmt_(self,adatum,fmt=None):
        """
        converts the internal representation, complex RI, to the requested
        format, fmt = 'RI', 'MA' or 'DB'. For the latter two formats a list
        of 2 elements is returned while for the 'RI' format the complex
        representation is kept.
        """
        if not fmt:
            fmt0 = self.format
        else :
            fmt0 = fmt.upper()

        assert fmt0 in ['RI','MA','DB']
        
        if fmt0 == 'RI':
            return adatum  # internal representation is already RI
        
        else:
            if isinstance(adatum, (list, np.ndarray)): 
                A = np.abs(np.array(adatum))
                P = np.angle(np.array(adatum),deg=1)
                if fmt0 == 'DB':
                    return ((20*np.log10(A)),P)
                else:  # can only be 'MA'
                    return (A,P)
            else:
                A = np.abs(adatum)
                P = np.angle(adatum,1)
                if fmt0 == 'DB':
                    return (20*np.log10(A),P)
                else:  # can only be 'MA'
                    return (A,P)
        
#===============================================================================

    def Datas(self, freqs=None, # [list/array of] frequencies
                    unit=None,  # frequency unit ['Hz', ...  , 'GHz']
                    mtype=None, # matrix type ['S',  Z', 'Y', 'G', 'H'] 
                    elfmt=None, # format ['RI', 'MA', 'DB']
                    elems=None, # which coeffs [ (i1,j1), (i2,j2), ... ]
                    kind=None,  # interpolation ['RI', 'MA', 'DB' ] ?
                    order=None, # interpolation order [1 ... 3]
                    zref=None,  # reference Z0 of the returned data
                    maxdf=-1    # max error on available frequencies
             ):
        """
        returns a complex matrix reprsenting the measured object as 'S', 'Z',
        'Y', 'G', 'H' for the given frequency with units (Hz|kHz|...) if an "S"
        type element is required then care should be take to supply zref as well
        in order to avoid confusion in which base the S is valid.
        if maxdf < 0 interpolation is allowed otherwise interpolation is allowed
        if it does not require the nearest frequency to be more distant than maxdf
        """
        
        if not mtype:
            mtype0 = self.mtype
        else:
            mtype0 = mtype.upper()
            
        if not zref:
            zref0 = self.zref
        else:
            zref0 = zref
            
        assert mtype0 in ['S','Z','Y']

        if mtype == 'S' and not zref:
            print("(TouchStone.Datas) Warning : use the zref parameter to make"
                  " sure S is correct.")
        
        if unit is None:
            unit = {1e+0:'HZ',1e+3:'KHZ',1e+6:'MHZ',1e+9:'GHZ'}[self.fscale]
            
        fscale = self.Scaler(unit=unit)

        ## interpolate (if required) and convert the matrix type
        
        if freqs is None:
            
            freqs0 = np.array(self.freqs)
            rdatas = self.datas[:,:,:]
                
        else:
            
            if isinstance(freqs, (list,np.ndarray)):
                freqs0 = np.array(freqs) * fscale
            else: # we assume type(freqs) in [int,float(or compatible)]:
                freqs0 = np.array([freqs]) * fscale
                        
            # this will enforce that the requested frequencies do not
            # differ more than maxdf from the available frequencies
            if maxdf >= 0. :
                smaxdf = maxdf * fscale # convert to the right units
                for freq in freqs0:
                    if np.min( np.abs(freq - self.freqs)) > smaxdf:
                        raise ValueError(
                              'TouchStoneClass3.Datas : requested frequency '
                              '%f difference with touchstone data larger than '
                              '%f %s' % (freq,maxdf,unit))
                    
            rdatas = self.Resample(freqs0,
                                   funit=self.funit,
                                   interpolation=kind,
                                   order=order).datas # Resample returns a touchstone object
        NEWCODE=False
        if NEWCODE:                           
            rdatasC = 1j*np.zeros(rdatas.shape) 
            for k in range(len(rdatas)):
                rdatasC[k] = self._convert_type_(rdatas[k],mtype=mtype0, zref=zref0)
        else:
            for k in range(len(rdatas)):
                rdatas[k] = self._convert_type_(rdatas[k],mtype=mtype0, zref=zref0)
            
        ## pick out the requested matrix elements as required

        cdatas = []
        if elems is None:
            cdatas = rdatasC if NEWCODE else rdatas
            
        else:
            if type(elems) in (tuple, int):
                if type(elems) is int:
                    telems = [(elems / self.nports, elems % self.nports)]
                else:
                    telems = [elems]
                                    
            elif type(elems) is list:
                telems = []
                for anelem in elems:
                    if type(anelem) is int:
                        telems.append((anelem / self.nports, 
                                       anelem % self.nports))
                    else:
                        telems.append(anelem)
                        
            for telem in telems:
                cdatas.append([])
                for adatum in rdatas:
                    cdatas[-1].append(np.array(adatum)[telem])

        ## convert to the requested format and return the answer
        
        # print('Datas ->', cdatas.shape)
        if (elems is not None) and (len(cdatas) == 1): ## only one element ?
            cdatas = cdatas[0]
            
        if len(cdatas) == 1: ## only one frequency ?
            cdatas = cdatas[0]
        # print('Datas ->', cdatas.shape)
        
        return self._convert_fmt_(cdatas,elfmt)
                                
#===============================================================================

    def FormatStr(self):
        """
        returns the format of the touchstone data as a format string
        """
        return  "%s %s %s R %.3f" % (
                   {1e+0:'HZ',1e+3:'KHZ',1e+6:'MHZ',1e+9:'GHZ'}[self.fscale],
                   self.mtype, self.format, self.zref)
    
#===============================================================================

    def Freqs(self,unit=None):
        """
        returns the frequencies listed in the touchstone data
        """
        return self.freqs / self.Scaler(unit)
    
#===============================================================================

    def Markers(self, unit = None):
        """
        check if there was a MARKERS comment which contains marked frequencies
        """
        return self.markers / self.Scaler(unit)
                
#===============================================================================

    def Nports(self):
        """
        returns the number of ports
        """
        return self.nports

#===============================================================================

    def Add_Data(self, data, frequency, unit=None):
        """
        Adds a matrix data at frequency with unit
        
        no checks are done on the type of data inserted
        """
        # convert the data to an numpy.ndarray
        tdata = np.array(data)
            
        if len(self.freqs) is 0:
            if (len(tdata.shape) is not 2) or (tdata.shape[0] is not tdata.shape[1]):
                raise TouchStoneError('Add_Data : data is not a square 2D array')
            self.nports = tdata.shape[0]
        else:
            if tdata.shape != self.datas[0].shape:
                raise TouchStoneError('Add_Data : data is not compatible'
                                      ' with the data stored already')
                
        tfreq = frequency * self.Scaler(unit)
        
        k = bisect.bisect(self.freqs, tfreq)
        
        self.freqs = np.concatenate((self.freqs[:k], [tfreq], self.freqs[k:]))
        if len(self.datas):
            self.datas = np.concatenate((self.datas[:k], [tdata], self.datas[k:]))
        else:
            self.datas = np.array([tdata])

#===============================================================================

    def Add_freq(self, frequency, data):
        """
        Adds a matrix point at frequency
        """
        print()
        print("--------------------------------------------------------------")
        print("TouchStoneClass.Add_freq : obsolete function (will be removed)")
        print("-> use TouchStoneClass.Add_Data(data, frequency, unit=None) <-")
        print("--------------------------------------------------------------")
        print()
        
        self.Add_Data(data, frequency, unit=None)
            
#===============================================================================

    def Remove_freq(self,frequency,unit=None):
        """
        removes a frequency point from the data set (it in fact appears to remove
        the frequency entry equal to or preceeding the given frequency (assuming
        that the frequecy entries are ordered).
        """
        
        f = frequency * self.Scaler(unit)
        
        # remove the frequency closest to the given one
        k = bisect.bisect(self.freqs, f)
        if 0 < k < len(self.freqs):
            if (f - self.freqs[k-1]) <= (self.freqs[k] - f):
                k -= 1
        elif k == len(self.freqs):
            k -= 1
            
        # print('removing', tfreq, '/', self.freqs[idx], ' at', idx ,'/',len(self.freqs))
        self.Remove_idx(k)
    
#===============================================================================

    def Remove_idx(self,idx):
        """
        removes entry at frequency #idx from the data set 
        """
    
        self.freqs = np.concatenate((self.freqs[:idx], self.freqs[(idx+1):]))
        self.datas = np.concatenate((self.datas[:idx], self.datas[(idx+1):]))
        
#===============================================================================

    def _get_tsf_fmt_(self,fstr):
        """
        analyses the format string as per Touchstone specification and sets the
        found key elements for the TochStoneClass instance
        """
        tlst = fstr.split()                ## split the line into items

        tfscale = None
        tfunit  = None
        tmtype  = None
        tformat = None
        tzref   = None
        
        while tlst:
            t = tlst.pop(0).upper()
            if t[-2:] == 'HZ':
                if not tfscale:
                    tfunit = t
                    try:
                        tfscale = {'HZ':1e+0,'KHZ':1e+3,'MHZ':1e+6,'GHZ':1e+9}[t]
                    except KeyError:
                        raise TouchStoneError('Unknown frequency unit "%s"' % (t))
                else:
                    raise TouchStoneError('Frequency unit defined multiple times')
            elif t in ['S','Z','Y','H','G']:
                if not tmtype:
                    tmtype = t
                else:
                    raise TouchStoneError('Data type defined multiple times')
            elif t in ['MA','DB','RI']:
                if not tformat:
                    tformat = t
                else:
                    raise TouchStoneError('Data format defined multiple times')
            elif t == "R":
                if not tzref:
                    try:
                        tzref = float(tlst[0])
                        tlst = tlst[1:]
                    except:
                        raise TouchStoneError('Base impedance format error (float)')
                else:
                    raise TouchStoneError('Base impedance defined multiple times')
            else:
                raise TouchStoneError('Unknown touchstone option "%s"' % t)

        ## check if we have all the necessary options : if not leave the settings
        ## these should always be present
        if tmtype:
            self.mtype = tmtype
        if tformat:
            self.format = tformat
        if tfscale:
            self.funit = tfunit
            self.fscale = tfscale          
        if tzref:
            self.zref = tzref

        # print '_get_tsf_fmt_ : %s %s %s R %f' % (self.funit,self.mtype,self.format,self.zref)
        
#===============================================================================

    def Get_Scatter(self, f, unit=None, maxdf=-1, **kwargs):
        """
        return a Scatter object from the data at frequency f with unit
            kwargs : 'Portnames' if not set TouchStone.ports will be used
                     'Zbase' if not set the TouchStone.zref will be used
        """

#         kwargs.setdefault('Portnames', self.ports)
#         kwargs.setdefault('Zbase', self.zref)
#         kwargs.update({'S' : self.Datas(f,
#                                         unit=unit,
#                                         mtype='S',
#                                         elfmt='RI',
#                                         zref = kwargs['Zbase'],
#                                         maxdf = maxdf
#                                        )
#                       })
#         print(kwargs)
        
        Portnames = kwargs.pop('Portnames', self.ports.copy())
        Zbase = kwargs.pop('Zbase', self.zref)
        if kwargs:
            raise ValueError('unrecognized kwarg(s): %r' % kwargs)
            
        S = self.Datas(f, unit=unit, mtype='S', elfmt='RI', zref = Zbase,
                       maxdf = maxdf)
        # print(S.shape)
        
        return sc.Scatter(fMHz = f * self.Scaler(unit) / self.Scaler('MHz'),
                          Zbase = Zbase, Portnames = Portnames, S = S)

##==============================================================================
            
    def Add_Scatter(self, SZ):
        """
        Add the data from a Scatter object at frequency f witn unit
        """

        M =  SZ.Get(self.mtype, Z0=self.zref)

        if not self.ports :
            self.ports = SZ.Portname
        else :
            assert self.ports == SZ.Portname, TouchStoneError(
                'Add_Scatter : Port names or ordering do not match')

        self.Add_Data(M, SZ.fMHz, 'MHz')       
            
##==============================================================================

    def Deembed(self,ports):
        """
        de-embeds the data set for the given ports
        ports: {portname: (length, Zfdr)} | {portname: length}
        portname: 'the name' | '#N' where N is the port number starting from 1 
        positive values for length move the reference plane inwards
        """

        fMHzs = self.Freqs('MHz')
        for k, fMHz in enumerate(fMHzs):
            SZ = self.Get_Scatter(fMHz, 'MHz')
            for port, LZ in ports.items():
                if isinstance(LZ, tuple):
                    L, Z = LZ
                else:
                    L, Z = LZ, self.zref
                if port[0] == '#':
                    port = SZ.Portname[int(port[1:]) - 1]
                SZ.trlAZV(port,-L,Z=Z)
            self.datas[k] = SZ.S[:,:]
            
        return self
    
##==============================================================================
            
    def Plot_vs_f(self, elems = None, kind='smith'):
        """
        plot vs. frequency of the elements in the data
        """
        
        # figure()
        if   kind.lower() == 'smith':
            print('smith')
        elif kind.lower() == 'polar':
            print('polar')
        elif kind.lower() == 'linlin':
            print('linlin')
        elif kind.lower() == 'linlog':
            print('linlog')
        elif kind.lower() == 'loglog':        
            print('loglog')
        else:
            raise TouchStoneError('Plot_vs_f : unknown kind (%s)' % str(kind))

        # pl.savefig()
        
#===============================================================================
#
# s t r 2 t o u c h s t o n e
#

    def str2touchstone(self,S):
        """
        read a touchstone file from a string S and sets the TouchStoneClass instance
           nports : the number of ports
           freqs  : the list of frequencies,
           datas  : the list of arrays (one for each frequency),
           mtype  : the type of data ('Z','Y','S','G' or 'H')
           format : the format ('MA','DB','RI')
           fscale : the frequency scaler (1e0,1e3,1e6,1e9)
           zref   : the reference impedance
        """

        GENERAL_MIXED, Gms, Zcs, Gmsk, Zcsk  = False, [], [], [], []
        
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        def readline(S):
            counter = 0
            S += '\n' # end on empty line
            while S:
                counter += 1
                aline, S = S.split('\n', 1)
                yield counter, aline.strip()
        
        iterator = readline(S)
        
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        regex_var = '(\$?[A-Za-z][\w\d_]*)'
        regex_float = '([+-]?\d*(\.\d*)?([eEdD]{1}[+-]?\d+)?)'
        regex_params = '! Parameters = \{([^\}]+)\}'
        
        def numbr(z):
            try:
                return float(re.sub('[dDe]', 'E', z))
            except ValueError:
                return z
        
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        def checkcomment():
            
            nonlocal GENERAL_MIXED
            
            # check if there is a reserved keyword
            # the only REM, PORTS, MARKERS, SHAPE, PART, NUMBERING are implemented
            # at present
            
            # an attempt is made to detect the source HFSS or CST/MWS and to extract
            # defined variables
                        
            counter, aline = next(iterator, (0,'<EOT>'))         
            if counter == 0 or aline == '' or aline[0] != '!':
                return counter, aline
                           
            tlist = re.findall('!REM (.+)',aline)
            if len(tlist) > 0:
                self.comments.append(tlist[0])
                
            tlist = re.findall('!PORTS (.+)',aline)
            if len(tlist) > 0:
                self.ports = [name.strip() for name in tlist[0].split(',')]
                
            tlist = re.findall('!MARKERS (.+)',aline)
            if len(tlist) > 0:
                self.markers = np.array([float(x) for x in tlist[0].split()])
                
            tlist = re.findall('!SHAPE (.+)',aline.upper())
            if len(tlist) > 0:
                self.shape = tuple([int(x) for x in tlist[0].split()[:2]])
                if len(self.shape) is 1:
                    self.shape.append(0)
                
            tlist = re.findall('!PART (.+)',aline)
            if len(tlist) > 0:
                self.part = tuple([x.lower() for x in tlist[0].split()[:2]])
                if len(self.part) is 1:
                    self.part = self.part[0]
            
            tlist = re.findall('!NUMBERING (.+)',aline)
            if len(tlist) > 0:
                self.numbering = tuple([x.lower() for x in tlist[0].split()[:2]])
                if len(self.shape) < 2:
                    self.shape = self.shape[0] # so numbering becomes type str rather than tuple
            
            # detect the source of the touchstone
            
            # HFSS ?
            tlist = re.findall('[Ee]xported from HFSS (.+)', aline)
            if len(tlist) > 0:
                self.ts_source = 'HFSS ('+tlist[0]+')'
                # check if there is a variable list
                counter, aline = next(iterator, (0,'<EOT>'))
                if aline.find('! Variables:') == 0:
                    # try and get all variables: these preceed the # line and start
                    # with an extra space: !  <variable> = <float><unit>\n
                    regex = ('!\s+'+regex_var+'\s+=\s+'+regex_float+'([A-Za-z]*)?')
                    ok = True
                    while ok:
                        counter, aline = next(iterator, (0,'<EOT>'))
                        # print('HFSS variable line :', aline[:-1])
                        if aline[0] != '!':
                            break
                        else: 
                            tlist = re.findall(regex,aline)
                            if len(tlist) > 0:
                                # print(tlist)
                                self.ts_variables[tlist[0][0]] = [
                                    numbr(tlist[0][1]), tlist[0][-1]]
                            else:
                                break
                    
                # are here because either there was no variables line
                # or the line did not start with '!' or the last line
                # read could not be regex'ed into a variable = value
                if aline.find('!Data is not renormalized') == 0.:
                    GENERAL_MIXED = True
             
            # MWS ?   
            tlist = re.findall('! TOUCHSTONE file generated by CST MICROWAVE STUDIO',
                                   aline)
            if len(tlist) > 0:
                self.ts_source = 'CST/MWS'
                # print('got',self.ts_source)
                
            if self.ts_source == 'CST/MWS':
                tlist = re.findall(regex_params, aline)
                # print('  ', aline)
                # print('  ', tlist)
                if len(tlist) > 0:
                    for p in re.findall(regex_var+'='+regex_float+';?',tlist[0]):
                        self.ts_variables[p[0]] = [numbr(p[1]),'mm?']
            
            # return the last line read
            return counter, aline
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
        # initialize source and variables
        self.ts_source = 'Unknown'
        self.ts_variables = {}
                
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
        # Reset some additional data :
        self.markers   = np.array([])
        self.part      = None
        self.numbering = None
        self.shape     = None
        # shapeports     = None
            
        # start processing comments at the start of the str/file                
        while True:
            counter, aline = checkcomment() # ckeckcomment reads the next line
            if aline and aline[0] != '!':   # note blank lines will be treated as
                break                       # empty comments
            
        # process the format statement which should come right after the comments
        if aline[0] != "#":                ## first line should be the options line
            raise TouchStoneError('Missing touchstone options line')
        self._get_tsf_fmt_(aline[1:])      ## don't take the "#" character

        ## OK now we should be able to read the file and look for the right frequency
        self.nports = 0 ## we don't know yet
        cfs = []
        self.freqs = []
        
        counter, aline = next(iterator, (0,'<EOT>')) 
        lastline = ''
        while counter:
            
            # remove comments
            parts = aline.split('!', 1)
            if len(parts) > 1:
                aline, rest = tuple(parts)

            else:
                aline, rest = parts[0], ''
            
            # check for gamma or port impedance lines
            if rest[:16] == ' Gamma         !' or (
                lastline == 'gamma' and rest[:15] !=' Port Impedance'):
                if lastline != 'gamma':
                    if len(Gmsk):
                        Gms.append(Gmsk)
                    Gmsk = []
                tlst = [float(x) for x in rest[16:].split()]
                for k in range(0,len(tlst),2):
                    Gmsk.append(tlst[k]+1j*tlst[k+1])
                lastline='gamma'
            
            elif rest[:15] == ' Port Impedance' or (
                lastline == 'impedance' and not aline):
                if lastline != 'impedance':
                    if Zcsk:
                        Zcs.append(Zcsk)
                    Zcsk = []
                tlst = [float(x) for x in rest[15:].split()]
                for k in range(0,len(tlst),2):
                    Zcsk.append(tlst[k]+1j*tlst[k+1])
                lastline = 'impedance'
            
            # skip empty lines
            if aline:
                                    
                tlst = [float(x) for x in aline.split()]    ## split into tokens -> floats 
                    
                R = len(tlst) % 2
                if R:                           ## a line with an odd number of elements is a new frequency line 
                    self.freqs.append(tlst[0])  ## collect the frequency
                    
                cfs += tlst[R:]                 ##  collect the data
                lastline = 'data'
                           
            counter, aline = next(iterator, (0,'<EOT>'))
            
        # append the last Gms and Zcs
        if len(Gmsk):
            Gms.append(Gmsk)
        if len(Zcsk):
            Zcs.append(Zcsk)
        
        self.freqs = np.array(self.freqs)

        if self.format == "MA":
            cfss = [cfs[j] * np.exp(1j*np.pi/180*cfs[j+1]) 
                    for j in range(0,len(cfs),2)]
            
        elif self.format == "DB":
            cfss = [10**(cfs[j]/20.) * np.exp(1j*np.pi/180*cfs[j+1])
                    for j in range(0,len(cfs),2)]
            
        else: ## can only be "RI"
            cfss = [cfs[j]+1j*cfs[j+1] for j in range(0,len(cfs),2)]
                            
        ## now we need to find the number of ports
        N2 = len(cfss) // len(self.freqs)
        self.nports = int(round(np.sqrt(N2)))
        if (self.nports**2 != N2):          ## the number of coefficients should be a perfect square
            raise TouchStoneError(
                'Number of coefficients (%d) is not a perfect square' % len(cfs)
            )

        ## and let's put all the data into shape
        self.datas = np.array(cfss).reshape((len(self.freqs), self.nports, self.nports))
        
        ## if this was a generalized mixed port impedance matrix we need some
        ## further processing 
        
        # print('GENERAL_MIXED %r, #ZCs %d' % (GENERAL_MIXED, len(Zcs)))
        
        if GENERAL_MIXED or len(Zcs) > 0:
            ## check if the port impedance matrix has the right shape
            if np.array(Zcs).shape != (len(self.freqs), self.nports):
                
                raise TouchStoneError(
                    'the data is not normalized but there is a mismatch in the '
                    'size of the port data %r does not match the frequency matrix '
                    'data %r' % (np.array(Zcs).shape, self.datas.shape)
                )
                
            self.Zcs = np.array(Zcs)
            self.Gms = np.array(Gms)
            
#             print(Zcs)
#             print(Gms)

            for k, (f, D, Zc, Gm) in enumerate(zip(self.freqs, self.datas, Zcs, Gms)):
#                 print('%4d %10.6f -- ' % (k+1,f), end = '')
#                 for p, (Zcp, Gmp) in enumerate(zip(Zc, Gm)):
#                     print('%d %10.5f%+10.5fj %10.7f%+10.7fj' %
#                       (p+1, Zcp.real, Zcp.imag, Gmp.real, Gmp.imag), 
#                       end ='\n                   ')
#                 print()
                
                # this will only work for data of type S
                if self.mtype != 'S':
                    raise TouchStoneError(
                        'the data is not normalized but the data is not of '
                        'type S but %s' % self.mtype
                    )
                
                # print('check !')
                # printMA(D,pfmt=pfmt)
                self.datas[k] = sc.convert_general(self.zref, D, Zc, 'hfss', 'std')
            
        else:
            ## we assume that all the port impedances are equal for all frequencies,
            ## for the gamma we will assume velocity of light in vaccuum
            
            self.Zcs = np.array([[self.zref+0j for p in range(self.nports)]
                                 for k in range(len(self.freqs))])
            
            gm1 = 2j*np.pi / self.Scaler('Hz') / speed_of_light
            self.Gms = np.array([[f for p in range(self.nports)] 
                                for f in self.freqs]) * gm1
                                
            # print(self.Gms)

        ## if there is !PORTS information the number of names must match
        ## otherwise it is ignored
        
        # print(self.ports, self.nports)
        
        if len(self.ports) != self.nports :
            self.ports = ['port-%3.3d' % (k+1) for k in range(self.nports)]

        ## finally the !SHAPE information if provided must be consistent with the
        ## number of ports
            
        if self.shape:
            if not self.shape[1]:
                self.shape[1] = self.nports // self.shape[0]
                
            if self.nports != self.shape[0] * self.shape[1]:
                print("TouchStone: _raw_read_touchstone_ : "
                      "!SHAPE data not consistent with number of ports")
                self.shape = None
         
        if _debug_ :
            print(self.ts_source)
            for v, val in self.ts_variables.items():
                print(v, ' -> ', val)
                
#===============================================================================
#
# R e a d F i l e
#
 
    def ReadFile(self, filepath):
        """
        read a touchstone file from filepath and sets the TouchStoneClass instance
           nports : the number of ports
           freqs  : the list of frequencies,
           datas  : the list of arrays (one for each frequency),
           mtype  : the type of data ('Z','Y','S','G' or 'H')
           format : the format ('MA','DB','RI')
           fscale : the frequency scaler (1e0,1e3,1e6,1e9)
           zref   : the reference impedance
        """

        with open(filepath,'r') as f:
            self.str2touchstone(f.read())

#===============================================================================
#
# W r i t e F i l e
#

    def WriteFile(self, fpath, comments=None, prependto=None, appendto=None,
                  coeffs_per_line=4, line_ending='\r\n'):
        """
        write a touchstone file
        """
        selfcomments = self.comments[:] # save orignal comments
        
        self.comments = []
        
        if comments:
            self.comments += [c for c in comments.split('\n')]
        
        if prependto:
            self.comments += [c for c in prependto.split('\n')]
            
        self.comments += selfcomments
        
        if appendto:
            self.comments += [c for c in appendto.split('\n')]

        with open(fpath,'w') as f:
            tmp = self.COEFFS_PER_LINE, self.LINE_ENDING
            self.COEFFS_PER_LINE = coeffs_per_line
            self.LINE_ENDING = line_ending
            f.write(str(self))
            self.COEFFS_PER_LINE, self.LINE_ENDING = tmp
        
        if comments:
            self.comments = selfcomments[:] # restore original comments
                
##==============================================================================!=====
##
## Rebuild TouchStones from symmetric parts : short form calls
##

    def CombineWithOdd(self, tsOdd, part=None, shape=None, numbering=None):
        return CombineEvenOdd(self, tsOdd, part, shape, numbering)
    
    def CombineWithEven(self, tsEven, part=None, shape=None, numbering=None):
        return CombineEvenOdd(tsEven, self, part, shape, numbering)

##==============================================================================!=====
##
## Rearrage the ports in the touchstone data structure
##
    def Renumber(self, shape=None,numbering=None,remap=None,insitu=False):
        """
        TouchStone().Renumber(shape,numbering) or TouchStone().Renumber(remap)
        shape = (rows, columns)
        numbering = (1st direction, 2nd direction) : direction = Down, Up, Left, Right
        remap = [map 0 to this, map 1 to this, ... map N to this]
        """
        # assert self.shape and self.numbering, 'TouchStone().Renumber() :
        # original shape and numbering unkown'
        if insitu:
            ts = self
        else:
            ts = self.copy()
            
        if remap is not None:
            for k,M in enumerate(ts.datas):
                M1 = M.copy()
                for kr in range(len(M)):
                    kr1 = remap[kr]
                    for kc in range(len(M)):
                        kc1 = remap[kc]
                        M[kr1,kc1] = M1[kr,kc]
        elif (shape is not None) or (numbering is not None):
            raise TouchStoneError('TouchStone().Renumber() : renumbering by shape '
                                  'and numbering not implemented')
        else:
            raise TouchStoneError('TouchStone().Renumber() : need to specify remap '
                                  'or (shape,numbering)')

        if not insitu:
            return ts
        
##==============================================================================!=====
##
## Resample the frequencies in the touchstone data structure
##
    def Resample(self, newfreqs, funit=None, interpolation=None,
                 order=None, itype='interp1d',  relferror=0.01):
        """
        returns a resampled touchstone data structure
        funit = HZ, KHZ, MHZ or GHZ
        interpolation = DB, MA or RI
        the returned result enhirits the funit and interpolation if supplied
        otherwise these are enherited from the source touchstone data structure
                
        """
        # possibly tsf = ts.TouchStone(tformat=self.FormatStr()) avoids to copy data needlessly
        tsf = TouchStone(tformat=self.FormatStr())
        tsf.nports = self.nports # we need this one !
    
        if interpolation is None:
            interpolation = self.format
        else:
            tsf.format = interpolation

        if funit is not None:
            tsf.funit = funit.upper() 
            tsf.fscale = {'HZ':1e0,'KHZ':1e3,'MHZ':1e6,'GHZ':1e9}[tsf.funit]

        if not hasattr(newfreqs, '__iter__'): # list and numpy.ndarray
            newfreqs = [newfreqs]   
        elif isinstance(newfreqs, np.ndarray) and (len(newfreqs.shape) is 0):
            # note: np.array(n) will also have an __iter__ however one cannot iterate it
            newfreqs.resshape((1,))
        tsf.freqs = np.array(newfreqs)
        
        # specicial case when only one existing frequency is asked for
        if len(newfreqs) == 1:
            relerrs = np.abs(1 - self.freqs * self.fscale / (newfreqs[0] * tsf.fscale))
            minerr = np.min(relerrs)
            if minerr <= relferror:
                tsf.datas = self.datas[np.where(np.abs(relerrs) == minerr)[0]][0:1]
                if len(tsf.datas) > 1:
                    print('TouchStone.Resample tsf.datas:', tsf.datas.shape)
                return tsf
                
        # a special case when there is only one frequency point
        if len(self.freqs) is 1:
            
            if np.abs(1 - newfreqs/self.freqs) > relferror :
                raise TouchStoneError(
                    'Resample: only one frequency point and new frequencies '
                    'differ more than %f%% relatively' % (100*relferror))
            
            tsf.datas = np.array([self.datas[0]]*len(tsf.freqs))
                                    
            return tsf

        # normal case
        freqs = self.freqs * self.fscale / tsf.fscale
        
        # check if the frequecies are contained in the source data (no extrapolation)
        nfqs = sorted(newfreqs)
        ##TODO: we could check if the extrapolation is not outlandish
        
        
        # fill the TouchStone
        tsf.datas = np.zeros((len(tsf.freqs), self.nports, self.nports))

        # find which interpolation order is possible
        if order:
            ikind = min(order,len(freqs)-1)
        else:
            ikind = max(min(len(freqs)-1,3),0)

        # interpolate
        
        if itype =='interp1d':
#             if timing :
#                 print('TouchStone().Resample() : using scipy.interpolate.interp1d')
            if interpolation == 'RI':
                R = interp1d(freqs,self.datas,kind=ikind,axis=0,
                             fill_value='extrapolate')(tsf.freqs)
                
            elif interpolation == 'DB':                
                R = np.exp(interp1d(freqs,np.log(np.abs(self.datas)),
                                    kind=ikind,axis=0,
                             fill_value='extrapolate')(tsf.freqs) + 
                           1j*interp1d(freqs,np.unwrap(np.angle(self.datas),axis=0),
                                        kind=ikind,axis=0,
                             fill_value='extrapolate')(tsf.freqs))
            elif interpolation == 'MA':
                MM = interp1d(freqs, 
                              np.abs(self.datas),
                              kind=ikind,
                              axis=0,
                              fill_value='extrapolate'
                             )(tsf.freqs)
                AA = interp1d(freqs, 
                              np.unwrap(np.angle(self.datas), axis=0),
                              kind=ikind,
                              axis=0,
                              fill_value='extrapolate'
                             )(tsf.freqs)
                             
                R = MM * np.exp(1j * AA)
                
            else:
                raise TouchStoneError('TouchStone().Resample() : supplied '
                                      'interpolation parameter not RI, MA or DB')
                     
        elif itype == 'splrepev':
#             if timing :
#                 print('TouchStone().Resample() : '
#                       'using scipy.interpolate.splrep/splev')
            sdatas = np.array(self.datas)
            R = np.array(tsf.datas)
            if interpolation == 'RI':
                for i in range(self.nports):
                    for j in range(self.nports):
                        rtck = splrep(freqs,np.real(sdatas[:,i,j]),s=0)
                        itck = splrep(freqs,np.imag(sdatas[:,i,j]),s=0)
                        R[:,i,j] = (splev(tsf.freqs,rtck,der=0)+
                                    1j*splev(tsf.freqs,itck,der=0))
            elif interpolation == 'DB':                
                for i in range(self.nports):
                    for j in range(self.nports):
                        dtck = splrep(freqs,np.log(np.abs(sdatas[:,i,j])),s=0)
                        ptck = splrep(freqs,np.unwrap(np.angle(sdatas[:,i,j])),s=0)
                        R[:,i,j] = (np.exp(splev(tsf.freqs,dtck,der=0)+
                                           1j*splev(tsf.freqs,ptck,der=0)))
            elif interpolation == 'MA':
                for i in range(self.nports):
                    for j in range(self.nports):
                        atck = splrep(freqs,np.abs(sdatas[:,i,j]),s=0)
                        ptck = splrep(freqs,np.unwrap(np.angle(sdatas[:,i,j])),s=0)
                        R[:,i,j] = (splev(tsf.freqs,atck,der=0)*
                                    np.exp(1j*splev(tsf.freqs,ptck,der=0)))
            else:
                raise TouchStoneError('TouchStone().Resample() : supplied '
                                      'interpolation parameter not RI, MA or DB')
                     
        else:
            raise TouchStoneError('TouchStone().Resample() : interpolation '
                                  'itype (%s) unknown' % itype)

#         if timing :
#             print('Resample / reassemble matrices         :',
#                    time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
        tsf.datas = R # [M for M in R] # because np.array().tolist() is slightly too agressive (arrays not preserved)!

        return tsf
        
##==============================================================================!=====
##
## Rebuild TouchStones from symmetric parts : main routine
##

def CombineEvenOdd(TS1N_even, TS1N_odd, part=None, shape=None, numbering=None):
    """
    reconstruct a 2Nx2N Touchstone data structure from two NxN Touchstone data
    structures for even odd excitation of a circuit with a port symmetry :

    TS2N, TS1N_even and TS1N_odd are TouchStoneClass objects

    The TS1N_odd is converted to the format of TS1N_even and the result has the
    same format.

    part : one of "Bottom", "Top", "Left", "Right" and defines for which part of the
           symmetric constuct the data is given : this will determine how the
           position of the newly created ports will be considered : e,g for a
           shape= (3,2) and numbering= ("Down","Right") then the first step will be
           to create the matrix:

                         [  TS1N_even + TS1N_odd | TS1N_even - TS1N_odd  ] 
        TS2N_raw = 0.5 * [ ----------------------+---------------------- ]
                         [  TS1N_even - TS1N_odd | TS1N_even + TS1N_odd  ]

        The newly added ports are interpreted as beloning to a new port layout :
                   
        shape (3,2)           0   3                    0   3  | 3+6   0+6
        ("Down","Right") ==>  1   4  part= "Left" ==>  1   4  : 4+6   1+6
          == 'Down'           2   5                    2   5  | 5+6   2+6
                                                 
                                                   "old ports" "added ports"
                                                   
        shape (3,2)           0   3                    3+6   0+6  |  0   3
        ("Down","Right") ==>  1   4  part= "Right" ==> 4+6   1+6  :  1   4
          == 'Down'           2   5                    5+6   2+6  |  3   5
                                                 
                                                       "old ports" "added ports" 

        shape (3,2)           0   3                   0    3
        ("Down","Right") ==>  1   4  part= "Top" ==>  1    4  "old ports"
          == 'Down'           2   5                   2    5         
                                                     --------
                                                     2+6  5+6
                                                     1+6  4+6 "added ports
                                                     0+6  3+6

                                                        2+6  5+6
                                                        1+6  4+6  "added ports"
                                                        0+6  3+6
                                                        --------
        shape (3,2)           0   3                      0    3
        ("Down","Right") ==>  1   4  part= "Bottom" ==>  1    4   "old ports"
          == 'Down'           2   5                      2    5         
 
        Finally the port numbering of TS2N_raw is remapped to follow the port
        numbering scheme of the supplied TS1N_even|odd, in this case :
        
                                                           0   3   6    9
        e.g. for part="Left"|"Right" : shape -> (3,4) ==>  1   4   7   10
                                                           2   5   8   11
                                                           
                                                           0    6
        e.g. for part="Top"|"Bottom" : shape -> (6,2) ==>  1    7
                                                           2    8
                                                           3    9
                                                           4   10
                                                           5   11

        None (default) defaults to a "Left" part.
                                                      
    shape : tuplet, (rows, columns), which defines the geometric patterns of the
            ports, e.g. :
    
                      x    x
        (3,2) ==>     x    x
                      x    x
                      
        None (default) defaults to (N,1) where N is the number of ports.

    numbering : tupple ( "Up" | "Down", "Left" | "Right" ) defines how the ports
                are numbered for the given shape, e.g. for the shape above (3,2) :

                             5   2                         0   3
        ("Up","Left")  ==>   4   1   ("Down","Right") ==>  1   4
                             3   0     == 'Down'           2   5
                             
                          5   4
        ("Left","Up") ==> 3   2
          == "Left"       1   0

        If only one numbering direction is given then the second direction defaults
        to either "Down" (for "Left" or "Right" given) or "Right" for (for "Up" or
        "Down" given).

        None (default) defaults to ("Down","Right").
                             
        The resulting TouchStone instance has the same port numbering scheme.
    """
    
    #===========================================================================
    #  C o m b i n e M 1 N s
    # 
    def CombineM1Ns(M_even, M_odd, remap = None, verbose=False):
        """
        reconstructs a 2Nx2N matrix from two NxN matrices evaluated for symmetric
        excitation (M1N_even) and anti-symmetric excitation (M1N_odd) on an antenna
        with a symmetry plane.
        """

        MD = 0.5 * (M_even + M_odd)
        MT = 0.5 * (M_even - M_odd)
        M = np.vstack([np.hstack([MD,MT]),np.hstack([MT,MD])])

        if remap is None :
            M1 = M
        else:
            M1 = np.array([[0j]*len(M)]*len(M))
            for k1 in range(len(M)):
                for k2 in range(len(M)):
                    M1[k1,k2] = M[remap[k1],remap[k2]]
        if verbose:          
            for k1 in range(len(M)):
                for k2 in range(len(M)):
                    print ('%3d %3d %6.3f%+6.3fj -> %3d %3d %6.3f%+6.3fj' %
                            (remap[k1],remap[k2],np.real(M[remap[k1],remap[k2]]),
                             np.imag(M[remap[k1],remap[k2]]),
                             k1,k2,np.real(M1[k1,k2]),np.imag(M1[k1,k2]))) 
        return M1

    #===========================================================================
    #  N u m b e r i n g
    # 
    def Numbering(shape, numbering = None):
        if numbering is None:
            numbering = ('down','right')

        if type(numbering) is str:
            if numbering.lower() in ['down','up']:
                numbering = (numbering.lower(), 'right')
            elif numbering.lower() in ['right','left']:
                numbering = (numbering.lower(),'down')
            else:
                raise TouchStoneError("CombineM1Ns : numbering is not "
                                      "'up', 'down', 'left' or 'right'")
            
        if ((type(numbering) is tuple) and (len(numbering) is 2)):
            if ((type(numbering[0]) is not str) and (type(numbering[1]) is not str)):
                raise TouchStoneError("CombineM1Ns : numbering not str or "
                                      "tuple of 2 str")

            ## everything should pass here if called correctly

            numbering1st = numbering[0].lower()
            numbering2nd = numbering[1].lower()
            
            if numbering1st in ['left','right']:
                tmap = np.arange(shape[0]*shape[1]).reshape(shape)
                if numbering1st == 'left' :
                    tmap = np.fliplr(tmap)
                    
                if numbering2nd == 'up':
                    tmap = np.flipud(tmap)
                    
                else:
                    if numbering2nd != 'down':
                        raise TouchStoneError("CombineM1Ns : numbering[1] is not "
                                              "'up' or 'down'")
                 
            elif numbering1st in ['up','down']:
                tmap = np.arange(shape[0]*shape[1]).reshape(shape[::-1]).transpose()
                if numbering1st == 'up':
                    tmap = np.flipud(tmap)
                
                if numbering[1].lower() == 'left':
                    tmap = np.fliplr(tmap)
                else:
                    if numbering[1].lower() != 'right':
                        raise TouchStoneError("CombineM1Ns : numbering[1] is not "
                                              "'left' or 'right'")
                 
            else:
                raise TouchStoneError("CombineM1Ns : numbering[0] is not 'up', "
                                      "'down', 'left' or 'right'")

        else:
            raise TouchStoneError("CombineM1Ns : numbering not str "
                                  "or tuple of 2 str")

        return tmap
    
    #===========================================================================
    ## check that we have part, shape and numbering information either from the
    ## calling sequence or because it was already in the TouchStone instance's
    ## definition

    # code below needs rechecking : why is TS1N_Odd not checked ?
    if not part:
        if not TS1N_even.part:
            print("CombineM1Ns : no part information given : Left part assumed")
            part = ('left',)
        else:
            part = TS1N_even.part
    else:
        if TS1N_even.part:
            print("CombineM1Ns : part information in Even matrices overruled "
                  "%s -> %s" % (str(part), str(TS1N_even.part)))
    if type(part) is str : part = part,
    
    if not shape:
        if not TS1N_even.shape:
            shape = (TS1N_even.nports,1)
            print("CombineM1Ns : no shape information given : "
                  "(%d,%d) assumed" % shape)
        else:
            shape = TS1N_even.shape
    else:
        if TS1N_even.shape and TS1N_even.shape != shape:
            print("CombineM1Ns : shape information in Even matrices overruled "
                  "(%d, %d) -> (%d, %d)" % (TS1N_even.shape + shape))

    if not numbering:
        if not TS1N_even.numbering:
            numbering = ('down','right')
            print("CombineM1Ns : no numbering information given : "
                  "(%s,%s) assumed" % numbering)
        else:
            numbering = TS1N_even.numbering
    else:
        if TS1N_even.numbering and TS1N_even.numbering != numbering:
            print(
             "CombineM1Ns : numbering information in Even matrices overruled "
             "(%s, %s) -> (%s, %s)" % (TS1N_even.numbering + numbering))
        
            
    ## check that the two input Touchstones are compatible

    if TS1N_even.FormatStr() != TS1N_odd.FormatStr():
#       raise TouchStoneError("CombineM1Ns : Format strings don't match : "
#                              "(%s) (%s)" % (TS1N_even.FormatStr(),
#                                             TS1N_odd.FormatStr())))
        print("CombineM1Ns : Format strings don't match : "
              "(%s) (%s)" % (TS1N_even.FormatStr(),TS1N_odd.FormatStr()))
        print("converting even excitation to format of odd excitation")
        TS1N_even.convertto(TS1N_odd.FormatStr(),inplace=True)

    if TS1N_even.nports != TS1N_odd.nports :
        print(type(TS1N_even.nports), type(TS1N_odd.nports),
              TS1N_even.nports is TS1N_odd.nports)
        raise TouchStoneError("CombineM1Ns : Number of ports don't match : "
                              "(%d) (%d)" % (TS1N_even.nports,TS1N_odd.nports))
    
    f_even = TS1N_even.Freqs()
    f_odd = TS1N_odd.Freqs()
    
    if len(f_even) != len(f_odd):
        raise TouchStoneError(
                        "CombineM1Ns : Frequency vector lengths don't match : "
                        "%d <> %d" % (len(f_even),len(f_odd)) )
    
    for f1ek, f1ok in zip(f_even,f_odd):
        if abs(f1ek - f1ok) >= 1. : # if the frequencies differ by 1Hz or more
            raise TouchStoneError("Frequency values don't match")

    # we are here : both structures have the same number of ports and have the
    # same frequencies so they appear to be compatible. Now we generate the
    # port mapping data :

    map1N = np.arange(TS1N_even.nports)

    if shape is None:
        shape = (TS1N_even.nports, 1)
        
    if type(shape) is int :
        ## the client was lazy and just provided the number of rows
        if TS1N_even.nports % shape is not 0:
            raise TouchStoneError(
                    'CombineM1Ns : shape not compatible with number of ports')
        shape = (shape, TS1N_even.nports / shape)
        
    elif ((type(shape) is tuple) and (len(shape) is 2)):
        if shape[0]*shape[1] != TS1N_even.nports:
            raise TouchStoneError(
                    'CombineM1Ns : shape not compatible with number of ports')

    else:
        raise TouchStoneError('CombineM1Ns : shape not int or tuple of length 2')

    map1N = Numbering(shape, numbering)
    
    # tpart = part[0].lower()
    if part[0].lower() == 'right':
        map2Nraw = np.hstack([np.fliplr(map1N+TS1N_even.nports),map1N])

    elif part[0].lower() == 'left':
        map2Nraw = np.hstack([map1N,np.fliplr(map1N+TS1N_even.nports)])
        
    elif part[0].lower() == 'top':
        map2Nraw = np.vstack([map1N,np.flipud(map1N+TS1N_even.nports)])
        
    elif part[0].lower() == 'bottom':
        map2Nraw = np.vstack([np.flipud(map1N+TS1N_even.nports),map1N])

    else:
        raise TouchStoneError("CombineM1Ns : part is not 'left', 'right', "
                              "'top' or 'bottom'")

    map2N = Numbering(map2Nraw.shape, numbering)

    remap = [None]*map2N.size
    for kr in range(map2N.shape[0]):
        for kc in range(map2N.shape[1]):
            remap[map2N[kr,kc]] = map2Nraw[kr,kc]

    ## now we need to create a new TouchStone object with the same format as 

    ts2N = TouchStone(tformat=TS1N_even.FormatStr())
    ts2N.shape     = map2Nraw.shape
    ts2N.numbering = numbering
    ts2N.markers   = TS1N_even.markers
    ts2N.nports    = map2Nraw.size
    if len(part) > 1:
        ts2N.part = part[1:]
    else:
        ts2N.part = None
    
    for k, fk in enumerate(f_even):
        ## Note stil a potential error if the format strings do not match
        ts2N.Add_Data(CombineM1Ns(TS1N_even.datas[k],TS1N_odd.datas[k],remap),
                      fk,None)

    return ts2N

#===============================================================================
#
# self testing code which is not run when TouchStoneClass is imported at the
# python prompt.
#

if __name__ == '__main__':
    
    import matplotlib.pyplot as pl
    from Utilities.findpath import findpath
    from Utilities.printMatrices import printMA
    
    print(os.path.abspath(os.path.curdir))
    TESTDATA = os.path.join('..','test data')
    
    tests = [220]         
    

    #---------------------------------------------------------------------------
    if 200 in tests:
        print('-' * 128)
        print('[200] Test reading a non-normalized touchstone file generated by HFSS\n')
        tsf = TouchStone(tformat = 'MHz S RI R 20')
        tsf.str2touchstone("""
        ! Touchstone file from project ConicalCoaxNoVars
        ! Exported from HFSS 2018.1.0
        !Data is not renormalized
        # GHZ S MA
        ! Modal data exported
        ! Port[1] = 1:1
        ! Port[2] = 2:1
        0.04             0.283292964088899 170.440143560084 0.95901286211023 -9.36825610582418 0.95901286211023 -9.36825610582417 0.283290805863086 -9.16941981923819 
        ! Gamma         !               9.01085163456468E-05 0.838428205428084 0.000119729472905922 0.838457783204718 
        ! Port Impedance54.8620993192298 -0.00570373395584595 30.5586547248969 -0.00428419247089095 
        
        0.0475           0.282786857369771 168.645947051117 0.959160373204135 -11.1253041843475 0.959160373204136 -11.1253041843475 0.282784507510799 -10.8898695400059 
        ! Gamma         !               9.82013937375871E-05 0.99562467504794 0.000130476204865541 0.995656906598423 
        ! Port Impedance54.8616289297255 -0.00523383947695707 30.5583016315078 -0.00393127482197127 
        
        0.055            0.282194036531023 166.851334209927 0.959333245835787 -12.8827022974075 0.959333245835787 -12.8827022974075 0.282191511113127 -12.610475376352 
        ! Gamma         !               0.00010567683426069 1.15282052720932 0.00014040301882958 1.15285521007033 
        ! Port Impedance54.8612582697252 -0.00486353944587519 30.5580234277451 -0.00365319890758358 
        """)
        print(tsf)
    
    #---------------------------------------------------------------------------
    if 210 in tests:
        print('-' * 128)
        print('[210] Test reading a non-normalized touchstone file generated by HFSS\n')
        tsf = TouchStone(filepath='/home/frederic/Documents/ITER/2018 ITER/' +
                                  'ITER_D_WWS896/02 Models/02 HFSS/' +
                                  'FDe_20190814_RF_losses_benchmarks/' +
                                  'ConicalCoaxNoVars_HFSSDesign1.s2p')
        print(tsf)
        print(tsf.Zcs)
        
        ftsf = 55.0 * tsf.Scaler('MHz')
        Zmixed = interp1d(tsf.freqs, tsf.Zcs, axis=0)(ftsf)
        print(Zmixed)
        S = tsf.Get_Scatter(ftsf).convert_to_mixed(Zmixed)
        printMA(S, '%20.16f%+20.14fdeg')
        
    #---------------------------------------------------------------------------
    if 220 in tests:
        # test dos line endings
        TSF = TouchStone(filepath=findpath('test_HFSSDesign1.s2p',[TESTDATA]))
        print(str(TSF).replace('\r','\\r').replace('\n','\\n\n'))
        fpath = 'dos_line_ending.s%dp' % TSF.nports
        TSF.WriteFile(os.path.join(TESTDATA,fpath))
        TSF2=TouchStone(filepath=findpath(fpath,[TESTDATA]))
        print(str(TSF2).replace('\r','\\r').replace('\n','\\n\n'))
        
    #---------------------------------------------------------------------------
    if 100 in tests:
        print('-' * 128)
        print('[100] Test reading variables from touchstone file generated by HFSS\n')
        tsf = TouchStone(tformat = 'MHz S RI R 20')
        tsf.str2touchstone("""
        ! Touchstone file from project ConicalCoax
        ! Exported from HFSS 2018.1.0
        ! Variables:
        !  L = 500mm
        !  Ri0 = 30mm
        !  RiL = 50mm
        !  Ro = 100mm
        !  d = 10mm
        # GHZ S MA R 50.000000
        ! Modal data exported
        ! Port[1] = 1:1
        ! Port[2] = 2:1
        0.001            0.0010455 79.198 0.9999908 -0.611 0.9999908 -0.612 0.0010193 98.910 
        ! Gamma         !               1.565E-05 0.0209746 1.806E-05 0.0209765
        ! Port Impedance50               0                50               0                
        0.002            0.0020453 84.239 0.9999848 -1.223 0.9999848 -1.223 0.0020250 92.714  
        ! Gamma         !               2.114E-05 0.0419398 2.431E-05 0.0419430
        ! Port Impedance50               0                50               0                
        """)
        print(tsf)
        print(tsf.ts_variables)
    
    #---------------------------------------------------------------------------
    if 0 in tests:
        TSdata1 = TouchStone(TSF, 'S MHZ R 30 RI')
        
        TSdata2 = TouchStone(TSF, 'S MHZ R 50 RI')
        
        print(TSdata1.Datas(freqs=50, mtype='S', elfmt='RI')[0,0],
              ' ==> ',TSdata1.Datas(freqs=50, mtype='Z', elfmt='RI')[0,0])
        
        print(TSdata2.Datas(freqs=50, mtype='S', elfmt='RI')[0,0],
              ' ==> ',TSdata2.Datas(freqs=50, mtype='Z', elfmt='RI')[0,0])

        print(TSdata2.Datas(freqs=50, mtype='S', elfmt='RI')[0,0],
              ' == ',TSdata1.Datas(freqs=50, mtype='S', elfmt='RI', zref=50)[0,0])
        
        print(TSdata1.Datas(freqs=50, mtype='S', elfmt='RI')[0,0],
              ' == ',TSdata2.Datas(freqs=50, mtype='S', elfmt='RI', zref=30)[0,0])
        
    if 1 in tests:
        fMHzs = np.arange(30,60,5)
        fname = os.path.join('/','home','frederic','Documents',
                             'Programming','Python','Scattertoolbox',
                             'matrices','bot-vtl.s3p')
        tsdata = TouchStone(fname)
        pl.figure()
        pl.plot(fMHzs,np.abs(tsdata.Datas(fMHzs,
                                          unit='MHz',
                                          mtype='S',
                                          elems=0,
                                          elfmt='RI')))
        pl.xlabel('frequency [MHz]')
        pl.ylabel('abs(S11)')
        pl.title(os.path.basename(fname))

    if 2 in tests:
        fname = os.path.join('/','home','frederic','Documents',
                             'Programming','Python','Scattertoolbox',
                             'matrices','bot-vtl.s3p')
        tsdata = TouchStone(fname,'Z RI R 30')
        p = tsdata.Datas(mtype='Z',elems=[(0,0),(0,1)],elfmt = 'RI')
        pl.figure()
        pl.plot(tsdata.Freqs(unit='kHz'),np.abs(p[0]),'r')
        pl.plot(tsdata.Freqs(freqs=[10,20,30],unit='MHz'),np.abs(p[1]),'b')

        pl.xlabel('frequency (red) [kHz], (blue) [MHz]')
        pl.ylabel('(red) abs(Z11), (blue) abs(Z12)')
        pl.title(os.path.basename(fname))
       
    if 3 in tests:
        fname = os.path.join('/','media','OS',
                             'Documents and Settings',
                             'frederic',
                             'Documents',
                             'insp6k documents',
                             'Desktop',
                             'RDL56_match_ampl_230409.s4p')
        tsdata = TouchStone(fname)
        fMHzs = tsdata.Freqs(unit='MHz')
        aptlfwd = np.array(tsdata.Datas(elfmt='RI',elems=(0,0)))
        vc5 = np.array(tsdata.Datas(elfmt='RI',elems=(1,1)))
        vc6 = np.array(tsdata.Datas(elfmt='RI',elems=(2,2)))
        
        pl.figure()
        pl.plot(fMHzs,np.abs(vc5/aptlfwd),'r')
        pl.plot(fMHzs,np.abs(vc6/aptlfwd),'b')
        
        pl.xlabel('frequency [MHz]')
        pl.ylabel('(red) abs(S22/S11), (blue) abs(S33/S11)')
        pl.title(os.path.basename(fname))

    if 4 in tests:
        dirpath = os.path.join('/','home','frederic','Documents',
                               'ITER','2010 ITER','002 Design Documents',
                               'RF Design','Z24x24','CY4_RightLower_MVr_281010')

        tsOdd  = TouchStone(os.path.join(dirpath,'lrdiel_xye_xze_271010.s6p'))
        tsEven = TouchStone(os.path.join(dirpath,'lrdiel_xye_xzm_271010.s6p'))
        tsBottomOdd = tsEven.CombineWithOdd(tsOdd)
        
        tsOdd  = TouchStone(os.path.join(dirpath,'lrdiel_xym_xze_271010.s6p'))
        tsEven = TouchStone(os.path.join(dirpath,'lrdiel_xym_xzm_271010.s6p'))
        tsBottomEven = tsEven.CombineWithOdd(tsOdd)
        
        tsTotal = tsBottomEven.CombineWithOdd(tsBottomOdd)

        Z = tsTotal.Datas(freqs=47.5,unit='MHz',mtype='Z',elfmt='RI')
        SZ = sc.Scatter(47.5,
                        Zbase=15.0,
                        Portnames=['s%2.2d'%(k+1) for k in range(24)],
                        Z=Z)
        dpol = [ -67.94960, -41.59262, -15.23565, -15.23565, -41.59262, -67.94960]
        dtor = [ -18.12977,   0.00000,   0.00000, -18.12977 ]
        for k1 in range(24):
            SZ.trlAZV('s%2.2d' % (k1+1),
                      (dpol[k1 % 6] + dtor[k1 / 6])/1000.+0.00335*0,
                      0.,15.,1.)
        Z = SZ.Zbase * np.dot(np.linalg.inv(np.eye(len(SZ.S))- SZ.S),
                              (np.eye(len(SZ.S)) + SZ.S))
        for k1 in range(6):
            print(('%6.3f%+6.3fj '*4) %
                  reduce(lambda x, y: x + (np.real(y),np.imag(y)),
                         [np.diag(Z)[k] for k in range(k1,24,6)],
                         ()))
        print()
        tsMVr = TouchStone(os.path.join(dirpath,'SZ24x24_lrdiel_281010.s24p'))
        ZMVr = tsMVr.Datas(freqs=47.5,unit='MHz',mtype='Z',elfmt='RI')
        for k1 in range(6):
            print(('%6.3f%+6.3fj '*4)
                  % reduce(lambda x, y: x + (np.real(y),np.imag(y)),
                           [np.diag(ZMVr).tolist()[k] for k in range(k1,24,6)],
                           ()))
        tsMVr.WriteFile(os.path.join(dirpath,'SZ24x24_lrdiel_281010_FDe.s24p'))
        tsMVr.convertto('MHz Z RI R 15').WriteFile(
            os.path.join(dirpath,'SZ24x24_lrdiel_281010_Converted_FDe.s24p'))
        tsTotal.convertto('MHz Z RI R 15').WriteFile(
            os.path.join(dirpath,'SZ24x24_lrdiel_xxxxxx_Converted_FDe.s24p'))

    if 5 in tests:
        tsf = TouchStone(tformat='MHZ RI S R 50.')
        # build a small 2 port circuit
        fMHzs = np.linspace(10.,120.,24)
        for fMHz in fMHzs:
            SZ = sc.Scatter(fMHz,Zbase=tsf.zref)
            SZ.trlAZV(['01','SS1'],2.0)
            SZ.trlAZV(['SS','SSsc'],1.0)
            SZ.termport('SSsc',-1.)
            SZ.joinports('SS','SS1','SS2')
            SZ.trlAZV(['SS2','02'],2.5)
            for k,port in enumerate(['01','02']):
                SZ.swapports(k,port)
            # tsf.Add_freq(fMHz,SZ.S) # internal representation is always RI
            tsf.Add_Data(SZ.S,fMHz,None) # internal representation is always RI

        newfreqs = np.linspace(tsf.freqs[0],tsf.freqs[-1],300)
        z11a = tsf.Datas(freqs=newfreqs,
                         unit=tsf.funit,
                         mtype='S',
                         kind='MA',
                         order=None,
                         elems = (0,0))
        z11b = tsf.Datas(freqs=newfreqs,
                         unit=tsf.funit,
                         mtype='S',
                         kind=None,
                         order=1,
                         elems = (0,0))

        pl.figure()
        th = np.linspace(0.,2*np.pi,360)
        pl.plot(np.cos(th),np.sin(th),'b-')
        for r in np.linspace(0.1,0.9,9):
            pl.plot(r*np.cos(th),r*np.sin(th),'b:')
        pl.plot(np.real(z11a),np.imag(z11a),'ro')
        pl.plot(np.real(z11b),np.imag(z11b),'g.')

        pl.figure()
        phi11a = np.unwrap(np.angle(z11a))
        phi11b = np.unwrap(np.angle(z11b))

        pl.plot(newfreqs,phi11a,'ro')
        pl.plot(newfreqs,phi11b,'g.')
        

        tsf.WriteFile('test.s2p')
        tsf.convertto('MHZ Z MA').WriteFile('test2.s2p')

        pl.show()
        
    if 9 in tests:
        dirpath = os.path.join('/','home','frederic','Documents',
                               'ITER','2010 ITER','002 Design Documents',
                               'RF Design','MWS Models','MyTrident')
        print(os.listdir(dirpath))
        TSdata1 = TouchStone(os.path.join(dirpath,'MyTrident OD 285mm.s4p')
                            ).Resample([33.3333],
                                       funit='MHz',
                                       interpolation='RI',
                                       itype='interp1d',
                                       timing=True)
        print(TSdata1.datas[0])
        TSdata2 = TouchStone(os.path.join(dirpath,'MyTrident OD 285mm.s4p')
                            ).Resample([33.3333],
                                       funit='MHz',
                                       interpolation='RI',
                                       itype='splrepev',
                                       timing=True)
        print(TSdata2.datas[0])
        
        
    if 10 in tests:
        dirpath = os.path.join('/','home','frederic','Documents',
                               'ITER','2010 ITER','002 Design Documents',
                               'RF Design','MWS Models','MWS tests')
        print(os.listdir(os.path.join(dirpath,'2port-test','Result')))
        TSdata1 = TouchStone(os.path.join(dirpath,'2port-test'))
        
    if 11 in tests:
        TSdata1 = TouchStone(tformat='MHZ RI Z R 20')
        S = np.array([[11,12,13],[21,22,23],[31,32,33]])
        TSdata1.Add_Data(S,1.)
        TSdata2 = TSdata1.Renumber(remap=[2,1,0],insitu=False)
        print(TSdata1.datas[0])
        print(TSdata2.datas[0])
        
        
        
