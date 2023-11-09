################################################################################
#                                                                              #
# Copyright 2018-present                                                       #
#                                                                              #
#                   Laboratory for Plasma Physics                              #
#                   Royal Military Academy                                     #
#                   Brussels, Belgium                                          #
#                                                                              #
#                   ITER Organisation                                          #
#                                                                              #
#                   EUROfusion                                                 #
#                                                                              #
# Author : frederic.durodie@rma.ac.be                                          #
#                          @gmail.com                                          #
#                          @ccfe.ac.uk                                         #
#                          @telenet.be                                         #
#                          .lpprma@tlenet.be                                   #
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
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 18 Dec 2020

@author: frederic
"""
__updated__ = "2023-09-22 15:55:47"

from .config import tLogger, setLogLevel, logit, ident, _newID, logident
from .rfBase import rfBase
from .rfObject import rfObject
# from .circuit import circuit # will be obsolete
from .rfCircuit import rfCircuit
from .rfTRL import rfTRL
from .rfRLC import rfRLC
from .rfGTL import rfGTL
from .rfArcObj import rfArcObj
from .rfCoupler import rfCoupler

from .S_from_Z import S_from_Z
from .S_from_Y import S_from_Y
from .S_from_VI import S_from_VI
from .Z_from_S import Z_from_S
from .Y_from_S import Y_from_S

from .ConvertGeneral import ConvertGeneral
from .ReadTSF import ReadTSF
from .maxfun import maxfun
from .resolveTLparams import TLresolver
from .plotVSWs import plotVSWs, scaleVSW, strVSW
from ._check_3D_shape_ import _check_3D_shape_

from .getlines import getlines
from .printMatrices import strM, printM, printMA, printRI
from .tictoc import tic, toc
from .ReadDictData import ReadDictData
from .findpath import findpath
from .compareSs import compareSs
from .whoami import whoami
from .str_dict import str_dict
