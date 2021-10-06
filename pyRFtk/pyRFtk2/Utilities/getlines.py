################################################################################
#                                                                              #
# Copyright 2018-2020                                                          #
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

__updated__ = '2021-05-03 14:11:21'

"""
Created on 27 Apr 2020

@author: frederic

Given a source return an iterator function returning the next line of text in the
source.

The source can be a multiline string, a string path to a file or a file descriptor
such as returned when opening a file.
"""

def getlines(src):

    def path_next():
        with open(src,'r') as f:
            for aline in f.readlines():
                yield aline.replace('\n','')
    
#     def str_next_slow(): # this one is much slower than the other one ...
#         s = src
#         while s:
#             try:
#                 aline, s = s.split('\n',1)
#             except:
#                 aline, s = s, ''
#             yield aline.replace('\r','')
            
    def str_next():
        for aline in src.split('\n'):
            yield  aline
            
    def file_next():
        for aline in src.readlines():
            yield aline.replace('\n','')
            
    if type(src) == str:
        if src.find('\n') >= 0:
            return str_next
        else:
            return path_next
    else:
        return file_next
    
if __name__ == '__main__':
    
    astring = """This is a multiline test string
    as mentioned
    it has multiple lines"""
    
    apath = 'nextline.py'

    def fun(src1, src2):     
                
        count = 0
        for aline in getlines(src1)():
            print('-->                                        ',aline)
            for aline2 in getlines(src2)():
                print('--', aline2)
            count += 1
        print(count,' lines')
        
    f = open(apath,'r')
    fun(f, astring)
    f.close()
    
    fun(astring,apath)
    
    
