__updated__ = '2021-05-03 14:11:21'


def getlines(src):
    """
    Given a source return an iterator function returning the next line of text in the
    source.

    The source can be a multiline string, a string path to a file or a file descriptor
    such as returned when opening a file.
    """
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
    
    
