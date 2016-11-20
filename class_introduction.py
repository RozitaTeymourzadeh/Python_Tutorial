# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:22:00 2016

@author: rozita.teymourzadeh
"""

class Bird:
    def __init__(self,kind,call):
        self.kind = kind
        self.call = call
    def do_call(self):
        print "%s says %s" %(self.kind,self.call)
        
class Parrot(Bird):
    def __init__(self):
        Bird.__init__(self,"Parrot","Kah")
        
class Coukoo(Bird):
    def __init__(self):
        Bird.__init__(self,"Coukoo","Cook")    
    
if __name__ == "__main__":
    Parrot = Parrot()
    Coukoo = Coukoo()
    
    Parrot.do_call()
    Coukoo.do_call()