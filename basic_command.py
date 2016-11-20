# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:20:12 2016
    
@author: rozita.teymourzadeh
"""
    
def basic_command():
        message = "Basic python syntax shown here!"
        print message
        
if __name__== "__main__":
        basic_command()
        
        
        type(int(3.5))
        text = 'String Type'
        print text
		
        text1 = text.replace('String','Float')
        print text1
		
        text2 = "hamester %s %s" %("Hello","!")
        print text2
        
        text3 = "Counter " * 20
        print text3
        
        length = len(text3)
        print length
        
        text4 = text2[9]
        print text4
		
        text5 = text2[9:15]
        print text5
		
        text6 = text2[-1]
        print text6 
        
		# List Tuple Dict
        a_list = [1, 2, 'ros', 2.3]
        Index2 = a_list[2]
        print Index2
        
        a_tuple = (2, 3, 'ros') # can not be changed
		
        a_Dict = {}
        a_Dict['name']='Roz'
        b_Dict={}
        b_Dict={'name':'Mike','Age':'21'}
        print  a_Dict['name']
        print b_Dict
        print b_Dict.keys()
        print b_Dict.values()
        
		# if statement
        if 1==1:
            print "It is equal"
        elif 1<3 :
            print "It is smaller than 3"
        else:
            print "Nothing"
			
        # while statement   
        input =''   
        while(input != 'N'):
            input = raw_input('What is your favorite color?')
            if input == 'Blue':
                break
            if input == 'Red':
                print "You Win!"
                
        # for statement       
        print range(10)
        
        for cheese in range (10):
            
            print cheese
            
            words = ['pen','Ball','Eight','5']
            for word in words:
                print word
                
            for index, word in enumerate(words):
                print index, word
        
		# Dictionary 
        person_detail = {}
        person_detail = {'name':'Mike','Age':'21', 'job':'Eng'}
        
        for keys, values in person_detail.items():
            print keys, values
       