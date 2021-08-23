# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:20:12 2016
    
@author: rozita.teymourzadeh
"""

config = {'base_url':'http://www.rozitateymourzadeh.com','pages':['home','about','contact']}
   
def base_url():
	return config['base_url']
	
def get_page_url(name):
	page_name = ''
	if name in config['pages']:
		page_name = name
		
	return "%s/%s" % (base_url(),page_name)
	
if __name__ == "__main__":
	print(get_page_url('home'))
	print(get_page_url('services'))
	print(get_page_url('contact'))
       