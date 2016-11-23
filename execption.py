class A_Exception(Exception):
	message = "Exception A fault accured"
class B_Exception(Exception):
	message = "Exception B fault accured"

if __name__ == "__main__":
	something = "YES"
	
	try:
		if something is None:
			raise Exception("Exception A fault accured")
			
		if something is "YES":
			raise A_Exception()

	except A_Exception, ae:
		print "A exception is:",ae.message
		
	except B_Exception, be:
		print "B exception is:", be.message
		
	except Exception, e:
		print "exception is:", e.message
		
	finally:
		print"We are done!"