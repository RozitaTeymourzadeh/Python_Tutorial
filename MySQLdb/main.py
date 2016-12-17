#!/usr/env python

from Databases.mysqldb import Database

""" Test MySQLdb"""

if __name__ == "__main__":
	db = Database()
	q = "DELET FROM testTable"
	db.query(q)

	q = """
	INSERT INTO testTable
	('name','age')
	VALUES
	('Anita',36),('Rozita',35),('Pantea',22)
	"""
	
	db.query(q)

	q = """
	SELECT * from testTable
	WHERE age = 35
	"""
	
	people = db.query(q)

	for person in people:
		print "Found: %s"% person['name']
	