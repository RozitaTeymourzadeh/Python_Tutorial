#!/usr/env python

import MySQLdb

class Database:
	host = "localhost" # Logo
	user = "testuser" # username
	passwd = "testpass" # Password
	db = "test" # Database name
	
	def __init__(self):
		self.connection = MySQLdb.connect(host = self.host, user = self.user, passwd = self.passwd, db = self.db)
	def query(self,q):
		cursor = self.connection.cursor(MySQLdb.cursors.DistCursor)
		cursor.execute(q)
		return cursor.fetchall()

	def __del__(self):
		self.connection.close()

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
	WHERE age = 21
	"""
	
	people = db.query(q)

	for person in people:
		print "Found: %s"% person['name']
	