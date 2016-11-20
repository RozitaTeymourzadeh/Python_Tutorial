output_file = open("Example.txt","w")
output_file
output_file.write("a line of text")
output_file.close()
with open("Example.txt","w") as output_file:
	lines=["line 1\n","line 2\n","line 3"]
	output_file.writelines(lines)
with open("Example.txt","r") as input_file:
	print input_file.readlines()
with open("Example.txt","r") as input_file:
	for line in input_file.readlines():
		print line	
