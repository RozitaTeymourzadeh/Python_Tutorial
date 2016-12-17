# To input with the unknown length of input argument   
def add_numbers(*args):
	total = 0
	for a in args:
		total += a
	print(total)

def health_no(age, apple, Cigs):
	answer = (100-age)+(apple*3.5)-(Cigs*2)
	print (answer)


add_numbers(4,5)

data = [27, 5, 0]
health_no(data[0],data[1],data[2])
# or
health_no(*data)