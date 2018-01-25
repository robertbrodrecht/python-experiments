"""A few tools to make it easier to deal with command line things."""

import re
import sys

class args:
	"""Help with parsing out CLI arguments."""

	def get(flag, value = True):
		"""Parses command line arguments for a particular argument.
		
		If the argument is not present, return False.
		If the argument is present, return True if value is False 
		or if there is no string associated with it. Otherwise,
		return the string value associated with the argument.
		"""
		
		args = []
		
		# Expand combined letter arguments into individual arguments
		for arg in sys.argv:
			if arg[0] == '-' and arg[1] != '-' and len(arg) > 2:
				arg = arg[1:]
				arg = re.sub('(\w)', ' -\\1', arg).strip()
				args.extend(arg.split(' '))
			else:
				args.append(arg)
	
		# Single dash for letters, double dash for words
		if len(flag) > 1:
			flag = '--' + flag
		else:
			flag = '-' + flag
		
		# Check if the flag is in the args
		if flag in args:
			index = args.index(flag)+1
	
			# If the next value is not a flag, return the value
			if value and len(args)-1 >= index and args[index][0] != '-':
				return args[index]
			return True
		
		return False

if __name__ == "__main__":
	print(args.get('v'))