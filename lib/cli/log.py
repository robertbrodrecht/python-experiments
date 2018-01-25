"""A few tools to make it easier to log on the command line."""

import time

class log:
	# Get the current time
	def now():
		"""Get the current second without microtime."""
		
		return int(time.time())
	
	# Get the change in time as HH:MM:SS
	def timeDelta(time_start, time_current = None):
		"""Get the time difference between two times in HH:MM:SS format."""
		
		if time_current == None:
			time_current = log.now()
		time_change = time_current - time_start
		time_change_gm = time.gmtime(time_change)
		time_change_str = time.strftime('%H:%M:%S', time_change_gm)
		return time_change_str
	
	# Print out how long it took something to complete
	def doneIn(start):
		"""Print a time difference between start and now."""
		
		print('Done in %s' % log.timeDelta(start), end='\n\n')

if __name__ == "__main__":
	start = log.now()
	time.sleep(2)
	log.doneIn(start)