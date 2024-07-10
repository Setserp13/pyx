
#Left shift the array arr by n positions
def left_shift(arr, n=1):
	return arr[n:] + arr[:n]

#Right shift the array arr by n positions
def right_shift(arr, n=1):
	return arr[-n:] + arr[:-n]
