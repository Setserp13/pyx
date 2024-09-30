
class Dict
	def get(dct, key, dflt_value=None): return dct[key] if key in dct else dflt_value

class List
	def get(arr, index, dflt_value=None): return arr[index] if index in range(len(arr)) else dflt_value

#Left shift the array arr by n positions
def left_shift(arr, n=1):
	return arr[n:] + arr[:n]

#Right shift the array arr by n positions
def right_shift(arr, n=1):
	return arr[-n:] + arr[:-n]
