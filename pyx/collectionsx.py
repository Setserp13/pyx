
class Dict:
	def items(dct, keys): return { x: dct[x] for x in keys if x in dct } #also works with list and tuple
	def get(dct, key, dflt_value=None): return dct[key] if key in dct else dflt_value

class List:
	def items(arr, indices): return [arr[x] for x in indices if x in range(len(arr))] #also works with dict
	def get(arr, index, dflt_value=None): return arr[index] if index in range(len(arr)) else dflt_value
	


def left_shift(arr, n=1): return arr[n:] + arr[:n]	#Left shift the array arr by n positions

def right_shift(arr, n=1): return arr[-n:] + arr[:-n]	#Right shift the array arr by n positions
