import math

def all(pred, iterable): return len(list(filter(pred, iterable))) == len(iterable)

def any(pred, iterable): return len(list(filter(pred, iterable))) > 0

def call(func, *args): return func(*resize(args, func.__code__.co_argcount))

def foreach(array, func): return [call(func, x, i) for i, x in enumerate(array)]

def funcvals(count, func): return list(map(lambda i: func(i), range(count)))

#also works with dict
def get_items(array, indices): return [lambda x: array[x] for x in indices]

def items(dict, keys): return { x: dict[x] for x in keys if x in dict }

def resize(array, new_size, default_value=None):
	result = []
	for i in range(min(len(array), new_size)):
		result.append(array[i])
	for i in range(len(array), new_size):
		result.append(default_value)
	return result

#full(count, default_value) = count * [default_value]
#get_last(array, index) =  array[-index]
#get_range(array, start, count) = array[start:start+count]

def get(array, index, default_value=None):
	return array[index] if in_range(array, index) else default_value

def in_range(array, index): return index > -1 and index < len(array)

def find_index(array, match):
	for i, x in enumerate(array):
		if call(match, x, i):
			return i
	return -1

#find_last_index(array, match) = return find_index(reversed(array), match)

def find_indices(array, match, start=0, stop=None, step=1):
	result = []
	for i, x in enumerate(array):
		if call(match, x, i):
			result.append(i)
	return result

def find_all(array, match): return get_items(array, find_indices(array, match))

def index_of(array, item): return find_index(array, lambda x: x == item)

#def last_index_of(array, item): return find_last_index(array, lambda x: x == item)

def indices_of(array, item): return find_indices(array, lambda x: x == item)

def distinct(list):
	result = []
	for item in list:
		if item not in result:
			result.append(item)
	return result

def sv(arr, separator=',', quote=''): #separated value
	result = ''
	for i in range(len(arr)):
		if i > 0:
			result += separator
		result += quote + str(arr[i]) + quote
	return result

#both a and b are array like with the same size
def operate(a, b, func):
	result = []
	for i in range(len(a)):
		result.append(func(a[i], b[i]))
	return result

def concat(a, b): return operate(a, b, lambda x, y: str(x) + str(y))

def reduce(arr, reducer, initialValue = None):
	if len(arr) == 0:
		return None
	initialIndex = 0
	if initialValue == None:
		initialValue = arr[0]
		initialIndex = 1    
	total = initialValue        
	for i in range(initialIndex, len(arr)):
		total = reducer(total, arr[i])
	return total


def find_key(dict, match):
	for k in dict:
		if match(dict[k]):
			return k
	return None

def key_of(dict, value): return find_key(dict, lambda x: value == x)

def group_of(dict, value): return find_key(dict, lambda x: value in x)
