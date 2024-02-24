import math


def all(pred, iterable): return len(list(filter(pred, iterable))) == len(iterable)


def any(pred, iterable): return len(list(filter(pred, iterable))) > 0



def funcvals(count, func):
	result = []
	for i in range(count):
		result.append(func(i))
	return result



def get_range(array, start=0, stop=None, step=1):
	result = []
	for i in indices(array, start, stop, step):
		result.append(array[i])
	return result


def full(count, default_value=None):
	result = []
	for i in range(count):
		result.append(default_value)
	return result

def call(func, *args):
	return func(*resize(args, func.__code__.co_argcount))

def indices(array, start=0, stop=None, step=1):
	if stop == None:
		stop = len(array)
	return range(start, stop, step)

def indices_reverse(array, start=None, stop=-1, step=-1):
	if start == None:
		start = len(array) - 1
	return range(start, stop, step)

def foreach(array, func, start=0, stop=None, step=1):
	for i in indices(array, start, stop, step):
		call(func, array[i], i)

def foreach_reverse(array, func, start=None, stop=-1, step=-1):
	for i in indices_reverse(array, start, stop, step):
		call(func, array[i], i)

def map(array, func, start=0, stop=None, step=1):
	result = []
	for i in indices(array, start, stop, step):
		result.append(call(func, array[i], i))
	return result




def resize(array, new_size, default_value=None):
	result = []
	for i in range(min(len(array), new_size)):
		result.append(array[i])
	for i in range(len(array), new_size):
		result.append(default_value)
	return result

def last(array):
	return array[len(array) - 1]

def get_last(array, index=0):
	return array[(len(array) - 1) - index]

def get(array, index, default_value = None):
	if index > -1 and index < len(array):
		return array[index]
	else:
		return default_value






def find_index(array, match, start=0, stop=None, step=1):
	for i in indices(array, start, stop, step):
		if call(match, array[i], i):
			return i
	return -1

def find_last_index(array, match, start=None, stop=-1, step=-1):
	for i in indices_reverse(array, start, stop, step):
		if call(match, array[i], i):
			return i
	return -1

def find_indices(array, match, start=0, stop=None, step=1):
	result = []
	for i in indices(array, start, stop, step):
		if call(match, array[i], i):
			result.append(i)
	return result

def find_all(array, match, start=0, stop=None, step=1):
	return get_items(array, find_indices(array, match, start, stop, step))


def index_of(array, item, start=0, stop=None, step=1):
	return find_index(array, lambda x: x == item, start, stop, step)

def last_index_of(array, item, start=None, stop=-1, step=-1):
	return find_last_index(array, lambda x: x == item, start, stop, step)

def indices_of(array, item, start=0, stop=None, step=1):
	return find_indices(array, lambda x: x == item, start, stop, step)





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


#also works with dict
def get_items(array, indices): return map(indices, lambda x: array[x])



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