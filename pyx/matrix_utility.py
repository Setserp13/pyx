from pyx.array_utility import call
import math
import pyx.array_utility as au

def foreach(matrix, func):
	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
			call(func, matrix[i], i, j)

def map_columns(list, func): return [func(column(list, i), i) for i in range(max_len(list))]

def column(list, index): return [list[i][index] for i in range(len(list)) if index < len(list[i])]

def map(array, func, start, stop, step): return [call(func, array[i], i) for i in indices(array, start, stop, step)]

def funcvals(row_count, column_count, func):
	result = []
	for i in range(row_count):
		row = []
		for j in range(column_count):
			row.append(func(i, j))
		result.append(row)
	return result

def full(row_count, column_count, default_value):
	return funcvals(row_count, column_count, lambda i, j: default_value)




def max_len(list):
	return au.reduce([len(row) for row in list], lambda total, currentValue: max(total, currentValue))

def transpose(arr):
	"""if len(arr) == 0: return arr
	return funcvals(len(arr), len(arr[0]), lambda i, j: arr[j][i])"""
	width = max_len(arr)
	for i in range(len(arr)): arr[i] = au.resize(arr[i], width)
	result = full(width, len(arr), None)
	for i in range(len(result)):
		for j in range(len(result[i])):
			result[i][j] = arr[j][i]
	return result



	

#matrix like
class MatrixLike:
	def __init__(self, matrix, length, getValue, setValue):
		self.matrix = matrix
		self._length = length
		self._getValue = getValue
		self._setValue = setValue

	def length(): return self._length(matrix)

	def getValue(i, j): return self._getValue(matrix, i, j)

	def setValue(i, j, value): self._setValue(matrix, i, j, value)

	def inRange(i, j):
		return i > -1 and i < self.length()[0] and j > -1 and j < self.length()[1]

	def getValueOrDefault(i, j, defaultValue=None):
		if self.inRange(i, j): return self.getValue(i, j)
		return defaultValue

	def getValues(self, indices):
		result = []
		for i in indices:
			result.append(self.getValue(*indices))
		return result

	def getRow(self, index): return au.funcvals(self.length()[1], lambda i: self.getValue(index, i))
	
	def getColumn(self, index): return au.funcvals(self.length()[0], lambda i: self.getValue(i, index))	

	def forEach(self, action):
		for i in range(self.length()[0]):
			for i in range(self.length()[1]):
				call(action, self.getValue(i, j), i, j)	

	def findIndex(self, match):
		for i in range(self.length()[0]):
			for i in range(self.length()[1]):
				if(call(match, self.getValue(i, j), i, j)):
					return [i, j]
		return None

	def findIndices(self, match):
		result = []
		for i in range(self.length()[0]):
			for i in range(self.length()[1]):
				if(call(match, self.getValue(i, j), i, j)):
					result.append([i, j])
		return result

	def indexOf(self, value): return self.findIndex(lambda x: x == value)

	def indicesOf(self, value): return self.findIndices(lambda x: x == value)

	def find(self, match): return self.getValueOrDefault(*self.findIndex(match))

	def findAll(self, match): return self.getValues(self.findIndices(match))

	def map(self, func):
		result = []
		for i in range(self.length()[0]):
			for i in range(self.length()[1]):
				result.append(call(func, self.getValue(i, j), i, j))
		return result

	def getRange(self, startIndex, count):
		return funcvals(count[0], count[1], lambda i, j: self.getValue(i + startIndex[0], j + startIndex[1]))
