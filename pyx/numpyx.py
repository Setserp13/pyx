import numpy as np

def aabb(*points):
	return rect.min_max(np.minimum.reduce(points), np.maximum.reduce(points))

def clamp(point, min, max): return np.minimum(np.maximum(point, min), max)

def lerp(a, b, t): return a * (1 - t) + b * t

class rect:
	def __init__(self, min, size):
		self.min = np.array(min)
		self.size = np.array(size)

	@property
	def center(self): return self.min + self.extents

	@property
	def extents(self): return self.size * 0.5

	@property
	def max(self): return self.min + self.size

	def normalize_point(self, value): return (value - self.min) / self.size

	def denormalize_point(self, value): return value * self.size + self.min

	def normalize_vector(self, value): return value / self.size

	def denormalize_vector(self, value): return value * self.size

	def normalize_rect(self, value):
		return rect(self.normalize_point(value.min), self.normalize_vector(value.size))

	def denormalize_rect(self, value):
		return rect(self.denormalize_point(value.min), self.denormalize_vector(value.size))

	def clamp(self, point): return clamp(point, self.min, self.max)

	#STATIC
	def center_size(center, size): return rect(center - size * 0.5, size)

	def min_max(min, max): return rect(min, max - min)

	def aabb(a, b):
		return rect.min_max(np.minimum(a.min, b.min), np.maximum(a.max, b.max))

class grid:
	def __init__(self, cell_size, offset=None, cell_gap=None):
		self.cell_size = np.array(cell_size)
		self.offset = np.zeros(len(cell_size)) if offset is None else np.array(offset)
		self.cell_gap = np.zeros(len(cell_size)) if cell_gap is None else np.array(cell_gap)

	def cell_min(self, index):
		return self.offset + (self.cell_size + self.cell_gap) * index

	def cell_index(self, point):
		return (point - self.offset) // (self.cell_size + self.cell_gap)
	
	def cell(self, index):
		return rect(self.cell_min(index), self.cell_size)

	def lines(self, min_index, max_index, axis=None):
		result = []
		if axis is None:
			for i in range(len(min_index)):
				result += self.lines(min_index, max_index, i)
		else:
			for i in range(min_index[axis], max_index[axis]):
				a = np.array(min_index)
				b = np.array(max_index)
				a[axis] = b[axis] = i
				result.append([self.cell_min(a), self.cell_min(b)])
		return result
