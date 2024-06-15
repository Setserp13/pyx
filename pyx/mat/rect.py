import math
from pyx.mat.vector import *

#GENERIC

class Rect:
	def __init__(self, min, size): #min and size are vectors
		self.min = min
		self.size = size
	
	def MinMax(min, max): return Rect(min, max - min)	

	@property
	def max(self): return self.min + self.size

	@max.setter
	def max(self, value): self.size = value - self.min

	@property
	def extents(self): return self.size * 0.5

	@extents.setter
	def extents(self, value): self.size = value * 2

	@property
	def center(self): return self.min + self.extents

	@center.setter
	def center(self, value): self.min = value - self.extents

	def denormalizePoint(self, p):
		return self.min + Vector.scale(self.size, p)
	
	def normalizePoint(self, p):
		return Vector.divide(p - self.min, self.size)
	
	def normalizeVector(self, v):
		return Vector.divide(v, self.size)

	def denormalizeVector(self, v):
		return Vector.scale(self.size, v)
	
	def normalizeRect(self, r):
		return Rect(self.normalizePoint(r.min), self.normalizeVector(r.size))

	def denormalizeRect(self, r):
		return Rect(self.denormalizePoint(r.min), self.denormalizeVector(r.size))

	def setPosition(self, pivot, value): #value is not normalized
		return Rect(self.min + (value - self.denormalizePoint(pivot)), self.size)

	def setAxisPosition(self, axis, pivot, value): #value is not normalized
		#return Rect(self.min.set(axis, self.min[axis] + (value - (self.min[axis] + (self.size[axis] * pivot)))), self.size)
		return Rect(self.min.set(axis, value - (self.size[axis] * pivot)), self.size) #simplified

	def minimumBounding(self, other):
		return Rect.MinMax(Vector.min(self.min, other.min), Vector.max(self.max, other.max))

	def containsPoint(self, p):
		for i in range(len(p)):
			if p[i] < self.min[i] or p[i] > self.max[i]:
				return False
		return True

	def containsRect(self, r):
		return self.containsPoint(r.min) and self.containsPoint(r.max)

	def to_list(self): return list(self.min) + list(self.size)
	def to_tuple(self): return tuple(self.min) + tuple(self.size)

	def clamp(a, b):
		return Rect.MinMax(Vector(*Vector.max(a.min, b.min)), Vector(*Vector.min(a.max, b.max)))

	#def __str__(self): return f'min: {self.min}, size: {self.size}'
	def __repr__(self): return f'Rect(min={self.min}, size={self.size})'

	def axisIntersection(a, b, axis=0): # If the intervals overlap, return the intersection, else, return None
		start = max(a.min[axis], b.min[axis])
		stop = min(a.max[axis], b.max[axis])
		return None if stop < start else (start, stop)
		
	def __add__(self, v): return Rect(self.min + v, self.size)
	
	def __sub__(self, v): return Rect(self.min - v, self.size)




#SPECIALIZED

def Rect2(x, y, width, height): return Rect(Vector(x, y), Vector(width, height))

def absolute_padding(rect, l=0.0, r=0.0, d=0.0, u=0.0):
	return Rect(rect.min + Vector(l, d), rect.size - Vector(l + r, d + u))

def relative_padding(rect, l=0.0, r=0.0, d=0.0, u=0.0):
	return absolute_padding(rect, l * rect.size[0], r * rect.size[0], d * rect.size[1], u * rect.size[1])

def padding(rect, l=0.0, r=0.0, d=0.0, u=0.0, relative=True):
	return relative_padding(rect, l, r, d, u) if relative else absolute_padding(rect, l, r, d, u)

def positionBounds(rect, object_size, pivot=Vector(0.5, 0.5)):
	l = object_size[0] * pivot[0]
	r = object_size[0] - l
	d = object_size[1] * pivot[1]
	u = object_size[1] - d
	return absolute_padding(rect, l, r, d, u)

def bottom_left(rect): return rect.denormalizePoint(Vector(0, 0))
def bottom_right(rect): return rect.denormalizePoint(Vector(0, 1))
def top_left(rect): return rect.denormalizePoint(Vector(1, 0))
def top_right(rect): return rect.denormalizePoint(Vector(1, 1))
def corners(rect): return [bottom_left(rect), top_left(rect), top_right(rect), bottom_right(rect)]
def area(rect): return rect.size[0] * rect.size[1]

"""left = rect.denormalizePoint(Vector(0, .5))
right = rect.denormalizePoint(Vector(1, .5))
bottom = rect.denormalizePoint(Vector(.5, 0))
top = rect.denormalizePoint(Vector(.5, 1))"""

def ChildRect2(parent, r, pivot=Vector(0.5, 0.5)):
	return parent.denormalizeRect(r).setPosition(pivot, parent.denormalizePoint(r.min))



class BBox:
	def circle(cx, cy, r): return Rect2(cx - r, cy - r, r * 2, r * 2)
	def ellipse(cx, cy, rx, ry): return Rect2(cx - rx, cy - ry, rx * 2, ry * 2)







def contains(min, max, value): return min <= value and value <= max

def contains2(min, max, value):
	return min[0] <= value[0] and value[0] <= max[0] and min[1] <= value[1] and value[1] <= max[1]

def contains3(min, max, value):
	return min[0] <= value[0] and value[0] <= max[0] and min[1] <= value[1] and value[1] <= max[1] and min[2] <= value[2] and value[2] <= max[2]

def containsN(min, max, value):
	for i in range(len(min)):
		if not contains(min[i], max[i], value[i]):
			return false
	return true






#SIZING

def set_size_component(size, index, value, preserve_aspect = True):
	if preserve_aspect:
		scale = value / size[index]
		#return tuple(map(lambda x: x * scale, size))
		return tuple(map(lambda x: x * scale, size[:index])) + (value,) + tuple(map(lambda x: x * scale, size[index+1:]))
	else:
		return size[:index] + (value,) + size[index+1:]

def aspect(size): return size[1] / size[0]	#Aspect ratio


def set_aspect(size, value):
	return (size[0], int(size[0] * value)) if aspect(size) > value else (int(size[1] / value), size[1])


def fit(viewport, viewbox, scale_method='meet', align=(0.5, 0.5)): #scale_method in ['meet', 'slice']
	for i in range(len(viewport)):
		size = tuple(map(lambda x: int(x), set_size_component(viewbox, i, viewport[i])))
		#print([viewport, viewbox, size])
		valid = True
		for j in range(len(viewport)):
			if (viewport[j] < size[j] if scale_method == 'meet' else viewport[j] > size[j]):
				valid = False			
		if valid: return size


def intersect(rect1, rect2): #where each rectangle is represented as a tuple (x, y, width, height)
	x1 = max(rect1[0], rect2[0])
	y1 = max(rect1[1], rect2[1])
	x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
	y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
	# Check if there's an intersection
	if x1 < x2 and y1 < y2:
		return (x1, y1, x2 - x1, y2 - y1)
	else:
		return None
