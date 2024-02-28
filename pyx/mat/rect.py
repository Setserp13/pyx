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
		return MinMax(min(self.min, other.min), max(self.max, other.max))

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



#SPECIALIZED

def Rect2(x, y, width, height): return Rect(Vector(x, y), Vector(width, height))

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
