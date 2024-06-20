import math

#GENERIC

class Vector(tuple):
	def __new__(self, *components): return tuple.__new__(Vector, components)	

	def __add__(self, other):
		return Vector(*tuple(map(lambda x, y: x + y, self, other)))
	
	def __neg__(self):
		return Vector(*tuple(map(lambda x: -x, self)))

	def __sub__(self, other):
		return Vector(*tuple(map(lambda x, y: x - y, self, other)))
	
	def __mul__(self, scalar):
		return Vector(*tuple(map(lambda x: x * scalar, self)))
	
	def __truediv__(self, scalar):
		return Vector(*tuple(map(lambda x: x / scalar, self)))

	def __rmul__(self, scalar): return self * scalar

	"""def __eq__(self, other):
		if len(self) != len(other):
			return False
		for i in range(len(self)):
			if self[i] != other[i]:
				return False
		return True"""

	def scale(a, b):
		return Vector(*tuple(map(lambda x, y: x * y, a, b)))
	
	def divide(a, b):
		return Vector(*tuple(map(lambda x, y: x / y, a, b)))

	def floordiv(a, b):
		return Vector(*tuple(map(lambda x, y: x // y, a, b)))
		
	def reduce(self, func, initialValue):
		result = initialValue
		for x in self: result = func(result, x)
		return result
	
	#raised norm
	def rsdNorm(self, n): return self.reduce(total, lambda x: total + (x ** n))	

	def norm(self, n): return self.rsdNorm ** (1 / n)

	#squared magnitude
	@property
	def sqrMagnitude(self): return self.rsdNorm(self, 2)

	"""def __getitem__(self, index): return self[index]

	def __setitem__(self, index, value):
		lst = list(self)
		lst[index] = value
		self = Vector(lst)"""

	def set(self, index, value):
		lst = list(self)
		lst[index] = value
		return Vector(*lst)

	@property
	def magnitude(self): return math.sqrt(self.sqrtMagnitude())

	def unclampedLerp(a, b, t): return tuple(map(lambda a, b: a + (b - a) * t, a, b))

	def lerp(a, b, t): return Vector.unclampedLerp(a, b, max(0, min(t, 1)))

	def min(a, b): return Vector(*tuple(map(lambda x, y: min(x, y), a, b)))
	def max(a, b): return Vector(*tuple(map(lambda x, y: max(x, y), a, b)))
	#def clamp(val, min, max): return tuple(map(lambda x, y, z: clamp(x, y, z), val, min, max))


#SPECIALIZED


class Vector2:
	zero = Vector(0, 0)
	one = Vector(1, 1)
	left = Vector(-1, 0)
	right = Vector(1, 0)
	down = Vector(0, -1)
	up = Vector(0, 1)

class Vector3:
	zero = Vector(0, 0, 0)
	one = Vector(1, 1, 0)
	left = Vector(-1, 0, 0)
	right = Vector(1, 0, 0)
	down = Vector(0, -1, 0)
	up = Vector(0, 1, 0)
	back = Vector(0, 0, -1)
	forward = Vector(0, 0, 1)

from pyx.mat.mat import polar_to_cartesian

def on_circle(n, r=1.0, center=Vector2.zero):
	return [Vector(*polar_to_cartesian(r, 2.0 * math.pi * (i / n))) + center for i in range(n)]

def on_arc(n, r=1.0, center=Vector2.zero, start=0.0, size=2.0 * math.pi): #where start is the start angle and size is the angular size, using default start and size is equal to call on_circle, n > 1
	return [Vector(*polar_to_cartesian(r, start + size * (i / (n - 1)))) + center for i in range(n)]
"""
#Set vector component i and preserve aspect ratio
return vector * (value / vector[i])
"""
