import numpy as np
from PIL import ImageColor

class Color(np.ndarray):
	"""
	Color stored as float32 RGBA in range 0-1.
	Subclass of numpy.ndarray.
	"""

	def __new__(cls, *args):
		# Parse input
		if len(args) == 1: # hex string or color name
			r, g, b, *a = cls._parse_color(args[0])
			a = a[0] if a else 1.0
		elif len(args) in (3, 4):
			r, g, b = args[:3]
			a = args[3] if len(args) == 4 else 1.0
		else:
			raise ValueError("Color expects hex, name, or R,G,B[,A]")
		
		# Convert to array
		arr = np.array([r, g, b, a], dtype=np.float32)
		
		# Create subclassed ndarray
		obj = np.asarray(arr).view(cls)
		return obj

	@classmethod
	def _parse_color(cls, value):
		# Se já for lista/tuple com 3 ou 4 valores, use direto
		if isinstance(value, (list, tuple)):
			if len(value) == 3:
				r, g, b = value
				a = 1.0
			elif len(value) == 4:
				r, g, b, a = value
			else:
				raise ValueError("Color list/tuple must have 3 or 4 values")
			return r, g, b, a

		# Caso contrário, assume string e deixa o ImageColor tratar
		r, g, b = ImageColor.getrgb(value)
		return r / 255, g / 255, b / 255, 1.0

	# -------- Properties -------- #

	@property
	def r(self): return float(self[0])

	@property
	def g(self): return float(self[1])

	@property
	def b(self): return float(self[2])

	@property
	def a(self): return float(self[3])

	@property
	def rgb(self):
		return tuple(float(x) for x in self[:3])

	@property
	def rgba(self):
		return tuple(float(x) for x in self[:4])

	@property
	def rgb255(self):
		return tuple(int(x * 255) for x in self[:3])

	@property
	def rgba255(self):
		return tuple(int(x * 255) for x in self[:4])

	@property
	def hex(self):
		r, g, b, a = self.rgba255
		return f"#{r:02x}{g:02x}{b:02x}{a:02x}"

	# -------- Helpers -------- #

	def with_alpha(self, a):
		"""Return new color with modified alpha (0-255 or 0-1)."""
		if a <= 1:
			a = int(a * 255)
		return Color(*self.rgba255[:3], a)

def gray(value): return Color([value, value, value, 1.0])

def shade(color, amt):
	#print(color)
	return np.clip(color * (1 + amt) if amt < 0 else color + amt, 0, 1)
