import numpy as np
from PIL import ImageColor
import re

class Color(np.ndarray):
	
	def __new__(cls, *arr):
		arr = np.asarray(arr)
		if arr.size == 3:
			arr = np.append(arr, 1.)
		return arr.view(cls)
	
	@classmethod
	def parse(cls, value):
		if isinstance(value, (list, tuple, np.ndarray)):
			value = np.array(value, dtype=float)
			if len(value) not in (3, 4):
				raise ValueError("Color must have 3 or 4 values")
			if len(value) == 3:
				value = np.append(value, 1.)
			value[:3] /= 255. if value[:3].max() > 1 else 1.
			value[3] /= 255. if value[3] > 1 else 1.
			return cls(*value)
		if isinstance(value, str) and value.startswith("rgba"):
			nums = np.array(re.findall(r'[\d.]+', value), dtype=float)
			return cls(*(np.append(nums[:3] / 255., nums[3:4] if len(nums) > 3 else 1.)))
		if isinstance(value, str) and value.startswith("#") and len(value) == 9:	#FROM HEX
			return cls.from_hex(value)
		return cls.from_name(value)	#FROM NAME
	
	@classmethod
	def from_name(cls, value):
		if value.lower() in ('none', 'transparent'):
			return cls(*np.zeros(4))
		return cls(*np.array(ImageColor.getrgb(value)) / 255., 1.)

	@classmethod
	def from_hex(cls, value):
		if len(value) == 7:
			value += 'ff'
		return cls(*(int(value[i:i+2], 16) / 255. for i in range(1, 9, 2)))
	
	# -------- Properties -------- #

	@property
	def r(self): return float(self[0])

	@property
	def g(self): return float(self[1])

	@property
	def b(self): return float(self[2])

	@property
	def a(self): return float(self[3])	#if len(self) > 3 else 1.0

	@property
	def rgba(self): return self[:4]

	@property
	def rgb(self): return self[:3]

	@property
	def rgba32(self): return (self.rgba * 255).astype(int)

	@property
	def rgb32(self): return self.rgba32[:3]

	@property
	def hex(self): return '#' + ''.join(f'{v:02x}' for v in self.rgba32)

	# -------- Helpers -------- #

	def with_alpha(self, a):
		"""Return new color with modified alpha (0-255 or 0-1)."""
		if a <= 1:
			a = int(a * 255)
		return Color(*self.rgba255[:3], a)

	def invert(self):	#Invert RGB (keep alpha).
		rgb = 1.0 - self[:3]
		return Color(*rgb, self[3])

	def grayscale(self):	#Convert to grayscale using luminance formula.
		r, g, b = self[:3]
		value = 0.299*r + 0.587*g + 0.114*b
		return gray(value, self[3])

	def lighten(self, amount):	#Lighten toward white. amount: 0–1
		rgb = self[:3] + (1.0 - self[:3]) * amount
		return np.clip(np.array([*rgb, self[3]]), 0.0, 1.0)

	def darken(self, amount):	#Darken toward black. amount: 0–1
		rgb = self[:3] * (1.0 - amount)
		return np.clip(np.array([*rgb, self[3]]), 0.0, 1.0)

	def adjust_brightness(self, amount):	#Add/subtract brightness. amount: -1 to +1
		rgb = self[:3] + amount
		return np.clip(np.array([*rgb, self[3]]), 0.0, 1.0)

	def adjust_contrast(self, amount):	#Contrast adjustment. amount: -1 to +1. 0 = no change
		factor = 1.0 + amount
		rgb = 0.5 + (self[:3] - 0.5) * factor
		return np.clip(np.array([*rgb, self[3]]), 0.0, 1.0)

	def gamma_correct(self, gamma):	#Apply gamma correction. gamma > 1 darkens. gamma < 1 lightens
		rgb = self[:3] ** gamma
		return np.clip(np.array([*rgb, self[3]]), 0.0, 1.0)


def gray(value, a=1.0): return Color([value, value, value, a])

def shade(color, amt):
	#print(color)
	return np.clip(color * (1 + amt) if amt < 0 else color + amt, 0, 1)
