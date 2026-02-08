#class Tween:
from pyx.timeline import Instant, Layer

class Keyframe(Instant):
	def __init__(self, time, value):
		super().__init__(time)
		self.value = value

	"""@property
	def start_frame(self): return self.start * Movie.fps
	@property
	def end_frame(self): return self.end * Movie.fps"""

class Channel(Layer):
	def __init__(self, node, property, keyframes=None):
		super().__init__([] if keyframes is None else keyframes)
		self.node = node
		self.property = property

	def add_keyframe(self, time, value):
		self.append(Keyframe(time, value))
		return self

	def add_keyframes(self, frames):
		for x in frames:
			self.add_keyframe(*x)
		return self

	def to_usda(self, indent=0):
		return '\t' * indent + f'double3 xformOp:{self.property}.timeSamples = ' + '{' + ','.join([f'{x.start * Movie.fps}: {x.value}' for x in self]) + '}'

class Clip(Layer):
	def __init__(self, channels, name='clip'):
		super().__init__(channels)
		self.name = name

	@property
	def start_frame(self): return self.start * Movie.fps
	@property
	def end_frame(self): return self.end * Movie.fps

	def to_usda(self, indent=0):
		return '\n'.join([x.to_usda(indent=indent) for x in self])

class Movie(Clip):
	fps = 24

	def __init__(self, clips, name='movie'):
		super().__init__(clips, name)

	def to_usda(self):
		return '\n'.join(
			[
				'(',
				f'\tstartTimeCode = {self.start_frame}',
				f'\tendTimeCode = {self.end_frame}',
				f'\tframesPerSecond = {self.fps}',
				f'\tdefaultPrim = "World"',
				')'
			]
		)
