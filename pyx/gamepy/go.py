import numpy as np
import pyx.numpyx as npx
from pyx.gamepy.color import Color
import pyx.PILx as PILx
from pyx.mat.transform import Node2D, Node3D
import pyx.osx as osx
from PIL import Image
import uuid

class Texture2D():
	def __init__(self, path):
		self.path = path
		self.array = PILx.read_image(path)
		self.id = uuid.uuid4()
	
	@property
	def size(self): return np.array(self.array.shape[:2])

	@property
	def size(self): return np.array(self.array.shape[:2])

class Sprite2D(Node2D):
	def __init__(self, texture=None, pivot=np.ones(2) * 0.5, region_rect = npx.rect(np.zeros(2), np.ones(2)), border=None, **kwargs):
		self.texture = texture
		self.pivot = pivot
		self.region_rect = region_rect
		self.border = border	#used by 9-slice	#l, t, r, b
		self.modulate = Color(1.0, 1.0, 1.0, 1.0)
		self.self_modulate = Color(1.0, 1.0, 1.0, 1.0)
		self.draw_mode = None	#[None, 'tiled', '9-slice']
		#pivot, region_rect and border are normalized
		super().__init__(**kwargs)

	@property
	def size(self): return self.texture.size * self.region_rect.size * self.scale
	@size.setter
	def size(self, value): self.scale = value / (self.texture.size * self.region_rect.size)

	@property
	def self_aabb(self): return npx.rect(np.zeros(2), self.size).set_position(self.pivot, self.position)

	@property
	def aabb(self): return npx.aabb([x.self_aabb for x in [self, *self.descendants()] if hasattr(x, 'self_aabb')])

	@property
	def border_pixels(self): return np.concatenate((self.border[:2] * self.texture.size, self.border[2:] * self.texture.size))

	@property
	def region_rect_pixels(self): return npx.rect(np.zeros(2), self.texture.size).denormalize_rect(self.region_rect)

class MeshInstance3D(Node3D):
	def __init__(self, mesh=None, pivot=np.ones(3) * 0.5, **kwargs):	#mesh can be a PrimitiveMesh, a path, or a Mesh
		self.mesh = mesh
		self.pivot = pivot
		super().__init__(**kwargs)

class Sprite2DAnimation():
	def __init__(self, texture=None, duration=1.0, name=None, loop=True, speed=1.0, frame_duration_based=False):
		self.texture = texture
		self.name = osx.filename(texture.path) if name is None else name
		self.loop = loop
		self.speed = speed
		regions = PILx.get_atlases(Image.open(texture.path))
		frame_duration = duration if frame_duration_based else duration / len(regions)
		self.frames = []
		for i, x in enumerate(regions):
			self.frames.append({"region": x, "duration": frame_duration})

	@property
	def duration(self): return sum(x['duration'] for x in self.frames)
