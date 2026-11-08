import cv2
import numpy as np
from PIL import ImageColor
import pyx.numpyx_geo as geo
import pyx.PILx as PILx
from pyx.gamepy.color import Color

def height(self): return int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))

def fps(self): return self.get(cv2.CAP_PROP_FPS)

def frame_count(self): return self.get(cv2.CAP_PROP_FRAME_COUNT)

def size(self): return (height(self), width(self))

def width(self): return int(self.get(cv2.CAP_PROP_FRAME_WIDTH))

def duration(self): return frame_count(self) / fps(self)

def frame_index(self, t): return int(fps(self) * t)

def get_frame(self, index):
	self.set(cv2.CAP_PROP_POS_FRAMES, min(index, frame_count(self) - 1))
	retval, image = self.read()
	return image

cv2.VideoCapture.height = height
cv2.VideoCapture.fps = fps
cv2.VideoCapture.frame_count = frame_count
cv2.VideoCapture.size = size
cv2.VideoCapture.width = width
cv2.VideoCapture.duration = duration




def cv2_point(point):
	# Convert a floating-point coordinate into an integer OpenCV point.
	return tuple(np.round(point[:2]).astype(int))

def cv2_color(color):
	c = Color.parse(color)
	return tuple(int(x) for x in c.rgb32[[2, 1, 0]]), c.a	# Convert RGB to BGR (OpenCV channel order) and return alpha



XFORM_NONE = 0
XFORM_AABB = 1
XFORM_AFFINE = 2

XFORM_MODE = XFORM_AFFINE

def cv2_layer(image, aabb, draw, color, padding=2):
	"""Render a drawing into a temporary layer and return its mask and origin."""

	minimum = np.floor(aabb[:2] - padding).astype(int)
	maximum = np.ceil(aabb[:2] + aabb[2:] + padding).astype(int)

	width, height = maximum - minimum

	if width <= 0 or height <= 0:
		return None

	overlay = np.zeros(
		(height, width, *image.shape[2:]),
		dtype=image.dtype
	)

	mask = np.zeros((height, width), dtype=np.uint8)

	offset = -minimum

	draw(overlay, color, offset)
	draw(mask, 255, offset)

	return overlay, mask, minimum


def cv2_transform_layer(overlay, mask, matrix, origin, image_size):
	"""Apply an affine transformation to a temporary drawing layer."""

	translation = np.eye(3)
	translation[:2, 2] = origin

	warp_matrix = matrix[:2, :] @ translation

	width, height = image_size

	dsize = (width, height)

	overlay = cv2.warpAffine(
		overlay, warp_matrix, dsize,
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT
	)

	mask = cv2.warpAffine(
		mask, warp_matrix, dsize,
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT
	)

	return overlay, mask, np.zeros(2, dtype=int)


def cv2_clip_layer(overlay, mask, origin, image):
	"""Clip a temporary layer to the destination image."""

	x, y = origin
	h, w = mask.shape

	x1 = max(0, x)
	y1 = max(0, y)
	x2 = min(image.shape[1], x + w)
	y2 = min(image.shape[0], y + h)

	if x1 >= x2 or y1 >= y2:
		return None

	overlay = overlay[y1-y:y2-y, x1-x:x2-x]
	mask = mask[y1-y:y2-y, x1-x:x2-x]

	return overlay, mask, image[y1:y2, x1:x2]


def cv2_composite(image, overlay, mask, alpha=1.):
	"""Composite a masked layer onto an image."""

	weight = mask.astype(np.float32) / 255 * alpha

	if image.ndim == 3:
		weight = weight[..., None]

	image[:] = (
		overlay.astype(np.float32) * weight +
		image.astype(np.float32) * (1 - weight)
	).clip(0, 255).astype(image.dtype)


def cv2_paint(image, color, draw, obj=None, padding=2):
	result = cv2_color(color)

	if result is None:
		return

	bgr, alpha = result

	if alpha <= 0:
		return

	# No transformation.
	if obj is None or XFORM_MODE == XFORM_NONE:
		if alpha >= 1:
			draw(image, bgr, np.zeros(2))
		else:
			overlay = image.copy()
			draw(overlay, bgr, np.zeros(2))
			cv2.addWeighted(
				overlay, alpha,
				image, 1 - alpha,
				0, dst=image
			)
		return

	if XFORM_MODE not in (XFORM_AABB, XFORM_AFFINE):
		raise ValueError(f'Invalid XFORM_MODE: {XFORM_MODE}')

	aabb = obj.aabb
	matrix = obj.xform.global_TRS

	# Get the layer's local bounding box.
	layer_aabb = aabb

	if XFORM_MODE == XFORM_AABB:
		layer_aabb = aabb @ matrix

	# Render the temporary layer.
	result = cv2_layer(image, layer_aabb, draw, bgr, padding)

	if result is None:
		return

	overlay, mask, origin = result

	# Apply the complete affine transformation.
	if XFORM_MODE == XFORM_AFFINE:
		overlay, mask, origin = cv2_transform_layer(
			overlay, mask, matrix, origin,
			(image.shape[1], image.shape[0])
		)

	# Clip to the destination image.
	result = cv2_clip_layer(overlay, mask, origin, image)

	if result is None:
		return

	overlay, mask, image_region = result

	# Composite.
	cv2_composite(image_region, overlay, mask, alpha)

def draw_image(self, image):
	if self.array is None:
		return

	source = np.asarray(self.array).transpose(1, 0, 2)

	if source.ndim == 2:
		source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
	elif source.shape[2] == 3:
		source = cv2.cvtColor(source, cv2.COLOR_RGB2BGRA)
	elif source.shape[2] == 4:
		source = cv2.cvtColor(source, cv2.COLOR_RGBA2BGRA)

	size = np.asarray(self.size, dtype=float)

	if np.any(size <= 0):
		return

	resized = cv2.resize(
		source,
		tuple(np.round(size).astype(int)),
		interpolation=cv2.INTER_LINEAR
	)

	opacity = self.attrib.get('opacity', 1.0)

	def draw(img, color, offset):
		x, y = np.round(self.min + offset).astype(int)
		h, w = resized.shape[:2]

		x1, y1 = max(0, x), max(0, y)
		x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)

		if x1 >= x2 or y1 >= y2:
			return

		roi = resized[y1-y:y2-y, x1-x:x2-x]

		if img.ndim == 2:
			img[y1:y2, x1:x2] = roi[:, :, 3] * opacity
		else:
			img[y1:y2, x1:x2] = roi[:, :, :3]

	cv2_paint(
		image,
		(255, 255, 255, opacity),
		draw,
		self,
		padding=0
	)

"""def cv2_paint(image, color, draw, obj=None, padding=2):
	result = cv2_color(color)

	if result is None:
		return

	bgr, alpha = result

	# No transformation.
	if obj is None or XFORM_MODE == XFORM_NONE:
		if alpha >= 1:
			draw(image, bgr, np.zeros(2))
		else:
			overlay = image.copy()
			draw(overlay, bgr, np.zeros(2))
			cv2.addWeighted(
				overlay, alpha,
				image, 1 - alpha,
				0, dst=image
			)
		return

	# Get the local bounding box.
	aabb = obj.aabb
	matrix = obj.xform.global_TRS

	# Transform the bounding box, keeping the geometry axis-aligned.
	if XFORM_MODE == XFORM_AABB:
		aabb = aabb @ matrix

	minimum = np.floor(aabb[:2] - padding).astype(int)
	maximum = np.ceil(aabb[:2] + aabb[2:] + padding).astype(int)
	#minimum = np.floor(aabb.min - padding).astype(int)
	#maximum = np.ceil(aabb.max + padding).astype(int)

	size = np.asarray(maximum - minimum, dtype=int).reshape(-1)
	
	if size.size != 2 or np.any(size <= 0):
		return
	
	width, height = size
	
	overlay = np.zeros(
		(height, width, *image.shape[2:]),
		dtype=image.dtype
	)
	
	mask = np.zeros((height, width), dtype=np.uint8)

	offset = -minimum

	draw(overlay, bgr, offset)
	draw(mask, 255, offset)

	# Apply the complete affine transformation.
	if XFORM_MODE == XFORM_AFFINE:
		translation = np.eye(3)
		translation[:2, 2] = minimum

		warp_matrix = matrix[:2, :] @ translation

		dsize = (image.shape[1], image.shape[0])

		overlay = cv2.warpAffine(
			overlay, warp_matrix, dsize,
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT
		)

		mask = cv2.warpAffine(
			mask, warp_matrix, dsize,
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT
		)

		x, y = 0, 0

	else:
		# Position the untransformed layer at the transformed AABB.
		x, y = minimum

	# Clip the drawing to the image boundaries.
	h, w = mask.shape

	x1 = max(0, x)
	y1 = max(0, y)
	x2 = min(image.shape[1], x + w)
	y2 = min(image.shape[0], y + h)

	if x1 >= x2 or y1 >= y2:
		return

	overlay = overlay[y1-y:y2-y, x1-x:x2-x]
	mask = mask[y1-y:y2-y, x1-x:x2-x]

	# Composite.
	weight = mask.astype(np.float32) / 255 * alpha

	if image.ndim == 3:
		weight = weight[..., None]

	image_region = image[y1:y2, x1:x2]

	image_region[:] = (
		overlay.astype(np.float32) * weight +
		image_region.astype(np.float32) * (1 - weight)
	).clip(0, 255).astype(image.dtype)"""



def cv2_stroke(obj):
	# Get the outline color from the shape attributes.
	return obj.attrib.get('stroke')


def cv2_fill(obj):
	# Get the interior color from the shape attributes.
	return obj.attrib.get('fill')


def cv2_stroke_width(obj):
	# Get the outline thickness, defaulting to one pixel.
	return obj.attrib.get('stroke-width', 1)


def cv2_draw_curve(image, points, color, width=1, closed=False):
	# Convert curve coordinates into OpenCV integer coordinates.
	points = np.round(points[:, :2]).astype(np.int32)

	if len(points) < 2:
		return

	# Draw connected line segments, optionally connecting the last
	# point back to the first one.
	cv2.polylines(
		image, [points], closed, color,
		max(1, round(width)), cv2.LINE_AA
	)


def cv2_draw_polygon(image, points, color):
	# Convert vertices into OpenCV integer coordinates.
	points = np.round(points[:, :2]).astype(np.int32)

	# Fill the polygon using antialiased edges.
	cv2.fillPoly(image, [points], color, lineType=cv2.LINE_AA)


#DRAW SHAPES

def draw_circle(self, image):
	center = cv2_point(self.center)
	radius = round(self.radius)
	width = cv2_stroke_width(self)

	cv2_paint(image, cv2_fill(self),
		lambda img, color, offset: cv2.circle(
			img, tuple(center + offset), radius, color, cv2.FILLED,
			lineType=cv2.LINE_AA
		), self
	)

	cv2_paint(image, cv2_stroke(self),
		lambda img, color, offset: cv2.circle(
			img, tuple(center + offset), radius, color,
			max(1, round(width)),
			lineType=cv2.LINE_AA
		), self
	)


def draw_arc(self, image):
	center = cv2_point(self.center)
	axes = np.round(self.radius).astype(int)

	start_angle = np.degrees(self.start)
	end_angle = np.degrees(self.end)

	cv2_paint(image, cv2_stroke(self),
		lambda img, color, offset: cv2.ellipse(
			img,
			tuple(center + offset),
			tuple(axes),
			0,
			start_angle,
			end_angle,
			color,
			max(1, round(cv2_stroke_width(self))),
			lineType=cv2.LINE_AA
		), self
	)


def draw_polyline(self, image):
	# Get the polyline vertices.
	points = np.asarray(self)

	if len(points) < 2:
		return

	# Check whether the polyline forms a closed contour.
	closed = self.closed

	if closed:
		# Only closed polylines can have a filled interior.
		cv2_paint(image, cv2_fill(self),
			lambda img, color, offset: cv2_draw_polygon(
				img, points + offset, color
			), self
		)

	# Draw the outline, closing the final segment when necessary.
	cv2_paint(image, cv2_stroke(self),
		lambda img, color, offset: cv2_draw_curve(
			img, points + offset, color, cv2_stroke_width(self), closed
		), self
	)


def draw_line(self, image):
	# Reuse the polyline renderer with a single open segment.
	# Copy the original attributes so fill and stroke settings are preserved.
	draw_polyline(
		geo.polyline(self, closed=False).set(**getattr(self, 'attrib', {})),
		image
	)


def draw_rect(self, image):
	# Convert the rectangle corners into a closed polyline.
	# Closing the contour ensures that all four edges are drawn.
	draw_polyline(
		geo.polyline(self.corners, closed=True).set(**getattr(self, 'attrib', {})),
		image
	)


def draw_bezier(self, image):
	# Sample the Bezier curve into a polyline for rasterization.
	# Preserve the original Bezier attributes on the sampled geometry.
	draw_polyline(
		self.sample(steps=100).set(**getattr(self, 'attrib', {})),
		image
	)




"""def draw_image(self, image):
	if self.array is None:
		return

	source = np.asarray(self.array).transpose(1, 0, 2)

	if source.ndim == 2:
		source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
	elif source.shape[2] == 3:
		source = cv2.cvtColor(source, cv2.COLOR_RGB2BGRA)
	elif source.shape[2] == 4:
		source = cv2.cvtColor(source, cv2.COLOR_RGBA2BGRA)

	size = np.asarray(self.size, dtype=float)

	if np.any(size <= 0):
		return

	resized = cv2.resize(
		source,
		tuple(np.round(size).astype(int)),
		interpolation=cv2.INTER_LINEAR
	)

	opacity = self.attrib.get('opacity', 1.0)

	def draw(img, color, offset):
		x, y = np.round(self.min + offset).astype(int)
		h, w = resized.shape[:2]
	
		x1, y1 = max(0, x), max(0, y)
		x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
	
		if x1 >= x2 or y1 >= y2:
			return
	
		roi = resized[y1-y:y2-y, x1-x:x2-x]
	
		if img.ndim == 2:
			img[y1:y2, x1:x2] = roi[:, :, 3]
			return
	
		alpha = roi[:, :, 3:4].astype(np.float32) / 255
		alpha *= opacity
	
		foreground = roi[:, :, :3].astype(np.float32)
		background = img[y1:y2, x1:x2].astype(np.float32)
	
		img[y1:y2, x1:x2] = (
			foreground * alpha +
			background * (1 - alpha)
		).clip(0, 255).astype(img.dtype)

	cv2_paint(
		image,
		(255, 255, 255, 255),
		draw,
		self,
		padding=0
	)"""




# Dynamically attach the corresponding drawing method to each geometry class.
for cls in [geo.arc, geo.bezier, geo.circle, geo.line, geo.polyline, geo.rect, PILx.image]:
	name = cls.__name__

	# Capture the function name in the lambda default argument.
	# This prevents every generated method from referencing the last class.
	setattr(
		cls,
		f'draw',
		lambda self, image, name=name: globals()[f'draw_{name}'](self, image)
	)
