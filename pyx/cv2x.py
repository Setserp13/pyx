import cv2
import numpy as np
from PIL import ImageColor
import pyx.numpyx_geo as geo

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
	# Ignore missing or transparent colors.
	if color is None or color == 'none':
		return None

	# Convert named or hexadecimal colors into RGBA.
	if isinstance(color, str):
		r, g, b, a = ImageColor.getcolor(color, 'RGBA')
	else:
		values = list(color)

		# RGB colors are assumed to be fully opaque.
		if len(values) == 3:
			r, g, b = values
			a = 255
		else:
			r, g, b, a = values

		# Support normalized color components in the range [0, 1].
		if max(r, g, b, a) <= 1:
			r, g, b, a = [int(x * 255) for x in (r, g, b, a)]

	# OpenCV uses BGR instead of RGB.
	return (int(b), int(g), int(r)), a / 255


def cv2_paint(image, color, draw):
	# Convert the color into OpenCV format and extract its opacity.
	result = cv2_color(color)

	if result is None:
		return

	bgr, alpha = result

	if alpha >= 1:
		# Fully opaque colors can be drawn directly.
		draw(image, bgr)
	else:
		# Draw translucent colors on a temporary image.
		overlay = image.copy()
		draw(overlay, bgr)

		# Blend the overlay with the original image.
		cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


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


def draw_circle(self, image):
	# Get the circle geometry and outline thickness.
	center = cv2_point(self.center)
	radius = round(self.radius)
	width = cv2_stroke_width(self)

	# Draw the interior, if a fill color is defined.
	cv2_paint(image, cv2_fill(self),
		lambda img, color: cv2.circle(
			img, center, radius, color, cv2.FILLED,
			lineType=cv2.LINE_AA
		)
	)

	# Draw the outline, if a stroke color is defined.
	cv2_paint(image, cv2_stroke(self),
		lambda img, color: cv2.circle(
			img, center, radius, color, max(1, round(width)),
			lineType=cv2.LINE_AA
		)
	)


def draw_arc(self, image):
	# Sample the arc into points so it can be drawn as a polyline.
	points = cv2_curve_points(self)

	# Arcs are open curves, so only their outline is drawn.
	cv2_paint(image, cv2_stroke(self),
		lambda img, color: cv2_draw_curve(
			img, points, color, cv2_stroke_width(self)
		)
	)


def draw_polyline(self, image):
	# Get the polyline vertices.
	points = np.asarray(self.vertices)

	if len(points) < 2:
		return

	# Check whether the polyline forms a closed contour.
	closed = self.closed

	if closed:
		# Only closed polylines can have a filled interior.
		cv2_paint(image, cv2_fill(self),
			lambda img, color: cv2_draw_polygon(
				img, points, color
			)
		)

	# Draw the outline, closing the final segment when necessary.
	cv2_paint(image, cv2_stroke(self),
		lambda img, color: cv2_draw_curve(
			img, points, color, cv2_stroke_width(self), closed
		)
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


# Dynamically attach the corresponding drawing method to each geometry class.
for cls in [geo.arc, geo.bezier, geo.circle, geo.line, geo.polyline]:
	name = cls.__name__

	# Capture the function name in the lambda default argument.
	# This prevents every generated method from referencing the last class.
	setattr(
		cls,
		f'draw',
		lambda self, image, name=name: globals()[f'draw_{name}'](self, image)
	)
