import math
import xml.etree.ElementTree as ET
from lxml import etree
from functools import reduce
import pyx.rex as rex
from PIL import Image
import uuid
import os
import pyx.osx as osx
import pyx.PILx as PILx
import svgpathtools
from svgpathtools import parse_path
import svgutils.transform as sg
import re
from pyx.collectionsx import List, to_str, Dict, split_before
from pyx.lxmlx import *	#localname
import numpy as np
import pyx.numpyx as npx
import pyx.numpyx_geo as geo
import pyx.lxmlx as lxmlx
from pyx.mat.transform import Node2D, Transform
import pyx.mat.transform as xf
from pyx.gamepy.color import Color
import pyx.generic.generic as generic

def replace_all(root, replacements, exact_match=False):
	"""
	replacements: dict {old_text: new_text}
	"""
	for el in root.iter():
		try:
			tag = etree.QName(el).localname
		except: continue
		if tag in ("text", "tspan", "textPath"):
			if el.text:
				for k, v in replacements.items():
					if exact_match:
						if el.text == k:
							el.text = v
					else:
						el.text = el.text.replace(k, v)
			#if el.text and el.text in replacements:
			#	el.text = replacements[el.text]
	return root




from pyx.mat.transform import R2

def svg_arc_to_center(start, end, rx, ry, rotation, large_arc_flag, sweep_flag):
	start, end = np.asarray([start, end], dtype=float)
	rx, ry = abs(rx), abs(ry)

	if rx == 0 or ry == 0 or np.allclose(start, end):
		return None

	phi = math.radians(rotation)
	R = R2(phi)

	x1p, y1p = R @ ((start - end) / 2)

	scale = max(1., np.sqrt(x1p**2 / rx**2 + y1p**2 / ry**2))
	rx, ry = rx * scale, ry * scale

	rx2, ry2 = rx**2, ry**2
	k = np.sqrt(max(0., (rx2 * ry2 - rx2 * y1p**2 - ry2 * x1p**2) /
		(rx2 * y1p**2 + ry2 * x1p**2)))

	if large_arc_flag == sweep_flag:
		k = -k

	cxp, cyp = k * rx * y1p / ry, -k * ry * x1p / rx
	center = R.T @ [cxp, cyp] + (start + end) / 2

	u = np.array([(x1p - cxp) / rx, (y1p - cyp) / ry])
	v = np.array([(-x1p - cxp) / rx, (-y1p - cyp) / ry])

	start_angle = np.arctan2(u[1], u[0])
	delta = np.arctan2(np.cross(u, v), np.dot(u, v))

	if sweep_flag:
		delta %= math.tau
	else:
		delta = -(-delta % math.tau)

	return center, start_angle, start_angle + delta







from pyx.numpyx_geo import shape, line, arc, bezier
class path_command:
	def __init__(self, type, *parameters):
		self.type = type
		self.parameters = list(parameters)

	@property
	def relative(self):
		return self.type.islower()

	@relative.setter
	def relative(self, value):
		self.type = self.type.lower() if value else self.type.upper()

	@property
	def d(self):
		if self.type.upper() == 'Z':
			return self.type

		return f"{self.type} {' '.join(map(str, self.parameters))}"

	def __repr__(self): return self.d

	def __len__(self): return len(self.parameters)

	def __iter__(self): return iter(self.parameters)

	def __getitem__(self, index): return self.parameters[index]

	def __setitem__(self, index, value): self.parameters[index] = value


class path(list, shape):
	def __init__(self, *commands):
		list.__init__(self, commands)

	@property
	def d(self): return ' '.join(command.d for command in self)

	def move_to(self, v): self.append(path_command('M', *v))
	def move_by(self, v): self.append(path_command('m', *v))
	def line_to(self, v): self.append(path_command('L', *v))
	def line_by(self, v): self.append(path_command('l', *v))
	def bezier_to(self, *v):
		v = np.asarray(v)
		self.append(path_command('LQC'[len(v) - 1], *v.flatten()))
	def close(self): self.append(path_command('Z'))

	@classmethod
	def parse(cls, s):
		p = re.compile(r'[A-Za-z]|[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?')

		return cls(*[
			path_command(group[0], *map(float, group[1:]))
			for group in split_before(p.findall(s), str.isalpha)
		])

	@property
	def subpaths(self):
		return [
			path(*group)
			for group in split_before(self, lambda cmd: cmd.type in 'Mm')
		]


	@property
	def normalized(self):
		result = path()

		sizes = {
			'M': 2, 'L': 2,
			'H': 1, 'V': 1,
			'C': 6, 'S': 4,
			'Q': 4, 'T': 2,
			'A': 7, 'Z': 0
		}

		for cmd in self:
			t = cmd.type
			upper = t.upper()
			p = cmd.parameters
			size = sizes[upper]

			if size == 0:
				result.append(path_command(t))
				continue

			if len(p) < size or len(p) % size:
				raise ValueError(f"Invalid parameters for SVG command {t}")

			for i in range(0, len(p), size):
				command_type = t

				if upper == 'M' and i > 0:
					command_type = 'l' if t.islower() else 'L'

				result.append(path_command(
					command_type,
					*p[i:i + size]
				))

		return result


	@property
	def segments(self):
		segments = []

		current = np.zeros(2, dtype=float)
		start = current.copy()

		previous_type = None
		previous_control = None
		#print(self.normalized)
		for cmd in self.normalized:
			#print(cmd)
			t = cmd.type.upper()
			p = np.asarray(cmd.parameters, dtype=float)
			relative = cmd.relative

			def points(v):
				v = np.asarray(v, dtype=float).reshape(-1, 2)
				return current + v if relative else v

			if t == 'M':
				current, = points(p)
				start = current.copy()

			elif t in 'HV':
				axis = 'HV'.index(t)
				end = current.copy()
				end[axis] = current[axis] + p[0] if relative else p[0]
				segments.append(bezier([current.copy(), end.copy()]))
				current = end

			elif t in 'LQCTS':
				v = points(p)
				
				if t in 'TS':
					p1 = (
						2 * current - previous_control
						if previous_type in {'Q': 'QT', 'S': 'CS'}[t]
						else current.copy()
					)

					v = np.insert(v, 0, p1, axis=0)

				segments.append(bezier([current.copy(), *v]))

				current = v[-1]
				previous_control = None if t == 'L' else v[-2]

			elif t == 'A':
				rx, ry, rotation, large, sweep, x, y = p
				end, = points([x, y])

				if not np.isclose(rx, ry) or not np.isclose(rotation, 0):
					raise NotImplementedError(
						"Only circular arcs without rotation are supported"
					)

				result = svg_arc_to_center(
					current,
					end,
					rx,
					ry,
					rotation,
					int(large),
					int(sweep)
				)

				if result is not None:
					center, start_angle, end_angle = result

					segments.append(arc(
						center,
						rx,
						start_angle,
						end_angle
					))
				else:
					segments.append(bezier([
						current.copy(), end.copy()
					]))

				current = end

			elif t == 'Z':
				if not np.allclose(current, start):
					segments.append(bezier([
						current.copy(), start.copy()
					]))

				current = start.copy()

			else:
				raise ValueError(f"Unsupported SVG command: {cmd.type}")

			if t not in ('C', 'S', 'Q', 'T'):
				previous_control = None

			previous_type = t

		return geo.group(segments)

	@property
	def aabb(self):
		segments = self.segments
		return segments.aabb
	@aabb.setter
	def aabb(self, value):
		pass

generic.make_Element(path)

def path_to_svg(obj, **kwargs): return lxmlx.element("path", d=obj.d, **kwargs)
path.to_svg = lambda self: path_to_svg(self, **get_attrib(self))

def path_from_svg(obj): return path.parse(obj.get('d'))











def parse_svg_transform(transform):
	M = np.eye(3)
	pattern = r"([a-zA-Z]+)\s*\(([^)]*)\)"

	for name, args in re.findall(pattern, transform):
		values = [ float(v) for v in re.split(r"[,\s]+", args.strip()) if v ]
		if name == "translate":
			T = xf.T(np.array([values[0], values[1] if len(values) > 1 else 0]))

		elif name == "scale":
			T = xf.S(np.array([values[0], values[1] if len(values) > 1 else values[0]]))

		elif name == "rotate":
			angle = math.radians(values[0])
			T = xf.R2(angle)
			if len(values) == 3:
				c = np.array([values[1], values[2]])
				T1 = xf.T(c)
				T2 = xf.T(-c)
				T = T1 @ T @ T2

		elif name == "matrix":
			a, b, c, d, e, f = values
			T = np.array([[a, c, e], [b, d, f], [0, 0, 1]])

		else:
			raise ValueError(f"Unsupported transform: {name}")

		M = M @ T

	return M


def transform_from_svg(obj):
	transform = obj.get('transform', '')
	M = parse_svg_transform(transform)
	a, c, tx = M[0]
	b, d, ty = M[1]
	sx = math.hypot(a, b)
	rotation = math.atan2(b, a)
	sy = (a * d - b * c) / sx
	return Node2D(position=(tx, ty), scale=(sx, sy), rotation=rotation)


def from_svg(obj, asnode=False):
	shape = None
	node = transform_from_svg(obj)
	tag = etree.QName(obj).localname
	#print(tag)
	match tag:
		case 'circle': shape = circle_from_svg(obj)
		case 'ellipse': shape = ellipse_from_svg(obj)
		case 'g' | 'svg':
			children = [from_svg(x, asnode) for x in obj]
			children = [x for x in children if not x is None]
			if len(children) == 0:
				return None
			if asnode:
				node.extend(children)
			else:
				shape = geo.group(children)
				for x in shape:
					node.append(x.attrib['transform'])
			
		case 'line': shape = line_from_svg(obj)
		case 'path': shape = path_from_svg(obj)
		case 'polyline': shape = polyline_from_svg(obj)
		case 'polygon': shape = polygon_from_svg(obj)
		case 'rect': shape = rect_from_svg(obj)
		case 'text': shape = text_from_svg(obj)

	if asnode:
		node.shape = shape
		result = node
	else:
		if shape is None:
			return None
		shape.set(transform=node)
		result = shape

	result.id = obj.get('id')
	result.set(**get_colors(obj))

	desc = lxmlx.find(obj, lambda x: etree.QName(x).localname == "desc", iter=lambda e: e)
	#print(desc)
	result.desc = {} if desc is None else rex.strpdict(desc.text, sep=[';', '='])
	return result


def page_rect(obj):
	viewBox = obj.get('viewBox')
	return geo.rect2(0, 0, *get(obj, float, 'width', 'height')) if viewBox is None else geo.rect2(*[float(x) for x in viewBox.split(' ')])

def set_page_size(svg, size):
	set(svg, viewBox=f"0 0 {size[0]} {size[1]}", width=size[0], height=size[1])	

def rotate_page(svg, angle=90):	#angle is in degree
	page_size = page_rect(svg).size
	cx, cy = page_size / 2

	set_page_size(svg, page_size[[1, 0]])	# swap page size

	g1 = g(
		transform=f"translate({cy} {cx}) rotate({angle}) translate({-cx} {-cy})"
	)

	for child in list(svg):	# move all elements into group
		svg.remove(child)
		g1.append(child)

	svg.append(g1)
	return svg

def circle_from_svg(obj): return geo.circle(get(obj, float, 'cx', 'cy'), *get(obj, float, 'r'))
def ellipse_from_svg(obj): return geo.ellipse(np.array(get(obj, float, 'cx', 'cy')), np.array(get(obj, float, 'rx', 'ry')) * 2.)
def rect_from_svg(obj): return geo.rect2(*get(obj, float, 'x', 'y', 'width', 'height'))
def line_from_svg(obj): return geo.line([get(obj, float, 'x1', 'y1'), get(obj, float, 'x2', 'y2')])
def parse_points2(s):
	matches = re.findall(r'(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)', s)
	return np.array(matches, dtype=float)
def polygon_from_svg(obj):  return geo.polyline(parse_points2(obj.get('points')), closed=True)
def polyline_from_svg(obj): return geo.polyline(parse_points2(obj.get('points')), closed=False)

def remove_unit(value): return float(re.match(r'[-+]?\d*\.?\d+', value).group())

def get_tspan_attrib(elem):
	result = {}
	result['inner_text'] = elem.text or ''
	result['position'] = np.array(get(elem, float, 'x', 'y'))
	result['pivot'] = np.zeros(2)	#np.ones(2) * .5
	style = get_style(elem)
	for k in ('font-family', 'font-size'):
		value = style.get(k, elem.get(k, None))
		if value:
			if k in ('font-size'):
				if isinstance(value, str):
					value = remove_unit(value)
			result[{'font-family': 'font', 'font-size': 'font_size'}[k]] = value
	return result

def tspan_from_svg(elem):
	attrib = get_tspan_attrib(elem)
	return PILx.tspan(**attrib)

def text_from_svg(elem):
	attrib = get_tspan_attrib(elem)
	result = PILx.text(**attrib)
	lines = []

	for child in elem:
		if child.tag.endswith('tspan'):
			line = tspan_from_svg(child)

			if line.font is None:
				line.font = result.font

			if line.font_size is None:
				line.font_size = result.font_size

			lines.append(line)
	result.inner_text = '\n'.join([x.inner_text for x in lines])
	return result


def find_layers(root):
	ns = {'svg': 'http://www.w3.org/2000/svg', 'inkscape': 'http://www.inkscape.org/namespaces/inkscape'}
	return [
		x for x in root.findall('.//svg:g', namespaces=ns) if x.get('{http://www.inkscape.org/namespaces/inkscape}groupmode', None) == 'layer'
	]

def remove_at(ls, i): return ls[:i] + ls[i+1:]

def root_rects(rects): #Rects that are not subrects of another one in the list
	return [x for i, x in enumerate(rects) if len(list(filter(lambda y: y.contains_rect(x), remove_at(rects, i)))) == 0]





def get_style(element): return rex.strpdict(element.get('style', ''))

def get_style_property(element, property):
	try:
		return get_style(element)[property]
	except:
		return None

def get_style_properties(element, *properties): return Dict.items(get_style(element), properties)	#List.items(get_style(element), properties)

def set_style(element, **kwargs):
	style = get_style(element)
	for k in kwargs:
		style[k] = kwargs[k]
	element.set("style", rex.strfdict(style))

def get_colors(obj):
	result = {}
	style = get_style(obj)
	dflt_values = {'fill': 'black', 'stroke': 'none'}
	for k in ('fill', 'stroke', 'stop-color', 'flood-color', 'solid-color', 'lighting-color', 'color'):
		value = style.get(k, obj.attrib.get(k, dflt_values.get(k)))	# style overwrites presentation attributes
		if value is None:
			continue
		try:
			value = Color.parse(value)
			a = f"{k.replace('-color', '')}-opacity"
			opacity = style.get(a, obj.attrib.get(a))
			if opacity is not None:
				value[3] = float(opacity)
		except: pass
		result[k] = value
	return result



def get_transform(obj):
	transform = obj.get('transform', '')
	result = {}
	pattern = r'\b\w+\([^)]+'#\)'
	for x in re.findall(pattern, transform):
		key, value = x.split('(')
		result[key] = rex.findnumbers(value)
	return result

def set_transform(obj, **kwargs): #Only translate, scale, rotate, skewX, skewY and matrix must be in kwargs
	transform = get_transform(obj)
	for k in kwargs:
		transform[k] = kwargs[k]
	obj.set('transform', ' '.join([f"{k}({' '.join(str(x) for x in transform[k])})" for k in transform]))

#def find_id(self, id):
#	return find(self, lambda x: 'id' in x.attrib and x.attrib['id'] == id)

def islayer(self): return self.tag == '{http://www.w3.org/2000/svg}g' and self.get('{http://www.inkscape.org/namespaces/inkscape}groupmode', None) == 'layer'

def layer_of(self):
	return find_ancestor(self, lambda x: islayer(x))

def ishidden(self):
	return find_ancestor(self, lambda x: islayer(x) and get_style_property(x, 'display') == 'none') != None

def clip_image(root, obj, img_path, abspath=False, mode='slice', align=0.5):
	parent = obj.getparent()
	#print(img_path)
	try:
		img = Image.open(img_path)
	except:
		return print('Image not found')
	if abspath:
		img_path = 'file:///' + img_path
	else:
		pass
		#img_path = os.path.basename(img_path)
	img_path = img_path.replace(os.sep, '/')

	img_rect = npx.fit_rect(bbox(obj), np.array(img.size), mode)

	obj_index = list(parent).index(obj)
	print(img_path)
	img = image(*img_rect.min, *img_rect.size, img_path)
	parent.insert(obj_index, img)
	clip(img, obj)

def clip(obj, mask):
	root = obj.getroottree().getroot()
	clip_path_id = 'clipPath' + str(int(uuid.uuid4()))
	obj.set('clip-path', f'url(#{clip_path_id})')
	"""defs = root.find('.//{http://www.w3.org/2000/svg}defs')
	if defs is None:
		defs = etree.SubElement(root, '{http://www.w3.org/2000/svg}defs')"""
	defs = get_or_create(root, '{http://www.w3.org/2000/svg}defs')
	clip_path = etree.SubElement(defs, '{http://www.w3.org/2000/svg}clipPath', attrib={'clipPathUnits': 'userSpaceOnUse', 'id': clip_path_id})
	clip_path.append(mask)







#TEXT

def tspans(text): return text.findall('.//{http://www.w3.org/2000/svg}tspan')

def lines(text): return [x for x in tspans(text) if x.get('{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}role', None) == 'line']





from io import BytesIO
import base64
from pyx.osx import readb

def embed_images(svg_tree, svg_folder):
	for image_element in svg_tree.findall(".//{http://www.w3.org/2000/svg}image"):
		href = image_element.get("{http://www.w3.org/1999/xlink}href")
		if href and not href.startswith("data:"):
			image_path = os.path.join(svg_folder, href).replace('%20', ' ')
			#print(image_path)
			if os.path.exists(image_path):
				image_data = readb(image_path)
				encoded_image = base64.b64encode(image_data).decode("utf-8")
				image_element.set("{http://www.w3.org/1999/xlink}href", f"data:image/png;base64,{encoded_image}")
	return svg_tree





def circle_to_svg(obj, **kwargs): return lxmlx.element("circle", cx=obj.center[0], cy=obj.center[1], r=obj.radius, **kwargs)
def ellipse_to_svg(obj, **kwargs): return lxmlx.element("ellipse", cx=obj.center[0], cy=obj.center[1], rx=obj.extents[0], ry=obj.extents[1], **kwargs)
def polygon_to_svg(obj, **kwargs): return lxmlx.element("polygon", points=" ".join(f"{x[0]},{x[1]}" for x in obj), **kwargs)
def polyline_to_svg(obj, **kwargs): return lxmlx.element("polyline", points=" ".join(f"{x[0]},{x[1]}" for x in obj), **kwargs)
def rect_to_svg(obj, **kwargs): return lxmlx.element("rect", x=obj.min[0], y=obj.min[1], width=obj.size[0], height=obj.size[1], **kwargs)
def line_to_svg(obj, **kwargs): return lxmlx.element("line", x1=obj[0][0], y1=obj[0][1], x2=obj[1][0], y2=obj[1][1], **kwargs)
def arc_to_svg(obj, **kwargs): return lxmlx.element("path", d=obj.d, **kwargs)
def image_to_svg(obj, **kwargs):
	return lxmlx.element("{http://www.w3.org/2000/svg}image", x=obj.min[0], y=obj.min[1], width=obj.size[0], height=obj.size[1], **{'{http://www.w3.org/1999/xlink}href': obj.filepath}, **kwargs)

def group_to_svg(obj, **kwargs): return lxmlx.element("g", children=[x.to_svg() for x in obj], **kwargs)

def href(image): return image.get('{http://www.w3.org/1999/xlink}href', None)

def capsule(center, size, **kwargs):
	r = min(*size) * 0.5
	return rect(*(np.array(center) - size * 0.5), *size, rx=str(r), ry=str(r), **kwargs)

NSMAP = {None: "http://www.w3.org/2000/svg"}
def svg(width, height): return etree.Element("svg", width=str(width), height=str(height), nsmap=NSMAP)

def tspan_to_svg(obj, **kwargs): #In SVG, the y attribute of a <text> element refers to the baseline of the text
	return  lxmlx.element('tspan', text=obj.inner_text, x=obj.aabb.min[0], y=obj.aabb.max[1], **{"font-family": osx.filename(obj.font), "font-size": obj.font_size}, **kwargs)

def text_to_svg(obj, **kwargs): #In SVG, the y attribute of a <text> element refers to the baseline of the text
	#g = lxmlx.element('text', x=obj.aabb.min[0], y=obj.aabb.max[1], **{"font-family": osx.filename(obj.font), "font-size": obj.font_size}, **kwargs)
	g = lxmlx.element('text', **{"font-family": osx.filename(obj.font), "font-size": obj.font_size}, **kwargs)
	for x in obj.lines:
		g.append(x.to_svg())
	#print(g)
	return g




def transform_to_svg(obj):	#order matters
	#return f'translate({obj.position[0]} {obj.position[1]}) scale({obj.scale[0]} {obj.scale[1]}) rotate({math.degrees(obj.rotation)})'
	return (
		f'translate({obj.position[0]} {obj.position[1]}) '
		f'rotate({math.degrees(obj.rotation)}) '
		f'scale({obj.scale[0]} {obj.scale[1]})'
	)

Node2D.to_svg = lambda self: transform_to_svg(self)

def get_attrib(obj):
	attrib = getattr(obj, "attrib", {})
	for k in ['fill', 'stroke']:
		if k in attrib:
			if isinstance(attrib[k], Color):
				attrib[k + '-opacity'] = attrib[k].a
				attrib[k] = attrib[k].hex[:-2]
	for k in attrib:
		if hasattr(attrib[k], "to_svg"):
			attrib[k] = attrib[k].to_svg()
	return attrib

geo.polybezier.to_svg = lambda self: path(self.d(), **get_attrib(self))
geo.circle.to_svg = lambda self: circle_to_svg(self, **get_attrib(self))
geo.ellipse.to_svg = lambda self: ellipse_to_svg(self, **get_attrib(self))
geo.line.to_svg = lambda self: line_to_svg(self, **get_attrib(self))
geo.polyline.to_svg = lambda self: polygon_to_svg(self, **get_attrib(self)) if self.closed else polyline_to_svg(self, **get_attrib(self))
geo.rect.to_svg = lambda self: rect_to_svg(self, **get_attrib(self))
geo.group.to_svg = lambda self: group_to_svg(self, **get_attrib(self))
geo.arc.to_svg = lambda self: arc_to_svg(self, **get_attrib(self))
PILx.tspan.to_svg = lambda self: tspan_to_svg(self, **get_attrib(self))
PILx.text.to_svg = lambda self: text_to_svg(self, **get_attrib(self))
PILx.image.to_svg = lambda self: image_to_svg(self, **get_attrib(self))


def draw(obj):
	size = obj.aabb.size
	obj.aabb = geo.rect(np.zeros(2), size)
	result = svg(*size)
	result.append(obj.to_svg())
	return result

def save_svg(obj, path='drawing.svg', overwrite=True):
	if not path.lower().endswith('.svg'):
		path += '.svg'

	if not overwrite:
		path = osx.to_distinct(path)

	lxmlx.save(draw(obj), path)

	return path

geo.group.save_svg = lambda self, path='drawing.svg', overwrite=True: save_svg(self, path, overwrite)

def load_svg(filepath, asnode=False):
	root = etree.parse(filepath).getroot()
	root = from_svg(root, asnode)
	#set_pivots(root)
	return root




