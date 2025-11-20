import xml.etree.ElementTree as ET
from lxml import etree
from functools import reduce
import pyx.rex as rex
from PIL import Image
import uuid
import os
import svgpathtools
from svgpathtools import parse_path
import svgutils.transform as sg
import re
from pyx.collectionsx import List, to_str
from pyx.lxmlx import *	#localname
import numpy as np
import pyx.numpyx as npx
import pyx.numpyx_geo as geo
import pyx.mat.bezier as bezier

"""def vertices(obj):
	if localname(obj.tag) == 'rect':
		return [list(x) for x in npx.corners(rect_bbox(obj))]
	elif localname(obj.tag) == 'path':
		return [[float(y) for y in x.replace(' ', '').split(',')] for x in obj.get('d', '').split(' ')[1:-1]]"""


#PATH
def parse_commands(s):
	p = re.compile(r'([A-Za-z])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
	r, c, a = [], None, []
	for cmd, num in p.findall(s):
		if cmd:
			if c: r.append((c,*map(float,a))); a=[]
			c = cmd
		elif num:
			a.append(num)
	if c: r.append((c,*map(float,a)))
	return r

def bezier_from_svg(obj):
	d = obj.get('d')
	commands = parse_commands(d)
	points = []
	endpoints = []
	pos = np.zeros(2)
	for t in commands:
		cmd, *args = t
		#print(args)
		v = []
		if cmd.upper() in 'MLQCTS':
			v = [np.array(x) for x in List.batch(args, 2)]
			if cmd.upper() in 'TS':
				v.insert(0, pos + (pos - points[-2]))
		elif cmd in 'Hh':
			v = [np.array([x, pos[1]]) for x in args]
		elif cmd in 'Vv':
			v = [np.array([pos[0], x]) for x in args]
		elif cmd in 'Zz':
			v.append(np.array(points[0]))

		v = geo.polyline(v)
		if cmd.islower():	#relative to the last point of the previous command
			k = 2 if cmd.upper() in 'QT' else 3 if cmd.upper() in 'CS' else 1
			for i in range(0, len(v), k):
				v[i:i + k] = v[i:i + k] + pos
				pos = v[i + k - 1]

		if cmd.upper() in 'MLHV':
			endpoints.extend([i + len(points) for i in range(len(v))])
		elif cmd.upper() in 'QCTS':
			k = 2 if cmd.upper() in 'QT' else 3 if cmd.upper() in 'CS' else 1
			start = len(points)
			for i in range(0, len(v), k):
				endpoints.append(start + i + k - 1)
		points.extend(v)
		pos = v[-1]
	#print(points)
	#miss Aa
	return bezier.path(points, endpoints=endpoints)


def circle_from_svg(obj): return geo.circle(get(obj, float, 'cx', 'cy'), *get(obj, float, 'r'))
def rect_from_svg(obj): return npx.rect2(*get(obj, float, 'x', 'y', 'width', 'height'))
def line_from_svg(obj): return geo.line([get(obj, float, 'x1', 'y1'), get(obj, float, 'x2', 'y2')])
def parse_points2(s):
	matches = re.findall(r'(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)', s)
	return np.array(matches, dtype=float)
def polygon_from_svg(obj):  return geo.polyline(parse_points2(obj.get('points')), closed=True)
def polyline_from_svg(obj): return geo.polyline(parse_points2(obj.get('points')), closed=False)

def circle_bbox(obj): return circle_from_svg(obj).aabb
def ellipse_bbox(obj):
	cx, cy, rx, ry = get(obj, float, 'cx', 'cy', 'rx', 'ry')
	return npx.rect2(cx - rx, cy - ry, rx * 2, ry * 2)
def rect_bbox(obj): return rect_from_svg(obj)	#npx.rect2(*get(obj, float, 'x', 'y', 'width', 'height'))
def line_bbox(obj):	return npx.aabb(*line_from_svg(obj))
def path_bbox(obj):
	path = parse_path(obj.get('d', None))
	bbox = path.bbox()
	sc = get_scale(obj)
	return npx.rect.min_max(np.array([bbox[0] * sc[0], bbox[2] * sc[1]]), np.array([bbox[1] * sc[0], bbox[3] * sc[1]]))
def polyline_bbox(obj):	return npx.aabb(polyline_from_svg(obj))
def g_bbox(group):
	bboxes = []
	for elem in group.iterdescendants():	#deep search
		"""try:
			b = bbox(elem)
		except:
			#print(elem)
			continue
		if b is None:
			continue"""
		b = bbox(elem)
		bboxes.append(b)
	return npx.aabb(bboxes) if len(bboxes) > 0 else None
#
def bbox(obj): return {	#Se o grupo tiver transformações (transform="translate(...)" etc), isso não será aplicado
		'circle': circle_bbox,
		'ellipse': ellipse_bbox,
		'image': rect_bbox,
		'line': line_bbox,
		'path': path_bbox,
		'polyline': polyline_bbox,
		'polygon': polyline_bbox,
		'rect': rect_bbox,
		'g': g_bbox
	}[localname(obj.tag)](obj)

def get_bboxes(objs): return [get_bbox(x) for x in objs]

def find_bboxes(root, tag, get_bbox):
	ns = {'svg': 'http://www.w3.org/2000/svg'}
	return [get_bbox(x) for x in root.findall(f'.//svg:{tag}', namespaces=ns)]


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

def get_style_properties(element, *properties): return List.items(get_style(element), properties)

def set_style(element, **kwargs):
	style = get_style(element)
	for k in kwargs:
		style[k] = kwargs[k]
	element.set("style", rex.strfdict(style))


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

def clip_image(root, obj, img_path):
	parent = obj.getparent()
	#print(img_path)
	try:
		img = Image.open(img_path)
	except:
		return print('Image not found')
	img_path = os.path.basename(img_path)
	scale_factor = max(np.array(get_bbox(obj).size) / np.array(img.size))
	img_size = np.array(img.size) * scale_factor
	obj_index = list(parent).index(obj)
	img = image(parent, npx.rect2(*get(obj, float, 'x', 'y'), *img_size), img_path)
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

def line_height(text):
	font_size = float(get_style_property(text, 'font-size').replace('px', ''))
	line_height = float(get_style_property(text, 'line-height'))
	#print([font_size, line_height])
	return font_size * line_height

def add_line(text, amount=1):
	line_count = len(lines(text))
	lh = line_height(text)
	for i in range(line_count, line_count + amount):
		text.append(etree.Element('{http://www.w3.org/2000/svg}tspan', attrib={
			'{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}role': 'line',
			'style': strfdict(get_style_properties(text, 'text-align', 'text-anchor', 'stroke-width')),
			'x': text.get('x'),
			'y': str(float(text.get('y')) + lh * i),
			'id': f"{text.get('id')}{i}"
		}))

def set_lines(text, text_lines):
	lns = lines(text)
	for x in lns[len(text_lines):len(lns)]:
		text.remove(x)
	add_line(text, len(text_lines) - len(lns))
	for i, x in enumerate(lines(text)):
		x.text = text_lines[i]

def set_font_size(text, value): set_style_property(text, 'font-size', f'{value}px')

def set_leading(text, value):
	font_size = float(get_style_property(text, 'font-size').replace('px', ''))
	set_style_property(text, 'line-height', str(1.0 + value / font_size))



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

def circle(cx, cy, r, **kwargs): return etree.Element("circle", cx=str(cx), cy=str(cy), r=str(r), **to_str(kwargs))

def ellipse(cx, cy, rx, ry, **kwargs): return etree.Element("ellipse", cx=str(cx), cy=str(cy), rx=str(rx), ry=str(ry), **to_str(kwargs))

def polygon(*points, **kwargs): return etree.Element("polygon", points=" ".join(f"{x[0]},{x[1]}" for x in points), **to_str(kwargs))

def polyline(*points, **kwargs): return etree.Element("polyline", points=" ".join(f"{x[0]},{x[1]}" for x in points), **to_str(kwargs))

def rect(x, y, width, height, **kwargs): return etree.Element("rect", x=str(x), y=str(y), width=str(width), height=str(height), **to_str(kwargs))

def line(x1, y1, x2, y2, **kwargs): return etree.Element("line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), **to_str(kwargs))

def g(*args, **kwargs):
	result = etree.Element("g", **to_str(kwargs))
	for x in args:
		result.append(x)
	return result

def path(d, **kwargs): return etree.Element("path", d=d, **to_str(kwargs))

def href(image): return image.get('{http://www.w3.org/1999/xlink}href', None)

#def rect_attrib(rct): return { 'x': str(rct.min[0]), 'y': str(rct.min[1]), 'width': str(rct.size[0]), 'height': str(rct.size[1]) }

def image(x, y, width, height, href):
	return etree.Element("{http://www.w3.org/2000/svg}image", x=str(x), y=str(y), width=str(width), height=str(height), **{'{http://www.w3.org/1999/xlink}href': href})

def capsule(center, size, **kwargs):
	r = min(*size) * 0.5
	return rect(*(np.array(center) - size * 0.5), *size, rx=str(r), ry=str(r), **kwargs)

NSMAP = {None: "http://www.w3.org/2000/svg"}
def svg(width, height): return etree.Element("svg", width=str(width), height=str(height), nsmap=NSMAP)

def polygon_arcs(vertices, closed=True, mask=None, radius=0.5, **kwargs):
	edges = geo.polyline.edges(vertices, closed=closed)
	if mask is None:
		mask = [True] * len(edges)
	d = f"M {vertices[0][0]} {vertices[0][1]}"
	for i, edge in enumerate(edges):
		r = edge.length * radius
		x, y = edge[1]
		d += f" A {r} {r} 0 0 1 {x} {y}" if mask[i] else f" L {x} {y}"
	return path(d=d, **kwargs)
