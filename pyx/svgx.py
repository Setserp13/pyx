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
from pyx.collectionsx import List, to_str
from pyx.lxmlx import *	#localname
import numpy as np
import pyx.numpyx as npx
import pyx.numpyx_geo as geo
import pyx.lxmlx as lxmlx
from pyx.mat.transform import Node2D, Transform
from pyx.gamepy.color import Color

def replace_all(root, replacements, exact_match=False):
	"""
	replacements: dict {old_text: new_text}
	"""
	for el in root.iter():
		try:
			tag = etree.QName(el).localname
		except: continue
		if tag in ("text", "tspan"):
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
	closed = False
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
			closed = True
			v.append(np.array(points[0]))

		v = geo.polyline(v)
		k = 2 if cmd.upper() in 'QT' else 3 if cmd.upper() in 'CS' else 1
		if cmd.islower():	#relative to the last point of the previous command
			axis = 0 if cmd.upper() == 'H' else 1 if cmd.upper() == 'V' else None
			#print(cmd, k, v)
			for i in range(0, len(v), k):
				if axis is None:
					v[i:i + k] = v[i:i + k] + pos
				else:
					v[i:i + k, axis] = v[i:i + k, axis] + pos[axis]
				pos = v[i + k - 1]
			#print(v)
		
		if cmd.upper() in 'MLHV':
			endpoints.extend([i + len(points) for i in range(len(v))])
		elif cmd.upper() in 'QCTS':
			start = len(points)
			for i in range(0, len(v), k):
				endpoints.append(start + i + k - 1)
		points.extend(v)
		pos = v[-1]
	#print(points)
	#miss Aa
	return geo.polybezier(points, endpoints=endpoints, closed=closed)

def transform_from_svg(obj):
	transform = get_transform(obj)
	if 'matrix' in transform:
		a, b, c, d, e, f = transform['matrix']
		M = np.array([[a, c, e], [b, d, f], [0,0,1]])
		return Node2D.from_matrix(M)
	else:
		result = Node2D()
		if 'translate' in transform:
			result.position = np.array(transform['translate'])
		if 'rotate' in transform:
			result.rotation = math.radians(float(transform['rotate'][0]))
		if 'scale' in transform:
			result.scale = np.array(transform['scale'])
			if len(result.scale) < 2:
				result.scale = np.append(result.scale, result.scale[0])
		return result

def from_svg(obj):
	result = None
	tag = etree.QName(obj).localname
	#print(tag)
	match tag:
		case 'circle': result = circle_from_svg(obj)
		case 'ellipse': result = ellipse_from_svg(obj)
		case 'g' | 'svg':
			children = [from_svg(x) for x in obj]
			result = geo.group([x for x in children if not x is None])
		case 'line': result = line_from_svg(obj)
		case 'polybezier': result = bezier_from_svg(obj)
		case 'polyline': result = polyline_from_svg(obj)
		case 'polygon': result = polygon_from_svg(obj)
		case 'rect': result = rect_from_svg(obj)
		case 'text': pass

	if result is None:
		return result
	result.set(transform=transform_from_svg(obj))
	if isinstance(result, geo.group):
		for x in result:
			result.attrib['transform'].append(x.attrib['transform'])
	
	"""result.id = obj.get('id')
	style = get_style(obj)
	fill = obj.get('fill', None)
	if not fill is None:
		style['fill'] = fill
	stroke = obj.get('stroke', None)
	if not stroke is None:
		style['stroke'] = stroke
	if not 'fill' in style:
		style['fill'] = 'black'
	if not 'stroke' in style:
		style['stroke'] = 'none'
	for k in ['fill', 'stroke']:
		style[k] = None if style[k] == 'none' else Color(style[k])
	for k in ['fill-opacity', 'stroke-opacity', 'stroke-width']:
		if k in style:
			style[k] = float(style[k])
		else:
			style[k] = 1.0
	result.style = style"""

	node = result
	node.set(**obj.attrib)
	style = get_style(obj)
	#print(style)
	for k in ('fill', 'stroke'):
		if k in style:
			node.attrib[k] = style[k]
		v = Color([0., 0., 0., 1.])
		if k in node.attrib:
			v = Color.parse(node.attrib[k])
		a = f'{k}-opacity'
		if a in style:
			node.attrib[a] = style[a]
		if a in node.attrib:
			v[3] = float(node.attrib[a])
		#print(v)
		node.attrib[k] = v

	desc = lxmlx.find(obj, lambda x: etree.QName(x).localname == "desc", iter=lambda e: e)
	#print(desc)
	result.desc = {} if desc is None else rex.strpdict(desc.text, sep=[';', '='])
	return result

def from_svg2(obj):
	
	shape = None
	tag = etree.QName(obj).localname

	node = transform_from_svg(obj)

	#print(tag)
	match tag:
		case 'circle': shape = circle_from_svg(obj)
		case 'ellipse': shape = ellipse_from_svg(obj)
		case 'g' | 'svg':
			children = [from_svg(x) for x in obj]
			node.extend(children)
		case 'line': shape = line_from_svg(obj)
		case 'polybezier': shape = bezier_from_svg(obj)
		case 'polyline': shape = polyline_from_svg(obj)
		case 'polygon': shape = polygon_from_svg(obj)
		case 'rect': shape = rect_from_svg(obj)
		case 'text': pass

	node.shape = shape
	node.set(**obj.attrib)
	style = get_style(obj)
	#print(style)
	for k in ('fill', 'stroke'):
		if k in style:
			node.attrib[k] = style[k]
		v = Color([0., 0., 0., 1.])
		if k in node.attrib:
			v = Color.parse(node.attrib[k])
		a = f'{k}-opacity'
		if a in style:
			node.attrib[a] = style[a]
		if a in node.attrib:
			v[3] = float(node.attrib[a])
		#print(v)
		node.attrib[k] = v

	desc = lxmlx.find(obj, lambda x: etree.QName(x).localname == "desc", iter=lambda e: e)
	#print(desc)
	node.desc = {} if desc is None else rex.strpdict(desc.text, sep=[';', '='])
	return node





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
def arc_to_svg(obj, **kwargs): return lxmlx.element("path", d=obj.d(), **kwargs)



def group_to_svg(obj, **kwargs): return lxmlx.element("g", children=[x.to_svg() for x in obj], **kwargs)

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
	return f'translate({obj.position[0]} {obj.position[1]}) scale({obj.scale[0]} {obj.scale[1]}) rotate({math.degrees(obj.rotation)})'

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
geo.tspan.to_svg = lambda self: tspan_to_svg(self, **get_attrib(self))
geo.text.to_svg = lambda self: text_to_svg(self, **get_attrib(self))



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






