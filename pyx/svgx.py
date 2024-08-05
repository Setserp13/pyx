import xml.etree.ElementTree as ET
from functools import reduce
from pyx.lsvg import *
from lxml import etree

from pyx.mat.rect import *
from pyx.mat.vector import *
from PIL import Image
import uuid
import os
from pyx.osx import wd
import tkinter as tk
from tkinter import filedialog
import svgpathtools
from svgpathtools import parse_path
import svgutils.transform as sg
import re
from pyx.array_utility import items
from pyx.lxmlx import find_ancestor, find, localname

def vertices(obj):
	if localname(obj.tag) == 'rect':
		return [list(x) for x in corners(rect_bbox(obj))]
	elif localname(obj.tag) == 'path':
		return [[float(y) for y in x.replace(' ', '').split(',')] for x in obj.get('d', '').split(' ')[1:-1]]

def path_bbox(obj):
	path = parse_path(obj.get('d', None))
	bbox = path.bbox()
	sc = get_scale(obj)
	return Rect.MinMax(Vector(bbox[0] * sc[0], bbox[2] * sc[1]), Vector(bbox[1] * sc[0], bbox[3] * sc[1]))

def get_bbox(obj): return {'circle': circle_bbox, 'ellipse': ellipse_bbox, 'image': rect_bbox, 'path': path_bbox, 'rect': rect_bbox}[localname(obj.tag)](obj)

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
	return [x for i, x in enumerate(rects) if len(list(filter(lambda y: y.containsRect(x), remove_at(rects, i)))) == 0]


def strpdict(obj, sep=[';', ':']):
	result = {}
	items = [] if obj == '' else obj.split(sep[0])
	for item in items:
		key, value = item.split(sep[1])
		result[key.strip()] = value.strip()
	return result

def strfdict(obj, sep=[';', ':']): return sep[0].join([f'{k}{sep[1]}{obj[k]}' for k in obj])


def get_style(element): return strpdict(element.get('style', ''))

def get_style_property(element, property):
	try:
		return get_style(element)[property]
	except:
		return None

def get_style_properties(element, *properties): return items(get_style(element), properties)

def set_style(element, **kwargs):
	style = get_style(element)
	for k in kwargs:
		style[k] = kwargs[k]
	element.set("style", strfdict(style))

def extract_numbers(text):
	numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
	return [float(x) if '.' in x else int(x) for x in numbers]

def get_transform(obj):
	transform = obj.get('transform', '')
	result = {}
	pattern = r'\b\w+\([^)]+'#\)'
	for x in re.findall(pattern, transform):
		key, value = x.split('(')
		result[key] = extract_numbers(value)
	return result

def set_transform(obj, **kwargs): #Only translate, scale, rotate, skewX, skewY and matrix must be in kwargs
	transform = get_transform(obj)
	for k in kwargs:
		transform[k] = kwargs[k]
	obj.set('transform', ' '.join([f"{k}({' '.join(str(x) for x in transform[k])})" for k in transform]))









def find_id(self, id):
	return find(self, lambda x: 'id' in x.attrib and x.attrib['id'] == id)

def islayer(self): return self.tag == '{http://www.w3.org/2000/svg}g' and self.get('{http://www.inkscape.org/namespaces/inkscape}groupmode', None) == 'layer'

def layer_of(self):
	return find_ancestor(self, lambda x: islayer(x))

def ishidden(self):
	return find_ancestor(self, lambda x: islayer(x) and get_style_property(x, 'display') == 'none') != None

from pyx.lsvg import clip

def clip_image(root, obj, img_path):
	parent = obj.getparent()
	#print(img_path)
	try:
		img = Image.open(img_path)
	except:
		return print('Image not found')
	img_path = os.path.basename(img_path)
	scale_factor = max(*Vector.divide(get_bbox(obj).size, img.size))
	img_size = Vector(*img.size) * scale_factor
	obj_index = list(parent).index(obj)
	img = image(parent, Rect2(*get(obj, float, 'x', 'y'), *img_size), img_path)
	parent.insert(obj_index, img)
	"""image = etree.Element('{http://www.w3.org/2000/svg}image', attrib={
		'x': obj.get('x'),
		'y': obj.get('y'),
		'width': str(img_size[0]),
		'height': str(img_size[1]),
		'{http://www.w3.org/1999/xlink}href': img_path
	})
	parent.insert(obj_index, image)"""
	clip(img, obj)


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




#pip install pipwin
#pipwin install cairocffi
#pip install cairosvg

import cairosvg

from lxml import etree

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

def svg_to_png(tree, output_png, rect, dpi=10):
    root = tree.getroot()	
    print(rect.to_tuple())
    if rect.size[0] < 1 or rect.size[1] < 1: return #'CUZ NO SIZE CAN BE LESSER THAN 1

    # Modify the SVG content to include a viewBox attribute
    root.attrib['viewBox'] = f"{rect.min[0]} {rect.min[1]} {rect.size[0]} {rect.size[1]}"

    # Create an in-memory file-like object
    svg_buffer = BytesIO()
    tree.write(svg_buffer, encoding='utf-8', xml_declaration=True)

    # Use the in-memory SVG content directly, no need for a temporary file
    svg_buffer.seek(0)

    cairosvg.svg2png(file_obj=svg_buffer, write_to=output_png, output_width=rect.size[0], output_height=rect.size[1], dpi=dpi)

