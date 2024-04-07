import xml.etree.ElementTree as ET
from functools import reduce

from pyx.mat.rect import *
from pyx.mat.vector import *

import os
from pyx.osx import wd
import tkinter as tk
from tkinter import filedialog
import svgpathtools
from svgpathtools import parse_path
import svgutils.transform as sg
import re

def get_floats(obj, *args): return [float(obj.get(x, None)) for x in args]

def circle_bbox(obj): return BBox.circle(*get_floats(obj, 'cx', 'cy', 'r'))

def ellipse_bbox(obj): return BBox.ellipse(*get_floats(obj, 'cx', 'cy', 'rx', 'ry'))

def rect_bbox(obj): return Rect2(*get_floats(obj, 'x', 'y', 'width', 'height')) #this also works with images

def vertices(obj):
	if localname(obj.tag) == 'rect':
		return [list(x) for x in corners(rect_bbox(obj))]
	elif localname(obj.tag) == 'path':
		return [[float(y) for y in x.replace(' ', '').split(',')] for x in obj.get('d', '').split(' ')[1:-1]]


def get_scale_from_transform(transform):
    if transform is None:
        return 1.0, 1.0  # Default scale factors for no transform

    match = re.search(r'scale\(([\d.]+),\s*([\d.]+)\)', transform)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r'scale\(([\d.]+)\)', transform)
    if match:
        return float(match.group(1)), float(match.group(1))
    return 1.0, 1.0

def get_scale(svg_element):
    transform_attr = svg_element.get('transform', None)
    
    return get_scale_from_transform(transform_attr)

def path_bbox(obj):
	path = parse_path(obj.get('d', None))
	bbox = path.bbox()
	sc = get_scale(obj)
	return Rect.MinMax(Vector(bbox[0] * sc[0], bbox[2] * sc[1]), Vector(bbox[1] * sc[0], bbox[3] * sc[1]))


def localname(tag): return tag.split('}')[-1] if '}' in tag else tag

def get_bbox(obj): return {'circle': circle_bbox, 'ellipse': ellipse_bbox, 'image': rect_bbox, 'path': path_bbox, 'rect': rect_bbox}[localname(obj.tag)](obj)

def get_bboxes(objs): return [get_bbox(x) for x in objs]

def find_bboxes(root, tag, get_bbox):
	ns = {'svg': 'http://www.w3.org/2000/svg'}
	return [get_bbox(x) for x in root.findall(f'.//svg:{tag}', namespaces=ns)]


#def find_tags(root, *tags): return [x for x in root.iter('*') if x.tag in ['{http://www.w3.org/2000/svg}' + f'{y}' for y in tags]]

def find_tags(root, *tags): return [x for x in root.iter('*') if localname(x.tag) in tags]

def find_layers(root):
	ns = {'svg': 'http://www.w3.org/2000/svg', 'inkscape': 'http://www.inkscape.org/namespaces/inkscape'}
	return [
		x for x in root.findall('.//svg:g', namespaces=ns) if x.get('{http://www.inkscape.org/namespaces/inkscape}groupmode', None) == 'layer'
	]

def remove_at(ls, i): return ls[:i] + ls[i+1:]

def root_rects(rects): #Rects that are not subrects of another one in the list
	return [x for i, x in enumerate(rects) if len(list(filter(lambda y: y.containsRect(x), remove_at(rects, i)))) == 0]


def get_style_property(element, property_name):
    style = element.get("style")
    if style is not None:
        styles = style.split(";")
        for s in styles:
            pair = s.split(":")
            if len(pair) == 2 and pair[0].strip() == property_name:
                return pair[1].strip()
    return None


def find_ancestor(self, match, dflt_value=None):
	parent = self.getparent()
	while parent is not None:
		#print(parent.tag)
		if match(parent):
			return parent
		parent = parent.getparent()
	return dflt_value

def islayer(self): return self.tag == '{http://www.w3.org/2000/svg}g' and self.get('{http://www.inkscape.org/namespaces/inkscape}groupmode', None) == 'layer'

def layer_of(self):
	return find_ancestor(self, lambda x: islayer(x))

def ishidden(self):
	return find_ancestor(self, lambda x: islayer(x) and get_style_property(x, 'display') == 'none') != None


#pip install pipwin
#pipwin install cairocffi
#pip install cairosvg

import cairosvg

import svgwrite
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

