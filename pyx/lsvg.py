import uuid
from lxml import etree

def clip(obj, mask):
	root = obj.getroottree().getroot()
	clip_path_id = 'clipPath' + str(int(uuid.uuid4()))
	obj.set('clip-path', f'url(#{clip_path_id})')
	defs = root.find('.//{http://www.w3.org/2000/svg}defs')
	if defs is None:
		defs = etree.SubElement(root, '{http://www.w3.org/2000/svg}defs')
	clip_path = etree.SubElement(defs, '{http://www.w3.org/2000/svg}clipPath', attrib={'clipPathUnits': 'userSpaceOnUse', 'id': clip_path_id})
	clip_path.append(mask)

def href(image): return image.get('{http://www.w3.org/1999/xlink}href', None)

def rect_attrib(rct): return { 'x': str(rct.min[0]), 'y': str(rct.min[1]), 'width': str(rct.size[0]), 'height': str(rct.size[1]) }

def rect(parent, rct):
	return etree.SubElement(parent, 'rect', **rect_attrib(rct), style="fill:none;stroke:black;stroke-width:1")

class bbox:
	def circle(cx, cy, r): return npx.rect2(cx - r, cy - r, r * 2, r * 2)
	def ellipse(cx, cy, rx, ry): return npx.rect2(cx - rx, cy - ry, rx * 2, ry * 2)

def circle_bbox(obj): return bbox.circle(*get(obj, float, 'cx', 'cy', 'r'))

def ellipse_bbox(obj): return bbox.ellipse(*get(obj, float, 'cx', 'cy', 'rx', 'ry'))

def rect_bbox(obj): return npx.rect2(*get(obj, float, 'x', 'y', 'width', 'height'))
