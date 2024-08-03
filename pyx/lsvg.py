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