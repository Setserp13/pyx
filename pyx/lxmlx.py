import ast
from lxml import etree
import pyx.osx as osx
from typing import Dict

def element(tag, text='', tail='', parent=None, children=[], **kwargs): #create a xml element
	#result = etree.Element(tag, **kwargs)
	result = etree.Element(tag, attrib={k: str(v) for k, v in kwargs.items()})
	if parent is not None:
		parent.append(result)
	result.text = text
	result.tail = tail
	result.extend(children)
	return result

def create_child(parent, tag, **kwargs):
	result = etree.Element(tag, **kwargs)
	parent.append(result)
	return result

def get_or_create(parent, tag, **kwargs):
	result = parent.find('.//' + tag)
	if result is None:
		result = etree.SubElement(parent, tag, **kwargs)
	return result

def find(self, match, iter=etree._Element.iter): #find descendant by default
	for x in iter(self):
		if match(x):
			return x
	return None

def findall(self, match, iter=etree._Element.iter): return [x for x in iter(self) if match(x)] #find descendants by default

def fetchall(self, **where): return findall(self, lambda x: all(k in x.attrib and x.attrib[k] == where[k] for k in where))

def fetchone(self, **where): return find(self, lambda x: all(k in x.attrib and x.attrib[k] == where[k] for k in where))

def find_ancestor(self, match, dflt_value=None):
	parent = self.getparent()
	while parent is not None:
		#print(parent.tag)
		if match(parent):
			return parent
		parent = parent.getparent()
	return dflt_value

def get(obj, type, *keys): return [type(obj.get(x)) for x in keys]

#def get(obj, *keys): return [ast.literal_eval(obj.get(x)) for x in keys]

#def get(obj, **kwargs): return [kwargs[k](obj.get(k)) for k in kwargs]

def set(obj, **kwargs):
	for k in kwargs:
		obj.set(k, str(kwargs[k]))

#def localname(tag): return tag.split('}')[-1] if '}' in tag else tag

def find_tags(root, *tags): return findall(root, lambda x: etree.QName(x.tag).localname in tags)	#localname(x.tag) in tags)
#def find_tags(root, *tags): return [x for x in root.iter('*') if localname(x.tag) in tags]



def leaves(element, current_path=''): #returns all leaves and their global names
	result = []
	#tag = element.tag
	tag = etree.QName(element).localname
	path = f'{current_path}/{tag}' if current_path else tag
	if len(element) == 0:  #element is a leaf
		#print(path)
		result.append({'element': element, 'path': path})
	else:
		for child in element:
			result += leaves(child, path)
	return result

def get_namespaces(root: etree._Element) -> Dict[str, str]:
	#Extract namespaces from an XML document. If a default namespace exists, it is mapped to 'ns'.
	nsmap = {}
	for k, v in root.nsmap.items():
		if k is None:
			nsmap["ns"] = v
		else:
			nsmap[k] = v
	return nsmap




def save(obj, path): osx.write(path, etree.tostring(obj, pretty_print=True, encoding='unicode'))
