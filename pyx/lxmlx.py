from lxml import etree

def create_child(parent, tag, **kwargs):
	result = etree.Element(tag, **kwargs)
	parent.append(result)
	return result

def find(self, match, iter=etree._Element.iter): #find descendant by default
	for x in iter(self):
		if match(x):
			return x
	return None

def find_ancestor(self, match, dflt_value=None):
	parent = self.getparent()
	while parent is not None:
		#print(parent.tag)
		if match(parent):
			return parent
		parent = parent.getparent()
	return dflt_value

def get(obj, type, *keys): return [type(obj.get(x)) for x in keys]

def set(obj, **kwargs):
	for k in kwargs:
		obj.set(k, kwargs[k])

def localname(tag): return tag.split('}')[-1] if '}' in tag else tag

def find_tags(root, *tags): return [x for x in root.iter('*') if localname(x.tag) in tags]
