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

def findall(self, match, iter=etree._Element.iter): return [x for x in iter(self) if match(x)] #find descendants by default

def fetchall(df, **where): return findall(self, lambda x: all(k in x.attrib and x.attrib[k] == where[k] for k in where))

def fetchone(df, **where): return find(self, lambda x: all(k in x.attrib and x.attrib[k] == where[k] for k in where))

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
		obj.set(k, str(kwargs[k]))

def localname(tag): return tag.split('}')[-1] if '}' in tag else tag

def find_tags(root, *tags): return findall(root, lambda x: localname(x.tag) in tags)
#def find_tags(root, *tags): return [x for x in root.iter('*') if localname(x.tag) in tags]

def leaf_paths(element, current_path=''): #returns all root-to-leaf paths
	paths = []
	path = f'{current_path}/{element.tag}' if current_path else element.tag
	if len(element) == 0:  #is a leaf element
		#print(path)
		paths.append(path)
	else:
		for child in element:
			paths += leaf_paths(child, path)
	return paths
