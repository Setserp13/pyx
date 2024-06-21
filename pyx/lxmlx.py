from lxml import etree

def create_child(parent, tag, **kwargs):
	result = etree.Element(tag, **kwargs)
	parent.append(result)
	return result
