from pyx.array_utility import index_of
from pyx.generic.generic import Element

class Node(Element):
	def __init__(self, parent=None, children=None, **kwargs):
		for k in kwargs:
			setattr(self, k, kwargs[k])
		self.parent = None #Declare self.parent
		self.set_parent(parent) #Node
		self.children = []	#init children
		self.extend([] if children is None else children)

	def set_parent(self, parent):
		if self.parent is not None:
			self.parent.children.remove(self)
	
		self.parent = parent
	
		if parent is not None:
			parent.children.append(self)
	
	@property
	def sibling_index(self):
		if self.parent != None:
			return index_of(self.parent.children, self)
		return 0

	def siblings(self):
		result = []
		if self.parent != None:
			for x in self.parent.children:
				if x != self:
					result.append(x)
		return result

	def append(self, value):
		old_parent = getattr(value, "parent", None)
		if old_parent != self:
			if old_parent != None:
				old_parent.remove(value)
			self.children.append(value)
			value.parent = self
		#value.setParent(self)	#append a child

	def extend(self, items):
		for x in items:
			self.append(x)
	
	def remove(self, value):
		if value in self.children:
			self.children.remove(value)
			self.parent = None

	def isroot(self): return self.parent == None

	def isleaf(self): return self.degree() == 0
	
	def neighbors(self):
		if self.parent == None:
			return self.children
		return [self.parent] + self.children

	def ancestors(self):
		result = []
		if self.parent != None:
			result.append(self.parent)
			result.extend(self.parent.ancestors())
		return result

	def descendants(self):
		result = []
		for child in self.children:
			result.append(child)
			result.extend(child.descendants())
		return result
	
	def size(self):
		result = 1
		for child in self.children:
			result = result + child.size()
		return result

	
	def nodesAtDistance(self, distance):
		if distance == 0:
			return [self]
		result = []
		for child in self.children:
			result = result + child.nodesAtDistance(distance - 1)
		return result

	@property
	def degree(self): return len(self.children)

	def leaves(self):
		result = []
		if self.isleaf():
			result.append(self)
		else:
			for child in self.children:
				result = result + child.leaves()
		return result	

	@property
	def breadth(self):
		if self.isleaf():
			return 1
		else:
			result = 0
			for child in self.children:
				result = result + child.breadth()
			return result

	def root(self):
		if self.isroot():
			return self
		return self.parent.root()

	def level(self):
		if self.isroot():
			return 0
		return self.parent.level() + 1

