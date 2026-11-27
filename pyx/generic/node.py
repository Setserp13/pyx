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

	@property
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
			value.parent = None

	@property
	def isroot(self): return self.parent == None

	@property
	def isleaf(self): return self.degree == 0

	@property
	def neighbors(self):
		if self.parent == None:
			return self.children
		return [self.parent] + self.children

	@property
	def ancestors(self):
		result = []
		if self.parent != None:
			result.append(self.parent)
			result.extend(self.parent.ancestors)
		return result

	@property
	def descendants(self):
		result = []
		for child in self.children:
			result.append(child)
			result.extend(child.descendants)
		return result

	@property
	def size(self): return 1 + len(self.descendants)

	def nodesAtDistance(self, distance):
		if distance == 0:
			return [self]
		result = []
		for child in self.children:
			result = result + child.nodesAtDistance(distance - 1)
		return result

	@property
	def degree(self): return len(self.children)

	@property
	def leaves(self): return [x for x in [self, *self.descendants] if x.isleaf]

	@property
	def breadth(self): return len(self.leaves)

	@property
	def root(self): return self if self.isroot else self.parent.root

	@property
	def level(self): return 0 if self.isroot else self.parent.level + 1

