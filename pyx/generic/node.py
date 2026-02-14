from pyx.array_utility import index_of

class Node:
	def __init__(self, parent=None, children=None, **kwargs):
		for k in kwargs:
			setattr(self, k, kwargs[k])
		self.parent = None #Declare self.parent
		self.setParent(parent) #Node
		self.children = [] if children is None else children
	
	def setParent(self, value):
		if self.parent != value:
			if self.parent != None:
				self.parent.children.remove(self)
			if value != None:
				value.children.append(self)
			self.parent = value
	
	def siblingIndex(self):
		if self.parent != None:
			return index_of(self.parent.children, self)
		return -1

	def selfAndSiblings(self):
		if self.parent != None:
			return self.parent.children
	
	def findSelfOrSiblings(self, match):
		result = []
		if self.parent != None:
			for child in self.parent.children:
				if match(child):
					result.append(child)
		return result

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
		#value.setParent(None)	#remove a child

	def isRoot(self): return self.parent == None

	def isLeaf(self): return self.degree() == 0
	
	def neighbors(self):
		if self.parent == None:
			return self.children
		return [self.parent] + self.children

	def ancestors(self):
		result = []
		if self.parent != None:
			result.append(self.parent)
			result = result + self.parent.ancestors()
		return result
	
	def forAncestors(self, func):
		result = []
		if self.parent != None:
			func(self.parent)
			self.parent.forAncestors(func)

	def descendants(self):
		result = []
		for child in self.children:
			result.append(child)
			result = result + child.descendants()
		return result
	
	def size(self):
		result = 1
		for child in self.children:
			result = result + child.size()
		return result

	def forSelfAndDescendants(self, func):
		func(self)
		for child in self.children:
			child.forSelfAndDescendants(func)
	
	def nodesAtDistance(self, distance):
		if distance == 0:
			return [self]
		result = []
		for child in self.children:
			result = result + child.nodesAtDistance(distance - 1)
		return result

	def forDescendants(self, func):
		for child in self.children:
			func(child)
			child.forDescendants(func)
	
	def forChildren(self, func):
		for child in self.children:
			func(child)

	def degree(self): return len(self.children)

	def leaves(self):
		result = []
		if self.isLeaf():
			result.append(self)
		else:
			for child in self.children:
				result = result + child.leaves()
		return result	
	
	def breadth(self):
		if self.isLeaf():
			return 1
		else:
			result = 0
			for child in self.children:
				result = result + child.breadth()
			return result

	def root(self):
		if self.isRoot():
			return self
		return self.parent.root()

	def level(self):
		if self.isRoot():
			return 0
		return self.parent.level() + 1

	def findChild(self, match):
		for x in self.children:
			if(match(x)):
				return x
		return None
