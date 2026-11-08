def is_index(key):
	return key.lstrip('-').isdigit()


def get(data, path, default=None):
	try:
		for key in path.split('/'):
			if isinstance(data, dict):
				data = data[key]
			elif is_index(key):
				data = data[int(key)]
			else:
				data = getattr(data, key)

		return data

	except (KeyError, AttributeError, IndexError, TypeError):
		return default


def set(data, path, value):
	keys = path.split('/')

	for key in keys[:-1]:
		if isinstance(data, dict):
			data = data.setdefault(key, {})
		elif is_index(key):
			data = data[int(key)]
		else:
			data = getattr(data, key)

	key = keys[-1]

	if isinstance(data, dict):
		data[key] = value
	elif is_index(key):
		data[int(key)] = value
	else:
		setattr(data, key, value)

	return data

def getattrs(obj, *keys): return [getattr(obj, k) for x in keys]

def setattrs(obj, **kwargs):
	for k in kwargs:
		setattr(obj, k, kwargs[k])
	return obj

class Element:
	def __new__(cls, *args, **kwargs):
		obj = super().__new__(cls)
		return Element.init(obj, **kwargs)

	@classmethod
	def init(cls, obj, **kwargs):
		obj.attrib = {}

		for k, v in kwargs.items():
			setattr(obj, k, v)

		return obj

	def set(self, **kwargs):
		for k in kwargs:
			self.attrib[k] = kwargs[k]
		return self
	
	#def get(self, *args): return [self.attrib[k] for k in args]
	def get(self, *args, **kwargs):
		return [
			[self.attrib[k] for k in args if k in self.attrib] +
			[self.attrib.get(k, default) for k, default in kwargs.items()]
		]

def Element_like(cls):	#def make_Element(cls):
	cls.set = lambda self, **attrib: setattrs(self, attrib={**getattr(self, "attrib", {}), **attrib})
	
	cls.get_dict = lambda self, *args, **kwargs: {
		**{k: self.attrib[k] for k in args if hasattr(self, 'attrib') and k in self.attrib},
		**{k: self.attrib.get(k, v) if hasattr(self, 'attrib') else v for k, v in kwargs.items()}
	}
	
	#cls.get = lambda self, *keys: [self.attrib[k] for k in keys] if hasattr(self, 'attrib') else None
	"""cls.get = lambda self, *args, **kwargs: [
		[self.attrib[k] for k in args if k in self.attrib] +
		[self.attrib.get(k, v) for k, v in kwargs.items()]
	] if hasattr(self, 'attrib') else None"""
	cls.get = lambda self, *args, **kwargs: self.get_dict(*args, **kwargs).values()


class Event(list):
	def invoke(self, *args, **kwargs):
		for x in self:
			x(*args, **kwargs)

	#def __call__(self, *args, **kwargs):
	#	self.invoke(*args, **kwargs)

	__call__ = invoke

class Invoker:
	def __init__(self, func, *args, **kwargs):
		self.func = func
		self.args = args
		self.kwargs = kwargs

	def invoke(self):
		return self.func(*self.args, **self.kwargs)

class Object:
	def __init__(self, **kwargs):
		setattrs(self, **kwargs)

class Validator(list):
	def invoke(self, *args):
		return all(x(*args) for x in self)
		#return len(filter(lambda x: not x(*args), self)) == 0

"""class Child(Parent): #Example
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)"""

class Value:
	def __init__(self, value, on_change):
		self._value = value
		self.on_change = on_change

	@property
	def value(self):
		return self._value

	@value.setter
	def value(self, value):
		self.on_change(self.value, value)
		self.value = value
