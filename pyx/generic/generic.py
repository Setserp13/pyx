class Event(list):
	def invoke(self, *args, **kwargs):
		for x in self:
			x(*args, **kwargs)

class Invoker:
	def __init__(self, func, *args, **kwargs):
		self.func = func
		self.args = args
		self.kwargs = kwargs

	def invoke(self):
		return self.func(*self.args, **self.kwargs)

class Object:
	def __init__(self, **kwargs):
		for key, value in kwargs.items():
			setattr(self, key, value)

class Validator(list):
	def invoke(self, *args):
		return len(filter(lambda x: not x(*args), self)) == 0

"""class Child(Parent): #Example
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)"""
