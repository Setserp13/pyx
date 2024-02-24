from kivy.uix.widget import Widget

def add_widgets(self, widgets):
	for x in widgets:
		self.add_widget(x)

def remove_widgets(self, widgets):
	for x in widgets:
		self.remove_widget(x)

def set_parent(self, widget):
	if self.parent != None:
		self.parent.remove_widget(self)
	widget.add_widget(self)

def add_child(self, widget):
	widget.set_parent(self)
	
def add_children(self, widgets):
	for x in widgets:
		self.add_child(x)

def sort_widgets(self, key, reverse=False):
	children = sorted(list(self.children), key=key, reverse=reverse)
	self.clear_widgets()
	self.add_widgets(children)

Widget.add_child = add_child
Widget.add_children = add_children
Widget.add_widgets = add_widgets
Widget.remove_widgets = remove_widgets
Widget.set_parent = set_parent
Widget.sort_widgets = sort_widgets