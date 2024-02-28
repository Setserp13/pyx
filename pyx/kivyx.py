from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget

#

#WIDGET

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
Layout.clear = lambda self: self.remove_widgets(self.children)
Widget.remove_widgets = remove_widgets
Widget.set_parent = set_parent
Widget.sort_widgets = sort_widgets

#

#GRID LAYOUT

GridLayout.get = lambda self, i, j: self.children[len(self.children) - 1 - (i * self.cols + j)]
GridLayout.get_row = lambda self, i: list(map(lambda j: self.get(i, j), range(self.cols)))
GridLayout.index = lambda self, child: [index(self.children) // self.cols, index(self.children) % self.cols]
GridLayout.is_row_empty = lambda self, i: are_empty(self.get_row(i))
GridLayout.remove_at = lambda self, i, j: self.remove_widget(self.get(grid, i, j))
GridLayout.remove_row = lambda self, i: self.remove_widgets(self.get_row(i))
GridLayout.row_count = lambda self: math.ceil(len(self.children) / self.cols)








def labels(objs, **kwargs): return [Label(text=str(x), **kwargs) for x in objs]

Layout.is_empty = lambda self: are_empty(self.children)

Widget.is_editable = lambda self: isinstance(self, (Spinner, TextInput))#FocusBehavior)
Widget.is_empty = lambda self: self.text == ''


def are_empty(widgets): return all(lambda x: x.is_empty(), widgets)





"""def set_row_texts(grid, i, values):
	for j in range(grid.cols):
		getItem(grid, i, j).text = values[j]

def emptyRow(schema):
	return list(map(lambda key: schema[key](text='', size_hint_x=None, size_hint_y=None, width=100, height=30), schema))

def addRow(grid, schema):
	for x in emptyRow(schema):
		grid.add_widget(x)"""