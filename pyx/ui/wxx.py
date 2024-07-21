import wx

# pip install -U wxPython
#https://docs.wxwidgets.org/3.0/overview_python.html

class WindowMenu(wx.Panel):
	def __init__(self, parent, items): #items is a dict; label: Window
		super(WindowMenu, self).__init__(parent)
		self.panel = wx.BoxSizer(wx.VERTICAL)

		for k in items:
			btn = wx.Button(self, label=k)
			params = items[k]
			btn.Bind(wx.EVT_BUTTON, lambda evt, k=k, params=params: WindowForm(k, *params).Show())
			self.panel.Add(btn)

		self.SetSizer(self.panel)
		self.Layout()

class WindowForm(wx.Frame):
	def __init__(self, title, on_submit, *fields): #fields is a list of tuples like (label, widget_builder, kwargs)
		super().__init__(None, title=title, size=(400, 300))

		panel = wx.Panel(self)
		vbox = wx.BoxSizer(wx.VERTICAL)

		form = Form(panel, on_submit)

		for x in fields:
			form.Add(x[0], x[1], **x[2])

		vbox.Add(form, flag=wx.ALL | wx.CENTER, border=10)

		panel.SetSizer(vbox)
	

class Form(wx.Panel):
	def __init__(self, parent, on_submit):
		super().__init__(parent)
		self.panel = wx.BoxSizer(wx.VERTICAL)

		self.dict = {}

		self.submit = wx.Button(self, label='Submit')
		self.submit.Bind(wx.EVT_BUTTON, lambda evt: on_submit(self))

		self.SetSizer(self.panel)
		self.Layout()

	def Add(self, label, widget_builder, **kwargs):
		field = wx.BoxSizer(wx.HORIZONTAL)
		field.Add(wx.StaticText(self, label=label))
		widget = widget_builder(self, **kwargs)
		field.Add(widget)

		self.dict[label] = widget

		self.panel.Add(field, 0, wx.ALL, 5)

	def get(self, key): return self.dict[key].GetValue()

	def values(self): return [self.get(k) for k in self.dict]

class Vector2Input(wx.Panel):
	def __init__(self, parent, value=(0, 0), validator=None):
		super().__init__(parent)

		self.validator = validator

		self.x_input = wx.TextCtrl(self, value=str(value[0]), style=wx.TE_PROCESS_ENTER)
		self.y_input = wx.TextCtrl(self, value=str(value[1]), style=wx.TE_PROCESS_ENTER)

		self.x_label = wx.StaticText(self, label="X:")
		self.y_label = wx.StaticText(self, label="Y:")

		self.Bind(wx.EVT_TEXT_ENTER, self.on_text_enter, id=self.x_input.GetId())
		self.Bind(wx.EVT_TEXT_ENTER, self.on_text_enter, id=self.y_input.GetId())

		sizer = wx.BoxSizer(wx.HORIZONTAL)
		sizer.Add(self.x_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
		sizer.Add(self.x_input, 0, wx.ALL, 5)
		sizer.Add(self.y_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
		sizer.Add(self.y_input, 0, wx.ALL, 5)

		self.SetSizer(sizer)
		self.Layout()

	def on_text_enter(self, event):
		if self.validator:
			valid = self.validator(self.GetValue())
			if not valid:
				wx.MessageBox("Invalid vector input!", "Error", wx.OK | wx.ICON_ERROR)

	def GetValue(self):
		try:
			x = float(self.x_input.GetValue())
			y = float(self.y_input.GetValue())
			return (x, y)
		except ValueError:
			return None

class FileSelector(wx.Panel):
	def __init__(self, parent):
		super().__init__(parent)

		self.selected_files = []

		self.btn_select = wx.Button(self, label="Select Files")
		self.btn_select.Bind(wx.EVT_BUTTON, self.on_select_files)

		self.file_count_label = wx.StaticText(self, label="Selected files: 0")

		sizer = wx.BoxSizer(wx.HORIZONTAL)
		sizer.Add(self.btn_select, 0, wx.ALL, 5)
		sizer.Add(self.file_count_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

		self.SetSizer(sizer)
		self.Layout()


	def GetValue(self):
		return self.selected_files

	def on_select_files(self, event):
		dialog = wx.FileDialog(
			self,
			"Select Files",
			wildcard="All files (*.*)|*.*",
			style=wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST
		)

		if dialog.ShowModal() == wx.ID_CANCEL:
			dialog.Destroy()
			return

		selected_paths = dialog.GetPaths()
		self.selected_files = selected_paths
		self.update_file_count_label(len(selected_paths))
		dialog.Destroy()

	def update_file_count_label(self, count):
		self.file_count_label.SetLabel(f"Selected files: {count}")
