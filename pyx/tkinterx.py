from pyx.array_utility import items
import pandas as pd
from pyx.osx import wd
import tkinter as tk
from tkinter import ttk, filedialog
import inspect
import xml.etree.ElementTree as ET

def askopenxmls(initialdir=wd(__file__)):
	return [ET.parse(x).getroot() for x in askopenfilenames(initialdir, 'xml')]

def askopenfilenames(initialdir=wd(__file__), filetypes='*'): #filetypes is a string like filetype1.filetype2.....filetypeN
	return filedialog.askopenfilenames(initialdir=initialdir, title='Choose files', filetypes=[(x, f'*.{x}') for x in filetypes.split('.')])

def askopenworkbook(initialdir=wd(__file__), title='Choose a workbook', **kwargs):
	file = filedialog.askopenfilename(filetypes=[("Excel file","*.xlsx *.csv"),("Excel file", "*.xls")], initialdir=initialdir, title=title, **items(kwargs, inspect.getfullargspec(filedialog.askopenfilename)[0]))
	if file.endswith('.csv'):
		return pd.read_csv(file, **items(kwargs, inspect.getfullargspec(pd.read_csv)[0]))
	elif file.endswith('.xlsx'):
		return pd.read_excel(file, **items(kwargs, inspect.getfullargspec(pd.read_excel)[0]))
	else:
		return None

class Path(tk.Frame):
	def __init__(self, root, filedialog=filedialog, textvariable=None, initialvalue='', initialdir='', on_select=None, **kwargs):
		super().__init__(root)
		self.textvariable = tk.StringVar() if textvariable == None else textvariable
		self.textvariable.set(initialvalue)
		tk.Entry(self, textvariable=self.textvariable, **kwargs).grid(sticky='w', row=0, column=0)
		self.on_select = on_select
		def command():
			path = filedialog(initialdir=self.textvariable.get())
			self.textvariable.set(path)
			if self.on_select != None:
				self.on_select(path)
		tk.Button(self, text='Browse', command=command).grid(sticky='w', row=0, column=1)

def Dir(root, textvariable=None, initialvalue='', initialdir='', **kwargs):
	return Path(root, filedialog=filedialog.askdirectory, textvariable=textvariable, initialvalue=initialvalue, initialdir=initialdir, **kwargs)

def File(root, textvariable=None, initialvalue='', initialdir='', **kwargs):
	return Path(root, filedialog=filedialog.askopenfilename, textvariable=textvariable, initialvalue=initialvalue, initialdir=initialdir, **kwargs)

WIDGETS = {
	'string': (tk.Entry, tk.StringVar, {}),
	'int': (tk.Spinbox, tk.IntVar, {'from_': 0, 'to': 100}),
	'bool': (tk.Checkbutton, tk.BooleanVar, {}),
	'list': (ttk.Combobox, tk.StringVar, {'values': []}),
	'dir': (Dir, tk.StringVar, {'initialdir': ''}),
	'file': (File, tk.StringVar, {'initialdir': ''})
	#'list': (tk.OptionMenu, tk.StringVar, {'menu': []})
}

class Form:
	def __init__(self, root, fields, onsubmit):
		self.data = {}
		for i, field in enumerate(fields):
			field_name = field['name']
			tk.Label(root, text=field_name).grid(sticky='w', row=i, column=0)
			widget_class, var_type, widget_kwargs = WIDGETS[field['type']]
			widget_var = var_type()

			#widget_kwargs = {k: field.get(k, v) for k, v in widget_kwargs.items()}

			kwargs = widget_kwargs.copy()
			if 'kwargs' in field:
				for k in field['kwargs']:
					kwargs[k] = field['kwargs'][k] 


			widget = widget_class(root, textvariable=widget_var, **kwargs)
			widget.grid(sticky='w', row=i, column=1)

			#widget.pack(fill=tk.X, padx=10, pady=10, expand=True)

			#self.data[field_name] = widget_var
			self.data[field['id']] = (widget, widget_var)
		print(self.data)
		submit_button = tk.Button(root, text='Submit', command=lambda: onsubmit(self.data))
		submit_button.grid(row=len(fields), column=0, columnspan=2)

