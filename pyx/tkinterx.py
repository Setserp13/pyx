from alpha.dict_util import items
import pandas as pd
from alpha.osx import wd
from tkinter import filedialog
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