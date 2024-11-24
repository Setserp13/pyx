import os
import string
import json

def ext(path): return os.path.splitext(path)[1]
def root(path): return os.path.splitext(path)[0]
def filename(path): return os.path.splitext(os.path.basename(path))[0]
def wd(__file__): return os.path.dirname(os.path.abspath(__file__))

# Clear the console output
def clear(): os.system('cls' if os.name == 'nt' else 'clear')

def set_filename(file, value):
	dir = os.path.abspath(os.path.dirname(file))
	os.rename(os.path.abspath(file), to_distinct(os.path.join(dir, value) + ext(file)))

def definitions(module): return [x for x in dir(module) if not hasattr(getattr(module, x), '__path__')]
def submodules(module): return [x for x in dir(module) if hasattr(getattr(module, x), '__path__')]

def clean_filename(filename):
	valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
	cleaned_filename = ''.join(c for c in filename if c in valid_chars)
	return cleaned_filename

def to_str(i, digits=0):
	return max(digits - len(str(i)), 0) * '0' + str(i)

def to_distinct(path, naming='root (%i)', digits=0):
	result = path
	root, ext = os.path.splitext(path)
	i = 1
	while(os.path.exists(result)):
		result = naming.replace('%i', to_str(i, digits)).replace('root', root) + ext
		i += 1
	return result

def open_all(ls, open, **kwargs):
	result = []
	for x in ls:
		try:
			result.append(open(x, **kwargs))
		except Exception as e:
			print(f"Error loading {x}: {e}")
	return result
	#return [y for y in [open(x, **kwargs) for x in ls] if y != None]



def remove_all(files):
	for x in files:
		os.remove(x)

def clear(dir): remove_all(ls(dir))








def setsim(a, b):
	return bagsim(a.unique(), b.unique())

def bagsim(a, b):
	count = 0
	for x in a:
		if x in b:
			count+=1
	return count / max(len(a), len(b))

def seqsim(a, b):
	count = 0
	for i in range(min(len(a), len(b))):
		if a[i] == b[i]:
			count+=1
	return count / max(len(a), len(b))

#TEXT FILE

def read(file, encoding='utf-8'):
	with open(file, 'r', encoding=encoding) as f:
		return f.read()
		
def write(file, content, encoding='utf-8'):
	with open(file, 'w', encoding=encoding) as f:
		f.write(content)
		
def append(file, content, encoding='utf-8'):
	with open(file, 'a', encoding=encoding) as f:
		f.write(content)

#BINARY FILE

def readb(file):
	with open(file, 'rb') as f:
		return f.read()

def writeb(file, content):
	with open(file, 'wb') as f:
		f.write(content)

def appendb(file, content):
	with open(file, 'ab') as f:
		f.write(content)

#JSON FILE

def readjson(file, dflt_value=None, encoding='utf-8'):
	try:
		with open(file, 'r', encoding=encoding) as f:
			return json.load(f)
	except Exception as e:
		print(f"Error: {e}")
		return dflt_value

def writejson(file, content, encoding='utf-8', ensure_ascii=False, indent='\t'):
	with open(file, 'w', encoding=encoding) as f:
    		json.dump(content, f, ensure_ascii=ensure_ascii, indent=indent)




def mkdirs(path):
	dirname = os.path.dirname(path)
	os.makedirs(os.sep if dirname == '' else dirname, exist_ok=True)
#Makes all missing directories in the path

def split(path): return path.split(os.sep)

def join(parts): return os.sep.join(parts)


#print(split("C:\Users\diogo\Downloads"))

def export_file(file, path, makedirs=False, export_method=write, **kwargs):
	mkdirs(path)
	export_method(file, path, **kwargs)






def find(pattern, dir):
	for x in os.listdir(dir):
		if re.search(pattern, x) != None:
			return os.path.join(dir, x)
	return None

def findr(pattern, dir):
	for root, dirs, files in os.walk(dir):
		for file in files + dirs:
			if re.search(pattern, file) is not None:
				return os.path.join(root, file)
	return None

import re

def findall(pattern, dir):
	return [os.path.join(dir, x) for x in os.listdir(dir) if re.search(pattern, x) != None]

#find('.pdf$', dir)
#find('xlsx,xlsm,xlsb,xltx,xltm,xls,xlt,xml,xlam,xla,xlw,xlr'.replace(',', '$|.'), dir)




def try_remove(file_path):
	try:
		# Attempt to delete the file
		os.remove(file_path)
		print(f"File '{file_path}' has been deleted successfully.")
	except OSError as e:
		# Handle any errors that may occur during deletion
		print(f"Failed to delete file '{file_path}': {e}")


def ls(dir, abs=True): return [os.path.join(dir, x) for x in os.listdir(dir)] if abs else os.listdir(dir)
#Return a list containing the names of the entries in the directory given by path. The list is in arbitrary order. If abs is true, it returns the absolute path of each entry.

def lsall(dirs, abs=True): return [ls(x, abs) for x in dirs]






