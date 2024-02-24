from pydub import AudioSegment
import cv2
import os
import string

def ext(path): return os.path.splitext(path)[1]
def root(path): return os.path.splitext(path)[0]
def filename(path): return os.path.splitext(os.path.basename(path))[0]
def wd(__file__): return os.path.dirname(os.path.abspath(__file__))



#os.path.ext = lambda path: os.path.splitext(path)[1]

#os.path.root = lambda path: os.path.splitext(path)[0]



def clean_filename(filename):
	valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
	cleaned_filename = ''.join(c for c in filename if c in valid_chars)
	return cleaned_filename

def to_str(i, digits=0):
	return max(digits - len(str(i)), 0) * '0' + str(i)

def to_distinct(path, naming='root (i)', digits=0):
	result = path
	print([os.path.root(path), os.path.ext(path)])
	root, ext = os.path.splitext(path)
	i = 1
	while(os.path.exists(result)):
		result = naming.replace('i', to_str(i, digits)).replace('root', root) + ext
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

def open_audios(path_list, **kwargs): return open_all(path_list, AudioSegment.from_file, **kwargs)

def open_videos(path_list, **kwargs): return open_all(path_list, cv2.VideoCapture, **kwargs)

def remove_all(files):
	for x in files:
		os.remove(x)

def clear(dir): remove_all(listdirabs(dir))








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


def open_file(path, encoding='utf-8'):
	with open(path, 'r', encoding=encoding) as file:
		return file.read()

def save_file(new_file, path, encoding='utf-8'):
	with open(path, 'w', encoding=encoding) as file:
		file.write(new_file)

def mkdirs(path):
	dirname = os.path.dirname(path)
	os.makedirs(os.sep if dirname == '' else dirname, exist_ok=True)
#Makes all missing directories in the path

def split(path): return path.split(os.sep)


def export_file(file, path, makedirs=False, export_method=save_file, **kwargs):
	mkdirs(path)
	export_method(file, path, **kwargs)


print(split("\\aal-ws002\Incubatório\Etiquetas Diversas - Incubatório"))

def open_audio(path):
	return AudioSegment.from_file(path, format=ext(path))

def export_audio(audio, path):
	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	audio.export(path, format=ext(path))


"""def export_text(text, file_path, encoding='utf-8'):
	dirname = os.path.dirname(file_path)
	if dirname != '':
		os.makedirs(dirname, exist_ok=True)
	with open(file_path, 'w', encoding=encoding) as file: file.write(text)

def open_txt(file_path, encoding='utf-8'):
	#print(file_path)
	with open(file_path, 'r', encoding=encoding) as file:
		return file.read()"""






def first_ext(dir, ext):
	for x in os.listdir(dir):
		if x.endswith(ext):
			return x
	return None






def endswithsome(s, values):
	for x in values:
		if s.endswith(x):
			return True
	return False



from typing import overload

@overload
def find_files_by_extension(dir, ext :str):
	find_files_by_extension(dir, [ext])

@overload
def find_files_by_extension(dir, ext :list):
	#if not isinstance(ext, list):
	#	ext = [ext]
	return [os.path.join(dir, x) for x in os.listdir(dir) if endswithsome(x, ext)]



def find(pattern, dir):
	for x in os.listdir(dir):
		if re.search(pattern, x) != None:
			return os.path.join(dir, x)
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









def open_audios_from(dir):
	return open_audios([os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.mp3') or x.endswith('.wav')])

def open_videos_from(dir):
	return open_videos([os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.mp4')])