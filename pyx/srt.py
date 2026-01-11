import copy
from pyx.timeline import Time, TimeRange, Track
from datetime import datetime
import textwrap
import pyx.mat.mat as mat
import pyx.osx as osx
import pyx.rex as rex

def wrap_text(text, width=42, len_func=len, join=True):
	lines = []
	current = ""
	for word in text.split():
		if len_func(current) + len_func(word) + 1 <= width:
			if current:
				current += " " + word
			else:
				current = word
		else:
			lines.append(current)
			current = word
	if current:
		lines.append(current)
	return "\n".join(lines) if join else lines

def batch_lines(text: str, n: int, join=True):
	lines = text.split("\n") if isinstance(text, str) else text
	result = [lines[i:i+n] for i in range(0, len(lines), n)]
	return ["\n".join(x) if join else x for x in result]

def timing(items, start, duration):
	ws = mat.weights([len(x.text) for x in items])
	for i in range(len(items)):
		items[i].start = Time(start)
		start = items[i].end = Time(start + duration * ws[i])

def words(text): return ''.join([c for c in text if c not in ',.:;']).split()	#text.split(' ')

def sentences(text): return text.replace('\n', ' ').replace('  ', ' ').split('.')

def split_item(item, split=words):
	result = []
	duration = item.end - item.start
	if split == None:
		texts = [item.text]
	elif isinstance(split, int):
		texts = textwrap.wrap(item.text, width=split)#=30)
	else:
		texts = split(item.text)
	for i in range(len(texts)):
		chunk = copy.deepcopy(item) #do it like that to preserve all the other properties
		chunk.text = texts[i]
		result.append(chunk)
	timing(result, item.start, duration)
	return result

def split_subs(subs, split=words): return SubRipFile([y for x in subs for y in split_item(x, split=split)])


def srtftime(x): return x.strftime("%H:%M:%S,%f")[:-3]

def srtptime(x): return datetime.strptime(x, "%H:%M:%S,%f").time()


class SubRipItem(TimeRange): #herde essa classe de uma classe mais genérica depois, tornando-a irmã de clipes de vídeo, imagens, audios, transições, efeitos e etc...
	def __init__(self, index, start, end, text):
		self.index = index
		self.text = text
		super().__init__(start, end)
	
	def strf(self): return f'{self.index}\n{srtftime(self.start.time)} --> {srtftime(self.end.time)}\n{self.text}\n\n'
		
	@staticmethod
	def strp(value):
		parts = value.split('\n')
		index, interval, text = parts[0], parts[1], '\n'.join(parts[2:])
		start, end = interval.split(' --> ')
		return SubRipItem(index, seconds(srtptime(start)), seconds(srtptime(end)), text)



class SubRipFile(Track):
	def strf(self): return ''.join([x.strf() for x in self])

	@staticmethod
	def strp(value):
		result = SubRipFile()
		items = [x for x in value.split('\n\n') if x != '']
		for x in items:
			result.append(SubRipItem.strp(x))
		return result

	@staticmethod
	def from_srt(path):
		return SubRipFile.strp(osx.read(path))
		


	def save(self, output_path, encoding='utf-8'):
		#print(self.strf())
		osx.write(output_path, self.strf(), encoding)

	def wrap(self, width=42):	#if width is None do not wrap
		for i in range(len(self)):
			text = self[i].text.replace('\n', ' ')
			self[i].text = text if width is None else wrap_text(text, width=width)

#print(SubRipFile.strp(osx.read('subs.srt')).strf())



def create_subs(text, labels, start=0): #Text and labels must have the same number of lines. Start must be in seconds.
	result = SubRipFile()
	if isinstance(labels, str):
		labels = rex.to_label_track(labels)
	for i, x in enumerate(text.split('\n')):
		result.append(SubRipItem(index=i+1, start=Time(start+labels[i][0]), end=Time(start+labels[i][1]), text=x))
		#result.append(SubRipItem(index=i+1, start=Time(milliseconds=int((start+labels[i][0])*1000)), end=Time(milliseconds=int((start+labels[i][1])*1000)), text=x))
	return result
	















