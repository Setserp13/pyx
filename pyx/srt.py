from datetime import datetime, time, timedelta
import pyx.osx as osx

def srtftime(x): return x.strftime("%H:%M:%S,%f")[:-3]

def srtptime(x): return datetime.strptime(x, "%H:%M:%S,%f").time()

def microseconds(x): return x.microsecond + 1_000_000 * (x.second + 60 * x.minute + 3600 * x.hour)

def seconds(x): return microseconds(x) / 1_000_000

class Time(time):
	def __new__(self, hours=0, minutes=0, seconds=0, milliseconds=0):
		millisecond = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
		second = millisecond // 1000
		minute = second // 60
		return super().__new__(self, hour=minute // 60, minute=minute % 60, second=second % 60, microsecond=(millisecond % 1000) * 1000)

class SubRipItem():
	def __init__(self, index, start, end, text):
		self.index = index
		self.start = start
		self.end = end
		self.text = text

	def strf(self): return f'{self.index}\n{srtftime(self.start)} --> {srtftime(self.end)}\n{self.text}\n\n'
		
	@staticmethod
	def strp(value):
		index, interval, text = value.split('\n')
		start, end = interval.split(' --> ')
		return SubRipItem(index, srtptime(start), srtptime(end), text)

	def shift(self, **kwargs):	#hours, minutes, seconds, microseconds
		self.start += timedelta(**kwargs)
		self.end += timedelta(**kwargs)

	def expand(self, **kwargs):
		self.start -= timedelta(**kwargs)
		self.end += timedelta(**kwargs)

class SubRipFile(list):
	def strf(self): return ''.join([x.strf() for x in self])

	@staticmethod
	def strp(value):
		result = SubRipFile()
		items = [x for x in value.split('\n\n') if x != '']
		for x in items:
			result.append(SubRipItem.strp(x))
		return result

	def shift(self, **kwargs):
		for x in self:
			x.shift(**kwargs)

	def expand(self, **kwargs):
		for x in self:
			x.expand(**kwargs)

	def save(self, output_path, encoding='utf-8'):
		#print(self.strf())
		osx.write(output_path, self.strf(), encoding)


#print(SubRipFile.strp(osx.read('subs.srt')).strf())

import pyx.rex as rex

def create_subs(text, labels, start=0): #Text and labels must have the same number of lines. Start must be in seconds.
	result = SubRipFile()
	labels = [rex.findnumbers(x) for x in labels.split('\n')]
	for i, x in enumerate(text.split('\n')):
		#print(i)
		result.append(SubRipItem(index=i+1, start=Time(milliseconds=int((start+labels[i][0])*1000)), end=Time(milliseconds=int((start+labels[i][1])*1000)), text=x))
	return result
	
