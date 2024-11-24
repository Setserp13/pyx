from datetime import time, datetime
import pyx.osx as osx

def srtftime(x): return x.strftime("%H:%M:%S,%f")[:-3]

def srtptime(x): return datetime.strptime(x, "%H:%M:%S,%f").time()

def total_second(x): return x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1_000_000

"""class SubRipTime():
	def __init__(self, hours=0, minutes=0, seconds=0, milliseconds=0):
		# hours: The hours as an integer greater than or equal to 0.
		# minutes: The minutes as an integer between 0 and 59.
		# seconds: The seconds as an integer between 0 and 59.
		# milliseconds: The milliseconds as an integer between 0 and 999
		milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
		seconds = milliseconds // 1000
		self.milliseconds = milliseconds % 1000
		minutes = seconds // 60
		self.seconds = seconds % 60
		self.hours = minutes // 60
		self.minutes = minutes % 60

	def strftime(self): return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d},{self.milliseconds:03d}"

	@staticmethod
	def strptime(value):
		# Assuming value is in the format HH:MM:SS,mmm
		hours, minutes, seconds_milliseconds = value.split(':')
		seconds, milliseconds = seconds_milliseconds.split(',')
		return SubRipTime(int(hours), int(minutes), int(seconds), int(milliseconds))

	def shift(self, hours=0, minutes=0, seconds=0, milliseconds=0):
		return SubRipTime(self.hours + hours, self.minutes + minutes, self.seconds + seconds, self.milliseconds + milliseconds)

	@property
	def time(self): return self.hours * 3600 + self.minutes * 60 + self.seconds + self.milliseconds / 1000"""

class SubRipItem():
	def __init__(self, index, start, end, text):
		self.index = index
		self.start = start
		self.end = end
		self.text = text

	def strf(self): return f'{self.index}\n{srtftime(self.start)} --> {srtftime(self.end)}\n{self.text}\n\n'
	#def strf(self): return f'{self.index}\n{self.start.strftime()} --> {self.end.strftime()}\n{self.text}\n\n'
		
	@staticmethod
	def strp(value):
		index, interval, text = value.split('\n')
		start, end = interval.split(' --> ')
		return SubRipItem(index, SubRipTime.strptime(start), SubRipTime.strptime(end), text)
		#return SubRipItem(index, SubRipTime.strptime(start), SubRipTime.strptime(end), text)

	def shift(self, hours=0, minutes=0, seconds=0, milliseconds=0):
		self.start = self.start.shift(hours, minutes, seconds, milliseconds)
		self.end = self.end.shift(hours, minutes, seconds, milliseconds)

	def expand(self, hours=0, minutes=0, seconds=0, milliseconds=0):
		self.start = self.start.shift(-hours, -minutes, -seconds, -milliseconds)
		self.end = self.end.shift(hours, minutes, seconds, milliseconds)

class SubRipFile(list):
	def strf(self): return ''.join([x.strf() for x in self])

	@staticmethod
	def strp(value):
		result = SubRipFile()
		items = [x for x in value.split('\n\n') if x != '']
		for x in items:
			result.append(SubRipItem.strp(x))
		return result

	def shift(self, hours=0, minutes=0, seconds=0, milliseconds=0):
		for x in self:
			x.shift(hours, minutes, seconds, milliseconds)

	def expand(self, hours=0, minutes=0, seconds=0, milliseconds=0):
		for x in self:
			x.expand(hours, minutes, seconds, milliseconds)

	def save(self, output_path, encoding='utf-8'):
		#print(self.strf())
		osx.write(output_path, self.strf(), encoding)


#print(SubRipFile.strp(osx.read('subs.srt')).strf())

	
