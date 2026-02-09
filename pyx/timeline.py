from datetime import time, timedelta

def microseconds(x): return x.microsecond + 1_000_000 * (x.second + 60 * x.minute + 3600 * x.hour)

def seconds(x): return microseconds(x) / 1_000_000


class Time(float):
	"""def __new__(cls, hours=0, minutes=0, seconds=0, milliseconds=0):
		total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
		return super().__new__(cls, total_seconds)"""
	
	def from_time(hour=0, minute=0, second=0, millisecond=0, microsecond=0):
		total_seconds = hour * 3600 + minute * 60 + second + millisecond / 1000 + microsecond / 1_000_000
		return Time(total_seconds)
	
	@property
	def hour(self): return int(self) // 3600

	@property
	def minute(self): return (int(self) % 3600) // 60

	@property
	def second(self): return int(self) % 60

	@property
	def millisecond(self): return int((self - int(self)) * 1000)

	@property
	def microsecond(self): return int((self - int(self)) * 1_000_000)
		
	@property
	def time(self): return time(hour=self.hour, minute=self.minute, second=self.second, microsecond=self.microsecond)

	#def __repr__(self):
	#	return f"Time({self.hours:02}:{self.minutes:02}:{self.seconds:02}.{self.milliseconds:03})"

class Interval():	#Clip
	"""def __init__(self, start, duration):
		self.start = Time(start)
		self.duration = Time(duration)"""
	
	def __init__(self, start, end):
		self.start = Time(start)
		self.duration = Time(end - start)
		#self.end = Time(end)

	#@classmethod
	#def start_end(cls, start, end): return cls(start, end - start)
	
	@classmethod
	def start_duration(cls, start, duration): return cls(start, start + duration)

	@property
	def end(self): return Time(self.start + self.duration)

	@end.setter
	def end(self, value): self.duration = Time(value - self.start)

	def shift(self, **kwargs):	#hours, minutes, seconds, microseconds
		self.start += timedelta(**kwargs)
		#self.end += timedelta(**kwargs)

	def expand(self, **kwargs):
		self.start -= timedelta(**kwargs)
		self.end += timedelta(**kwargs)

class Instant(Interval):
	def __init__(self, time):
		super().__init__(time, time)


class Layer(list):	#elements are Interval-like
	@property
	def start(self): return min([x.start for x in self])

	@property
	def end(self): return max([x.end for x in self])

	def shift(self, **kwargs):
		for x in self:
			x.shift(**kwargs)

	def expand(self, **kwargs):
		for x in self:
			x.expand(**kwargs)
