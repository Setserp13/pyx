import cv2

def height(self): return int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))

def fps(self): return self.get(cv2.CAP_PROP_FPS)

def frame_count(self): return self.get(cv2.CAP_PROP_FRAME_COUNT)

def size(self): return (height(self), width(self))

def width(self): return int(self.get(cv2.CAP_PROP_FRAME_WIDTH))

def duration(self): return frame_count(self) / fps(self)

def frame_index(self, t): return int(fps(self) * t)

def get_frame(self, index):
	self.set(cv2.CAP_PROP_POS_FRAMES, min(index, frame_count(self) - 1))
	retval, image = self.read()
	return image

"""cv2.VideoCapture.height = height
cv2.VideoCapture.fps = fps
cv2.VideoCapture.frame_count = frame_count
cv2.VideoCapture.size = size
cv2.VideoCapture.width = width
cv2.VideoCapture.duration = duration"""
