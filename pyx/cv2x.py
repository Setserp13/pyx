import cv2

def height(self): return int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))

def fps(self): return self.get(cv2.CAP_PROP_FPS)

def frame_count(self): return self.get(cv2.CAP_PROP_FRAME_COUNT)

def size(self): return (self.height(), self.width())

def width(self): return int(self.get(cv2.CAP_PROP_FRAME_WIDTH))

def duration(self): return self.frame_count() / self.fps()

def frame_index(self, t): return int(self.fps() * t)

def get_frame(self, index):
	self.set(cv2.CAP_PROP_POS_FRAMES, min(index, self.frame_count() - 1))
	retval, image = self.read()
	return image

cv2.VideoCapture.height = height
cv2.VideoCapture.fps = fps
cv2.VideoCapture.frame_count = frame_count
cv2.VideoCapture.size = size
cv2.VideoCapture.width = width
cv2.VideoCapture.duration = duration