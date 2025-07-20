import math
from PIL import Image, ImageFont, ImageDraw
import pyx.mat.mat as mat

def concat(lst, axis=0, equal_sized=False, mode='RGBA'):
	cell_size = lst[0].size if equal_sized else mat.arg(max, [x.size for x in lst])
	size = list(cell_size)
	size[axis] *= len(lst)
	#print(size)
	result = Image.new(mode, tuple(size))
	for i, x in enumerate(lst):
		pos = [0, 0]
		pos[axis] = i * cell_size[axis]
		#print(pos)
		result.paste(x, tuple(pos))
	return result

import numpy as np
import pyx.numpyx as npx

def sprites(img, regions): return [img.crop((*x.min, *x.max)) for x in regions]

def slice_by_cell_count(img, cell_count): return slice_by_cell_size(img, np.array(img.size) // np.array(cell_count))

def slice_by_cell_size(img, cell_size): return sprites(img, npx.rect2(0,0,*img.size).slice_by_cell_size(np.array(cell_size)))

"""def slice(img, cell_count, axis=0):
	size = img.size
	cell_size = list(size)
	cell_size[axis] = size[axis] // cell_count
	result = []
	for i in range(cell_count):
		min = [0, 0]
		max = list(size)
		min[axis] = i * cell_size[axis]
		max[axis] = (i+1) * cell_size[axis]
		cur = img.crop((min[0], min[1], max[0], max[1]))
		result.append(cur)	
	return result"""

#grid(lst, 1, 0) concats vertically and grid(lst, 1, 1) concats horizontally
def grid(lst, constraint_count, start_axis=0, equal_sized=False, mode='RGBA'):
	cell_size = lst[0].size if equal_sized else mat.arg(max, [x.size for x in lst])
	size = list(cell_size)
	size[start_axis] *= constraint_count
	axis2 = (start_axis+1)%2
	size[axis2] *= math.ceil(len(lst) / constraint_count)
	#print(size)
	result = Image.new(mode, tuple(size))
	for i, x in enumerate(lst):
		pos = [0, 0]
		pos[start_axis] = (i % constraint_count) * cell_size[start_axis]
		pos[axis2] = (i // constraint_count) * cell_size[axis2]
		#print(pos)
		result.paste(x, tuple(pos))
	return result

def getsize(lines, font, font_size, leading=0):
	width = max(get_size(line, font, font_size)[0] for line in lines)
	height = sum(get_size(line, font, font_size)[1] for line in lines) + leading * (len(lines) - 1)
	return width, height

def get_size(text, font, font_size):
	image_font = ImageFont.truetype(font, font_size)
	temp_image = Image.new("RGB", (1, 1))
	draw = ImageDraw.Draw(temp_image)
	bbox = draw.textbbox((0, 0), text, font=image_font)
	return (bbox[2] - bbox[0], bbox[3] - bbox[1])

def wrap(line, width, font, font_size):
	result = ['']
	for word in line.split():
		if result[-1] == '':
			result[-1] += word
		else:
			space = ' '
			size = get_size(result[-1] + space + word, font, font_size)
			if size[0] > width:
				result.append(word)
			else:
				result[-1] += space + word
	return result#'\n'.join(result)

def truncate(lines, height, font, font_size, leading=0):
	for i in range(len(lines)):
		if getsize(lines[:i+1], font, font_size, leading)[1] > height:
			return lines[:i], lines[i:]
	return lines, []

def best_fit(line, size, font, leading=0, max_font_size=300):
	font_size = max_font_size
	lines = [line]
	for i in range(max_font_size, 0, -1): #from max_font_size to 1
		font_size = i
		lines = wrap(line, size[0], font, font_size)
		text_size = getsize(lines, font, font_size, leading)
		if text_size[1] <= size[1]:
			break
	return lines, font_size

def palette(image):
	image = np.asarray(image)
	pixels = image.reshape(-1, image.shape[-1])
	#return list({tuple(p) for p in pixels})
	return np.unique(pixels, axis=0)



def resize(img, new_shape, interpolation=0):
	return [
		resize_nearest,
		resize_bilinear
	][interpolation](img, new_shape)

def scale(img, value, interpolation=0): return resize(img, img.shape[:2] * np.array(value), interpolation=interpolation)

def resize_nearest(img, new_shape):
	h, w = new_shape.astype(int)
	H, W = img.shape[:2]
	y = (np.arange(h) * H / h).astype(int)
	x = (np.arange(w) * W / w).astype(int)
	return img[y[:, None], x]

def resize_bilinear(img, new_shape):
	h, w = new_shape.astype(int)
	H, W = img.shape[:2]

	# Target float coordinates
	y = np.linspace(0, H - 1, h)
	x = np.linspace(0, W - 1, w)

	y0 = np.floor(y).astype(int)
	x0 = np.floor(x).astype(int)

	y1 = np.clip(y0 + 1, 0, H - 1)
	x1 = np.clip(x0 + 1, 0, W - 1)

	dy = (y - y0)[:, None]
	dx = (x - x0)[None, :]

	y0 = np.clip(y0, 0, H - 1)
	x0 = np.clip(x0, 0, W - 1)

	# Sample pixels
	top_left     = img[y0[:, None], x0[None, :]]
	top_right    = img[y0[:, None], x1[None, :]]
	bottom_left  = img[y1[:, None], x0[None, :]]
	bottom_right = img[y1[:, None], x1[None, :]]

	# Reshape weights for broadcasting over channels
	dx = dx[..., None]  # shape (1, w, 1)
	dy = dy[..., None]  # shape (h, 1, 1)

	# Bilinear interpolation
	top    = top_left * (1 - dx) + top_right * dx
	bottom = bottom_left * (1 - dx) + bottom_right * dx
	result = top * (1 - dy) + bottom * dy

	return result.astype(img.dtype)
