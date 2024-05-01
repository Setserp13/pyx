from PIL import ImageFont

def getsize(lines, font, font_size, leading=0):
	font = ImageFont.truetype(font, font_size)
	width = max(font.getsize(line)[0] for line in lines)
	height = sum(font.getsize(line)[1] for line in lines) + leading * (len(lines) - 1)
	return width, height

"""def wrap(line, width, font, font_size):
	result = ['']
	font = ImageFont.truetype(font, font_size)
	for x in line:
		size = font.getsize(result[-1] + x)
		if size[0] > width:
			result.append(x)
		else:
			result[-1] += x
	return result"""

def wrap(line, width, font, font_size):
	result = ['']
	font = ImageFont.truetype(font, font_size)
	for word in line.split():
		if result[-1] == '':
			result[-1] += word
		else:
			space = ' '
			size = font.getsize(result[-1] + space + word)
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
