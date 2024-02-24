from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas






def wrapText(pdf, text, style, w): #text can be a string or a list of strings. In the second case you can avoid words being splitted
	i = 0
	result = ''
	for x in text:#.split(' '):
		if pdf.stringWidth(result[i:] + x, style[0], style[1]) > w:
			i += len(result)
			result += '\n' + x.lstrip()
		else:
			result += x
	return result




def truncateText(pdf, text, style, w, h, line_spacing=10):
	cur_i = 0
	lns = 0
	for i in range(len(text)):
		if pdf.stringWidth(text[cur_i:i+1], style[0], style[1]) > w:
			print(text[:i])
			text = text[:i] + '\n' + text[i:]
			cur_i = i
		if(text[i] == '\n'):
			lns += 1
			cur_i = i
		if(lns == h // line_spacing):
			return text[:i], text[i:]
	return text, ''

def setStyle(canvas, font, fontSize, fontColor=(0,0,0,1)):
	canvas.setFont(font, fontSize)
	canvas.setFillColorRGB(*fontColor)

import functools

def textSize(canvas, text, style):
	setStyle(canvas, **style)
	lines = text.split('\n')
	return (functools.reduce(lambda a, b: max(canvas.stringWidth(a, style['font'], style['fontSize']),
	canvas.stringWidth(b, style['font'], style['fontSize'])), lines), len(lines) * canvas._leading + (len(lines) - 1)
	* style['lineSpacing'])	

def drawText(canvas, x, y, text, style):
	setStyle(canvas, style['font'], style['fontSize'], style['fontColor'])
	for i, obj in enumerate(text.split('\n')):
		canvas.drawString(x, y - (i + 1) * (canvas._leading + style['lineSpacing']), obj)



from pyx.generic import *

def drawJustifiedString(canvas, x, y, text, w, fontName, fontSize):
	canvas.setFont(fontName, fontSize)
	words = text.split(' ')
	spacing = (w - canvas.stringWidth(text.replace(' ', ''), fontName, fontSize)) / (len(words) - 1)
	for word in words:
		canvas.drawString(x, y, word)
		x += canvas.stringWidth(word, fontName, fontSize) + spacing
	

def drawTexts(pdf, canvas, page, rect, texts, styles):
	h = 0
	x = rect[0]
	y = rect[1] + rect[3]

	#print(texts)	

	while(len(texts) > 0):
		style = styles[texts[0][0]]
		setStyle(canvas, style['font'], style['fontSize'], style['fontColor']) #TO GET LEADING
		text = texts[0][1]
		#print(text)

		cur_i = 0
		for i in range(len(text)):
			if canvas.stringWidth(text[cur_i:i+1], style['font'], style['fontSize']) > rect[2] or text[i] == '\n':
				#canvas.drawString(x, y - h, text[:i])
				
				pdf.insert(page, Invoker(setStyle, canvas, style['font'], style['fontSize'], style['fontColor']))
				#print(text[cur_i:i])
				h += canvas._leading
				pdf.insert(page, Invoker(canvas.drawString, x, y - h, text[cur_i:i].replace('\n', '')))
				h += style['lineSpacing']
				cur_i = i
			if(h + canvas._leading + style['lineSpacing'] > rect[3]):
				texts[0][1] = text[i:]
				return texts
		texts.pop(0)
		#del texts[0]
	#print(texts)
	return texts

from alpha.array_utility import sv as sv
def drawTexts2(pdf, canvas, page, rect, texts, styles):
	h = 0
	x = rect[0]
	y = rect[1] + rect[3]

	#print(texts)	

	while(len(texts) > 0):
		style = styles[texts[0][0]]
		setStyle(canvas, style['font'], style['fontSize'], style['fontColor']) #TO GET LEADING
		text = texts[0][1]

		cur_i = 0
		for i in range(len(text)):
			if canvas.stringWidth(sv(text[cur_i:i+1], ' '), style['font'], style['fontSize']) > rect[2] or '\n' in text[i]:
				
				pdf.insert(page, Invoker(setStyle, canvas, style['font'], style['fontSize'], style['fontColor']))
				h += canvas._leading
				if '\n' in text[i]:
					pdf.insert(page, Invoker(canvas.drawString, x, y - h, sv(text[cur_i:i+1], ' ').replace('\n', '')))
				else:
					pdf.insert(page, Invoker(canvas.drawString, x, y - h, sv(text[cur_i:i], ' ').replace('\n', '')))
				h += style['lineSpacing']
				cur_i = i
			if(h + canvas._leading + style['lineSpacing'] > rect[3]):
				texts[0][1] = text[i:]
				return texts
		texts.pop(0)
		#del texts[0]
	#print(texts)
	return texts


#def drawText(pdf, x, y, text, line_spacing=0):
#	for i, obj in enumerate(text.split('\n')):
#		pdf.drawString(x, y - (i + 1) * (pdf._leading + line_spacing), obj)



"""def drawTexts(pdf, rects, text, style):
	for x in rects:
		#pdf_canvas.rect(x[0], x[1], x[2], x[3], fill=True, stroke=True)
		in_rect, out_rect = split_text(pdf, text, style, x)
		print(in_rect)
		in_rect = keepInWidth(pdf, in_rect, style, w)
		#pdf.drawString(x[0], x[1] + x[3], in_rect)
		pdf.drawText(x[0], x[1] + x[3], in_rect)
		text = out_rect"""













