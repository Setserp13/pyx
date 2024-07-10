import os
from manim import *
import re

from functools import reduce
from pyx.numpyx import *

class ValueTrackers():
	def __init__(self, **values):
		self.values = {	k: ValueTracker(values[k]) for k in values }

	def get_values(self):
		return { k: self.values[k].get_value() for k in self.values }

	def get_value(self, k):
		#print([k, self.values[k].get_value()])
		return self.values[k].get_value()

	def move_to(self, k, v): return self.values[k].animate.move_to(v)

def draw_at(mobject, x, y, **kwargs): return mobject(**kwargs).move_to(RIGHT * x + UP * y)

def cubic_beziergon(points):
	result = VMobject()
	curves = [[], [], [], []]
	for i in range(len(points) // 3):
		i3 = 3 * i
		for j in range(4):
			curves[j].append(points[(i3+j)%len(points)])
	result.set_anchors_and_handles(*curves)
	return result

def Tracer(scene, *path, dot_color=RED, stroke_width=4, stroke_color=RED, dissipating_time=None, rate_func=rate_functions.ease_in_out_quad):
	dot = Dot(color=RED)

	"""bounds = aabb(*path)
	def get_center():
		return clamp(bounds, dot.get_center())
	traced_point_func = get_center"""
	traced_point_func = dot.get_center
	

	trace = TracedPath(traced_point_func, stroke_width=stroke_width, stroke_color=stroke_color)#, dissipating_time=dissipating_time)
	scene.add(trace, dot)
	dot.move_to(path[0])
	for x in path:#[1:]:
		scene.play(dot.animate.move_to(x), rate_func=rate_func)

def zig_zag_path(cell_count):
	result = []
	for i in range(cell_count[1]):
		result += [np.array([0,i,0]), np.array([cell_count[0],i,0])]
	#print(result)
	return result

def MoveAllTo(scene, position, *objects):
	animations = [x.animate.move_to(position) for x in objects]
	scene.play(AnimationGroup(*animations))

def Operate(scene, a, b, o, c, axis=0, wait=0): #o(a, b) = c
	DIR = [RIGHT, UP][axis]
	scene.play(a.animate.next_to(o, direction=-DIR))
	scene.play(b.animate.next_to(o, direction=DIR))
	scene.wait(1)
	if c is not None:
		Combine(scene, c, o, a, b,	method=ScaleTransform)
	#MoveAllTo(scene, o.get_center(), a, b)

def ScaleTransform(a, b):
	b.move_to(a.get_center())
	return Succession(ShrinkToCenter(a).set_rate_func(rate_functions.ease_in_quad), GrowFromCenter(b).set_rate_func(rate_functions.ease_out_elastic))

def Combine(scene, target_object, *objects, method=Transform):
	#animations = [method(x, target_object) for x in objects]
	#scene.play(*animations)
	group = Group(*objects)
	scene.play(method(group, target_object))



def math_tokens(expression):
	pattern = r'(\d+|\+|\-|\*\*|\/|\*)'
	#pattern = r'(\d+|\+|\-|\*|\/|\^)'
	return re.findall(pattern, expression)

def eval_steps(expression, reverse=False):
	result = [expression]
	tokens = math_tokens(expression) #len(tokens) is always an odd number
	#print(tokens)
	steps = (len(tokens) - 1) // 2
	for i in range(steps):
		print(tokens[-3:])
		cur = ''.join(tokens[-3:] if reverse else tokens[:3])
		cur = eval(cur)
		cur = str(cur)
		#print(cur)
		tokens = tokens[:-3] + [cur] if reverse else [cur] + tokens[3:]
		result.append(''.join(tokens))
	result = [x.replace('**', '^') for x in result]
	#print(tokens)
	return result

from PIL import Image, ImageFilter

def DrawLine(scene, start, end, color=WHITE, run_time=1):
	print([start, end])
	segment = Line(start=start, end=end, color=color)
	scene.play(Create(segment), run_time=run_time)

def DrawLines(scene, lines, color=WHITE, run_time=1):
	for x in lines:
		DrawLine(scene, x[0], x[1], color, run_time / len(lines))


def RectangleMarker(scene, center, size, wait=2, color=RED, stroke_width=4, fill_opacity=0, create=Create, uncreate=Uncreate):
	rectangle = Rectangle(width=size[0], height=size[1], color=color)
	rectangle.set_z_index(-1)
	rectangle.move_to(center)
	rectangle.set_fill(opacity=fill_opacity)
	rectangle.set_stroke(width=stroke_width)  # Set stroke width if needed
	scene.play(create(rectangle))
	scene.wait(wait)
	scene.play(uncreate(rectangle))

def FadeInOutRectangleMarker(scene, center, size, wait=2, color=RED, stroke_width=4, fill_opacity=0.5):
	return RectangleMarker(scene, center, size, wait, color, stroke_width, fill_opacity, FadeIn, FadeOut)



def Background(scene, image_path, opacity=1, color=BLACK):
	config.background_color = color
	background = ImageMobject(image_path)
	background.set_z_index(-1000)
	background.set_opacity(opacity)
	background.scale_to_fit_height(config.frame_height)
	background.scale_to_fit_width(config.frame_width)
	background.move_to(ORIGIN)
	scene.add(background)

def MathSeries(terms, operation='+'):
	terms = [str(x) for x in terms]
	return [MathTex(operation.join(terms[:i+1])) for i in range(len(terms))]

def MathRes(scene, equations, transition=.1, wait=0, first=None, method=Transform):#TransformMatchingTex):
	start_index = 0
	if first == None:
		first = equations[0]
		start_index = 1
		scene.play(Write(first))
	for i in range(start_index, len(equations)):
		#print(equations[i])
		scene.play(method(first, equations[i]), run_time=transition)
		scene.wait(wait)
	return first

