from multipledispatch import dispatch
from numbers import Number
import numpy as np
import random
import math

def weights(values): return np.array(values) / sum(values)

@dispatch(Number, Number)
def floor(value, step): return math.floor(value / step) * step #returns the greatest multiple of step less than or equal to value

@dispatch(Number, Number)
def ceil(value, step): return math.ceil(value / step) * step #returns the smallest multiple of step greater than or equal to value

@dispatch(Number, Number, Number)
def clamp(value, min, max):
	if value < min:
		return min
	elif value > max:
		return max
	else:
		return value

@dispatch(tuple, tuple, float)
def clamp(value, min, max):
	return tuple(map(lambda x, y, z: clamp(x, y, z), value, min, max))

def clamp01(value): return clamp(value, 0.0, 1.0)

def repeat(t, length):
	return clamp(t - math.floor(t / length) * length, 0.0, length)

def ping_pong(t, length):
	t = repeat(t, length * 2)
	return length - abs(t - length)


@dispatch(Number, Number, float)
def lerp(a, b, t):
	return a * (1 - t) + b * t	#a + (b - a) * t

@dispatch(int, int, float)
def lerp(a, b, t):
	return int(a * (1 - t) + b * t)
	
@dispatch(tuple, tuple, float)
def lerp(a, b, t):
	return tuple(map(lambda x, y: lerp(x, y, t), a, b))

@dispatch(list, list, float)
def lerp(a, b, t):
	return tuple(map(lambda x, y: lerp(x, y, t), a, b))




from functools import reduce

def sqr_magnitude(v): return reduce(lambda a, b: a+b*b, v)

def magnitude(v): return math.sqrt(sqr_magnitude(v))

def sub(a, b): return map(lambda x, y: x - y, a, b)

def raised_norm(a, n): return np.sum(list(map(lambda x: abs(x) ** n, a)))

def norm(a, n): return raised_norm(a, n) ** (1.0 / n)

def raised_distance(a, b, n=2): return raised_norm(list(map(lambda x, y: y - x, a, b)), n)

def distance(a, b, n=2): return raised_distance(a, b, n) ** (1.0 / n)

#def distance(a, b): return magnitude(sub(a, b))

@dispatch(list, list, float)
def inverse_lerp(a, b, c): return distance(a, c) / distance(a, b) #NOT COMPLETE YET

@dispatch(Number, Number, Number)
def inverse_lerp(a, b, c): return (c - a) / (b - a)


def mean(iterable): return sum(iterable) / len(iterable)



@dispatch(Number, Number)
def random_range(start, stop):
	return random.uniform(start, stop)

@dispatch(int, int)
def random_range(start, stop):
	return random.randrange(start, stop + 1)

@dispatch(tuple, tuple)#, float)
def random_range(start, stop):
	return tuple(map(lambda x, y: random_range(x, y), start, stop))





def line_line_intersection(line1, line2):
    # Unpack the lines
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    # Calculate slopes (m)
    m1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')  # Avoid division by zero
    m2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')  # Avoid division by zero
    
    # Check if lines are parallel
    if m1 == m2:
        return None  # No intersection, parallel lines
    
    # Calculate intersection point
    x = (m1*x1 - m2*x3 + y3 - y1) / (m1 - m2)
    y = m1 * (x - x1) + y1
    
    return x, y


def line_segment_intersection(line, segment):
	point = line_line_intersection(line, segment)
	if point == None: return None
	t = inverse_lerp(segment[0], segment[1], point)
	if t < 0 or t > 1: return None
	return point

def random_in_circle(radius, center):
	vector = polar_to_cartesian(random_range(0.0, radius), random_range(0.0, 2.0 * math.pi))
	return tuple(map(lambda x, y: x + y, center, vector))

def polar_to_cartesian(r, theta): return r * math.cos(theta), r * math.sin(theta)
def cartesian_to_polar(x, y): return math.sqrt(x**2 + y**2), math.atan2(y, x)


#POINT

def on_arc(n, r=1.0, start=0.0, size=2.0 * math.pi): #where start is the start angle and size is the angular size, using default start and size is equal to call on_circle
	return [polar_to_cartesian(r, start + size * (i / (n - 1))) for i in range(n)]

def batch(total, size): return [min(total - (i * size), size) for i in range(math.ceil(total / size))]











