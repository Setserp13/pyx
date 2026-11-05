import math


# Linear

def linear(t):
	return t


# Polynomial: quadratic

def ease_in_quad(t):
	return t ** 2

def ease_out_quad(t):
	return 1 - (1 - t) ** 2

def ease_in_out_quad(t):
	return 2 * t ** 2 if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2

def ease_out_in_quad(t):
	return 0.5 * ease_out_quad(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_quad((t - 0.5) * 2)


# Polynomial: cubic

def ease_in_cubic(t):
	return t ** 3

def ease_out_cubic(t):
	return 1 - (1 - t) ** 3

def ease_in_out_cubic(t):
	return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2

def ease_out_in_cubic(t):
	return 0.5 * ease_out_cubic(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_cubic((t - 0.5) * 2)


# Polynomial: quartic

def ease_in_quart(t):
	return t ** 4

def ease_out_quart(t):
	return 1 - (1 - t) ** 4

def ease_in_out_quart(t):
	return 8 * t ** 4 if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2

def ease_out_in_quart(t):
	return 0.5 * ease_out_quart(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_quart((t - 0.5) * 2)


# Polynomial: quintic

def ease_in_quint(t):
	return t ** 5

def ease_out_quint(t):
	return 1 - (1 - t) ** 5

def ease_in_out_quint(t):
	return 16 * t ** 5 if t < 0.5 else 1 - (-2 * t + 2) ** 5 / 2

def ease_out_in_quint(t):
	return 0.5 * ease_out_quint(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_quint((t - 0.5) * 2)


# Sinusoidal

def ease_in_sine(t):
	return 1 - math.cos(t * math.pi / 2)

def ease_out_sine(t):
	return math.sin(t * math.pi / 2)

def ease_in_out_sine(t):
	return -(math.cos(math.pi * t) - 1) / 2

def ease_out_in_sine(t):
	return 0.5 * ease_out_sine(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_sine((t - 0.5) * 2)


# Exponential

def ease_in_expo(t):
	return 0 if t == 0 else 2 ** (10 * t - 10)

def ease_out_expo(t):
	return 1 if t == 1 else 1 - 2 ** (-10 * t)

def ease_in_out_expo(t):
	if t == 0 or t == 1:
		return t
	return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2

def ease_out_in_expo(t):
	return 0.5 * ease_out_expo(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_expo((t - 0.5) * 2)


# Circular

def ease_in_circ(t):
	return 1 - math.sqrt(1 - t ** 2)

def ease_out_circ(t):
	return math.sqrt(1 - (t - 1) ** 2)

def ease_in_out_circ(t):
	return (1 - math.sqrt(1 - (2 * t) ** 2)) / 2 if t < 0.5 else (math.sqrt(1 - (-2 * t + 2) ** 2) + 1) / 2

def ease_out_in_circ(t):
	return 0.5 * ease_out_circ(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_circ((t - 0.5) * 2)


# Back: overshoots the endpoint

def ease_in_back(t, s=1.70158):
	return (s + 1) * t ** 3 - s * t ** 2

def ease_out_back(t, s=1.70158):
	return 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2

def ease_in_out_back(t, s=1.70158):
	s *= 1.525
	return (
		((2 * t) ** 2 * ((s + 1) * 2 * t - s)) / 2
		if t < 0.5 else
		((2 * t - 2) ** 2 * ((s + 1) * (2 * t - 2) + s) + 2) / 2
	)

def ease_out_in_back(t):
	return 0.5 * ease_out_back(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_back((t - 0.5) * 2)


# Elastic: oscillates around the endpoint

def ease_in_elastic(t, amplitude=1, period=0.3):
	if t == 0 or t == 1:
		return t
	s = period / (2 * math.pi) * math.asin(1 / amplitude) if amplitude >= 1 else period / 4
	return -(amplitude * 2 ** (10 * t - 10) * math.sin((t * 10 - 10.75) * 2 * math.pi / period))

def ease_out_elastic(t, amplitude=1, period=0.3):
	if t == 0 or t == 1:
		return t
	return 1 + amplitude * 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * 2 * math.pi / period)

def ease_in_out_elastic(t, amplitude=1, period=0.45):
	if t == 0 or t == 1:
		return t
	return (
		-(amplitude * 2 ** (20 * t - 10) * math.sin((20 * t - 11.125) * 2 * math.pi / period)) / 2
		if t < 0.5 else
		(amplitude * 2 ** (-20 * t + 10) * math.sin((20 * t - 11.125) * 2 * math.pi / period)) / 2 + 1
	)

def ease_out_in_elastic(t):
	return 0.5 * ease_out_elastic(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_elastic((t - 0.5) * 2)


# Bounce

def ease_out_bounce(t):
	n = 7.5625
	d = 2.75

	if t < 1 / d:
		return n * t * t
	elif t < 2 / d:
		t -= 1.5 / d
		return n * t * t + 0.75
	elif t < 2.5 / d:
		t -= 2.25 / d
		return n * t * t + 0.9375
	else:
		t -= 2.625 / d
		return n * t * t + 0.984375

def ease_in_bounce(t):
  return 1 - ease_out_bounce(1 - t)

def ease_in_out_bounce(t):
	return (
		(1 - ease_out_bounce(1 - 2 * t)) / 2
		if t < 0.5 else
		(1 + ease_out_bounce(2 * t - 1)) / 2
	)

def ease_out_in_bounce(t):
	return 0.5 * ease_out_bounce(t * 2) if t < 0.5 else 0.5 + 0.5 * ease_in_bounce((t - 0.5) * 2)



# Smoothstep

def smoothstep(t):
	return t * t * (3 - 2 * t)

def smootherstep(t):
	return t ** 3 * (t * (t * 6 - 15) + 10)


# Step

def step(t):
	return 0 if t < 1 else 1


# Hold, then transition

def delayed(t, delay=0.25):
	return max(0, min(1, (t - delay) / (1 - delay)))


# Anticipation

def anticipate(t, strength=1):
	return t * t * ((strength + 1) * t - strength)


# Smooth overshoot

def smooth_overshoot(t, strength=0.2):
	return t + strength * math.sin(math.pi * t)


# Smoother oscillation

def oscillate(t, cycles=3, decay=3):
	return 1 - math.exp(-decay * t) * math.cos(2 * math.pi * cycles * t)
