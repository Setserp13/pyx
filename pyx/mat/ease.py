from easing_functions import *

class Ease:
    _easings = [
        "Quad", "Cubic", "Quartic", "Quintic", 
        "Sine", "Circular", "Exponential",
        "Elastic", "Back", "Bounce"
    ]
    _modes = ["In", "Out", "InOut"]
	Linear = LinearInOut(0, 1, 1)

# Dynamically attach easing functions as class attributes
for name in Ease._easings:
	for mode in Ease._modes:
		clsname = f"{name}Ease{mode}"
		if clsname in globals():
			setattr(Ease, f"{name}{mode}", globals()[clsname](0, 1, 1))

"""from easing_functions import *
#pip install easing-functions

#print(dir(easing_functions))

class Ease:

	QuadIn = QuadEaseIn(0, 1, 1)#start=0, end=1, duration=1)
	QuadOut = QuadEaseOut(0, 1, 1)
	QuadInOut = QuadEaseInOut(0, 1, 1)

	CubicIn = CubicEaseIn(0, 1, 1)
	CubicOut = CubicEaseOut(0, 1, 1)
	CubicInOut = CubicEaseInOut(0, 1, 1)

	QuarticIn = QuarticEaseIn(0, 1, 1)
	QuarticOut = QuarticEaseOut(0, 1, 1)
	QuarticInOut = QuarticEaseInOut(0, 1, 1)

	QuinticIn = QuinticEaseIn(0, 1, 1)
	QuinticOut = QuinticEaseOut(0, 1, 1)
	QuinticInOut = QuinticEaseInOut(0, 1, 1)

	SineIn = SineEaseIn(0, 1, 1)
	SineOut = SineEaseOut(0, 1, 1)
	SineInOut = SineEaseInOut(0, 1, 1)

	CircularIn = CircularEaseIn(0, 1, 1)
	CircularOut = CircularEaseOut(0, 1, 1)
	CircularInOut = CircularEaseInOut(0, 1, 1)

	ExponentialIn = ExponentialEaseIn(0, 1, 1)
	ExponentialOut = ExponentialEaseOut(0, 1, 1)
	ExponentialInOut = ExponentialEaseInOut(0, 1, 1)

	ElasticIn = ElasticEaseIn(0, 1, 1)
	ElasticOut = ElasticEaseOut(0, 1, 1)
	ElasticInOut = ElasticEaseInOut(0, 1, 1)

	BackIn = BackEaseIn(0, 1, 1)
	BackOut = BackEaseOut(0, 1, 1)
	BackInOut = BackEaseInOut(0, 1, 1)

	BounceIn = BounceEaseIn(0, 1, 1)
	BounceOut = BounceEaseOut(0, 1, 1)
	BounceInOut = BounceEaseInOut(0, 1, 1)

	Linear = LinearInOut(0, 1, 1)
"""
