import numpy as np
import pyx.sound as sd


# ============================================================
# MATERIAL
# ============================================================

class Material:

	def __init__(
		self,
		name,
		hardness=0.5,	# Resistance to deformation and penetration.
		density=0.5,	# Mass per unit volume; affects impact weight.
		roughness=0.5,	# Surface irregularity; affects friction and texture.
		damping=0.5,	# Energy absorption; controls how quickly vibrations decay.
		resonance=300,	# Natural resonant frequency in Hz.
	):

		self.name = name
		self.hardness = hardness
		self.density = density
		self.roughness = roughness
		self.damping = damping
		self.resonance = resonance

	def __repr__(self):
		return f'Material({self.name!r})'


class Materials:

	# Footwear
	sneaker = Material(
		'sneaker',
		hardness=0.25,
		density=0.5,
		roughness=0.7,
		damping=0.8,
		resonance=180,
	)

	boot = Material(
		'boot',
		hardness=0.7,
		density=0.9,
		roughness=0.5,
		damping=0.4,
		resonance=350,
	)

	# Surfaces
	concrete = Material(
		'concrete',
		hardness=0.95,
		density=0.95,
		roughness=0.3,
		damping=0.15,
		resonance=450,
	)

	wood = Material(
		'wood',
		hardness=0.6,
		density=0.6,
		roughness=0.4,
		damping=0.5,
		resonance=220,
	)

	gravel = Material(
		'gravel',
		hardness=0.5,
		density=0.7,
		roughness=0.95,
		damping=0.7,
		resonance=120,
	)

	# Other materials
	metal = Material(
		'metal',
		hardness=1.0,
		density=0.9,
		roughness=0.2,
		damping=0.05,
		resonance=900,
	)

	rubber = Material(
		'rubber',
		hardness=0.15,
		density=0.7,
		roughness=0.6,
		damping=0.95,
		resonance=100,
	)

	fist = Material(
		'fist',
		hardness=0.15,
		density=0.6,
		roughness=0.4,
		damping=0.9,
		resonance=150,
	)


# ============================================================
# INTERACTION
# ============================================================


class Impact:

	def __init__(
		self,
		material_a,
		material_b,
		force=1.0,
		speed=1.0,
		sample_rate=44100,
	):

		self.a = material_a
		self.b = material_b

		self.force = max(0, force)
		self.speed = max(0, speed)

		self.sample_rate = sample_rate

	def _lowpass(self, samples, frequency):

		alpha = 1.0 - np.exp(
			-2 * np.pi * frequency / self.sample_rate
		)

		result = np.empty_like(samples)
		result[0] = samples[0]

		for i in range(1, len(samples)):
			result[i] = (
				result[i - 1] +
				alpha * (samples[i] - result[i - 1])
			)

		return result

	def generate(self, duration=None):

		a = self.a
		b = self.b

		sr = self.sample_rate

		# --------------------------------------------------
		# Material interaction
		# --------------------------------------------------

		# The harder material dominates the impact.
		hardness = max(a.hardness, b.hardness)

		# Density contributes to the body of the sound.
		density = (
			a.density * 0.35 +
			b.density * 0.65
		)

		# Both materials contribute to surface texture.
		roughness = (
			a.roughness * 0.4 +
			b.roughness * 0.6
		)

		# The softer material absorbs more energy.
		damping = (
			a.damping * 0.65 +
			b.damping * 0.35
		)

		# The ground contributes most of the resonance.
		frequency = (
			b.resonance * 0.8 +
			a.resonance * 0.2
		)

		# --------------------------------------------------
		# Duration
		# --------------------------------------------------

		if duration is None:
			duration = 0.12 + density * 0.08

		count = max(1, int(duration * sr))

		t = np.arange(count) / sr

		# --------------------------------------------------
		# Impact strength
		# --------------------------------------------------

		force = np.sqrt(self.force * self.speed)

		# --------------------------------------------------
		# Decay
		# --------------------------------------------------

		decay = 18 + damping * 55

		envelope = np.exp(-decay * t)

		# --------------------------------------------------
		# Impact transient
		# --------------------------------------------------

		rng = np.random.default_rng()

		noise = rng.normal(0, 1, count)

		# Separate high-frequency impact noise.
		low_noise = self._lowpass(
			noise,
			1200 + hardness * 5000
		)

		high_noise = noise - low_noise

		# Very short attack.
		crack_envelope = np.exp(
			-t * (180 + hardness * 500)
		)

		crack = (
			high_noise * crack_envelope *
			(0.15 + hardness * 0.65)
		)

		# --------------------------------------------------
		# Low-frequency body
		# --------------------------------------------------

		body_frequency = (
			75 + density * 100
		)

		body_envelope = np.exp(
			-t * (22 + damping * 65)
		)

		body = np.sin(
			2 * np.pi * body_frequency * t
		)

		body *= body_envelope

		# --------------------------------------------------
		# Material resonance
		# --------------------------------------------------

		# Ground resonance.
		ground_decay = np.exp(
			-t * (25 + damping * 90)
		)

		ground = np.sin(
			2 * np.pi * frequency * t
		)

		ground *= ground_decay

		# Shoe resonance.
		shoe_decay = np.exp(
			-t * (35 + a.damping * 100)
		)

		shoe = np.sin(
			2 * np.pi * a.resonance * t
		)

		shoe *= shoe_decay

		# --------------------------------------------------
		# Surface texture
		# --------------------------------------------------

		texture = self._lowpass(
			noise,
			1000 + roughness * 7000
		)

		texture *= envelope * roughness * 0.12

		# --------------------------------------------------
		# Mix
		# --------------------------------------------------

		samples = (
			crack * 0.7 +
			body * 0.35 +
			ground * 0.18 +
			shoe * 0.08 +
			texture
		)

		# Apply impact strength.
		samples *= force

		# Soft saturation prevents excessive peaks.
		samples = np.tanh(samples * 1.5)

		# Normalize only if necessary.
		peak = np.max(np.abs(samples))

		if peak > 0:
			samples *= min(1.0, 0.9 / peak)

		return sd.audio(
			samples.astype(np.float32),
			sr
		)


# ============================================================
# FRICTION
# ============================================================

class Friction:

	def __init__(
		self,
		material_a,
		material_b,
		speed=1.0,
		force=1.0,
		sample_rate=44100,
	):

		self.a = material_a
		self.b = material_b
		self.speed = max(0, speed)
		self.force = max(0, force)
		self.sample_rate = sample_rate

	def generate(self, duration=0.5):

		count = int(duration * self.sample_rate)
		t = np.arange(count) / self.sample_rate

		roughness = (
			self.a.roughness +
			self.b.roughness
		) / 2

		damping = (
			self.a.damping +
			self.b.damping
		) / 2

		noise = np.random.normal(0, 1, count)

		# Roughness determines noise intensity.
		noise = np.diff(noise, prepend=0)

		# Modulation simulates irregular friction.
		modulation = (
			0.65 +
			0.35 * np.sin(
				2 * np.pi * (3 + self.speed * 5) * t
			)
		)

		envelope = np.sin(
			np.pi * np.clip(t / duration, 0, 1)
		) ** 0.5

		samples = (
			noise *
			roughness *
			(1 - damping * 0.5) *
			modulation *
			envelope *
			self.force
		)

		return sd.audio(samples, self.sample_rate)


# ============================================================
# SFX
# ============================================================

class SFX:

	sample_rate = 44100

	@classmethod
	def impact(
		cls,
		material_a,
		material_b,
		force=1.0,
		speed=1.0,
	):

		return Impact(
			material_a,
			material_b,
			force,
			speed,
			cls.sample_rate,
		).generate()

	@classmethod
	def friction(
		cls,
		material_a,
		material_b,
		speed=1.0,
		force=1.0,
		duration=0.5,
	):

		return Friction(
			material_a,
			material_b,
			speed,
			force,
			cls.sample_rate,
		).generate(duration)

	@classmethod
	def footsteps(
		cls,
		shoe,
		ground,
		weight=1.0,
		tempo=1.0,
		steps=4,
	):

		# Time between steps.
		interval = 0.5 / max(tempo, 0.01)

		parts = []

		for i in range(steps):

			# Slight variation between steps.
			variation = np.random.uniform(0.9, 1.1)

			audio = cls.impact(
				shoe,
				ground,
				force=weight * variation,
				speed=1.0,
			)

			parts.append((
				int(i * interval * cls.sample_rate),
				audio
			))

		# Combine all footsteps.
		total = max(
			position + len(samples)
			for position, samples in parts
		)

		result = np.zeros(total, dtype=np.float32)

		for position, samples in parts:
			end = position + len(samples)

			result[position:end] += samples

		return sd.audio(result, cls.sample_rate)

	@classmethod
	def punch(
		cls,
		fist,
		target,
		force=1.0,
	):

		return cls.impact(
			fist,
			target,
			force=force,
			speed=1.5,
		)


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == '__main__':
	# Walking on concrete.
	SFX.footsteps(
		shoe=Materials.sneaker,
		ground=Materials.concrete,
		weight=1.0,
		tempo=1.5,
	).write('footsteps_concrete.wav')


	# Heavy boots on wood.
	SFX.footsteps(
		shoe=Materials.boot,
		ground=Materials.wood,
		weight=2.0,
		tempo=1.0,
	).write('footsteps_wood.wav')


	# Punching wood.
	SFX.punch(
		fist=Materials.fist,
		target=Materials.wood,
		force=2.0,
	).write('punch_wood.wav')


	# Metal hitting concrete.
	SFX.impact(
		Materials.metal,
		Materials.concrete,
		force=3.0,
	).write('metal_impact.wav')


	# Sliding a sneaker on concrete.
	SFX.friction(
		Materials.sneaker,
		Materials.concrete,
		speed=2.0,
		duration=1.0,
	).write('friction.wav')


