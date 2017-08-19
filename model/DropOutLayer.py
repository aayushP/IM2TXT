import theano.tensor as tensor

class DropOutLayer:
	def __init__(self, use_noise, trng):
		self.use_noise = use_noise
		self.trng = trng

	def __call__(self, input):
		proj = tensor.switch(self.use_noise, input * self.trng.binomial(input.shape, p=0.5, n=1, dtype=input.dtype), input * 0.5)
		return proj
