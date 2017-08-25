import theano
import theano.tensor as tensor
from utils import TheanoFunctionWrapper as TFW

import numpy


class Optimizer(object):
	"""
		Base Optimizer
		Usage:
		opt = Optimizer(network_params)
		f_grad, f_update = opt.minimize(input, loss)
	"""
	def __init__(self, network_params, learning_rate=0.01):
		"""
		Args:
			network_params: List. params used by the entire network
			learning_rate: theano shared variable. learning rate, currently only necessaary for sgd
		"""
		self.learning_rate = learning_rate
		self.network_params = network_params

	def compute_gradients(self,loss):
		return tensor.grad(loss, wrt=self.network_params)

	def get_grad_update_function(self, grads):
		# Implemted by inherited class
		return NotImplemented

	def get_grad_apply_function(self, grads):
		# Implemted by inherited class
		return NotImplemented

	def minimize(self, input, loss):
		self.grads = self.compute_gradients(loss)
		# computes costs and learning rates
		f_grad=TFW(input, loss, updates=self.get_grad_update_function(), profile=False)
		f_update=TFW([self.learning_rate],[], updates=self.get_grad_apply_function(), on_unused_input='ignore')

		return f_grad, f_update


class SGDOptimizer(Optimizer):
	# Override any default values and define any shared
	# variables specific to this optimizer here
	def __init__(self,  network_params, hard_attn_up=None, learning_rate=0.0002):
		super(SGDOptimizer, self).__init__(network_params, learning_rate)
		self.gshared = [theano.shared(p.get_value() * numpy.float32(0.)) for p in self.network_params]
		self.hard_attn_up = hard_attn_up

	def get_grad_update_function(self):
		gsup = [(gs, g) for gs, g in zip(self.gshared, self.grads)]
		return gsup+self.hard_attn_up

	def get_grad_apply_function(self):
		return [(p, (p - self.learning_rate * g).astype('float32'))
				  for p, g in zip(self.network_params, self.gshared)]

class RMSPropOptimizer(Optimizer):
	# Override any default values here
	def __init__(self,  network_params, learning_rate=0.001):
		super(RMSPropOptimizer, self).__init__(network_params, learning_rate)
		self.zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.)) for p in self.network_params]
		self.running_grads = [theano.shared(p.get_value() * numpy.float32(0.)) for p in self.network_params]
		self.running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.)) for p in self.network_params]

	def get_grad_update_function(self):
		# currently casting any arithmetic operation with astype to float32
		# if this doesnt work, we need to expand the list comprehensions to for loops

		zgup = [(zg, g) for zg, g in zip(self.zipped_grads, self.grads)]
		rgup = [(rg, (0.95 * rg + 0.05 * g).astype('float32')) for rg, g in zip(self.running_grads, self.grads)]
		rg2up = [(rg2, (0.95 * rg2 + 0.05 * (g ** 2)).astype('float32'))
					for rg2, g in zip(self.running_grads2, self.grads)]

		return zgup+rgup+rg2up

	def get_grad_apply_function(self):
		updir = [theano.shared(p.get_value() * numpy.float32(0.)) for p in self.network_params]
		updir_new = [(ud, (0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)).astype('float32'))
						 for ud, zg, rg, rg2 in zip(updir, self.zipped_grads, self.running_grads, self.running_grads2)]
		param_up = [(p, (p + udn[1]).astype('float32')) for p, udn in zip(self.network_params, updir_new)]
		return updir_new+param_up


class AdamOptimizer(Optimizer):
	# Override any default values and define any shared
	# variables specific to this optimizer here
	def __init__(self,  network_params, learning_rate=0.0002):
		super(AdamOptimizer, self).__init__(network_params, learning_rate)
		self.gshared = [theano.shared(p.get_value() * numpy.float32(0.)) for p in self.network_params]

	def get_grad_update_function(self):
		return [(gs, g) for gs, g in zip(self.gshared, self.grads)]

	def get_grad_apply_function(self):
		lr0 = self.learning_rate
		b1 = 0.1
		b2 = 0.001
		e = 1e-8
		updates = []
		i = theano.shared(numpy.float32(0.))
		i_t = i + 1.
		fix1 = 1. - b1**(i_t)
		fix2 = 1. - b2**(i_t)
		lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

		# TODO: Check whether ".values()" is necessary for self.network_params
		for p, g in zip(self.network_params.values(), self.gshared):
			m = theano.shared(p.get_value() * numpy.float32(0.))
			v = theano.shared(p.get_value() * numpy.float32(0.))
			m_t = (b1 * g) + ((1. - b1) * m)
			v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
			g_t = m_t / (tensor.sqrt(v_t) + e)
			p_t = p - (lr_t * g_t)
			updates.append((m, m_t.astype('float32')))
			updates.append((v, v_t.astype('float32')))
			updates.append((p, p_t.astype('float32')))

		updates.append((i, i_t))
		return updates
