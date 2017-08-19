import numpy
import theano
import theano.tensor as tensor
import utils


class FFLayer:
    def __init__(self, shape, name):
        self.scale =0.01
        self.W = theano.shared((self.scale * numpy.random.randn(shape[0], shape[1])).astype('float32'), name=name+"_W")
        self.b = theano.shared(numpy.zeros(shape[1],).astype('float32'), name=name+"_b")

    def params(self):
        return [self.W, self.b]

    def __call__(self, input, activation):
        output = tensor.dot(input, self.W)+ self.b
        if activation == 'relu':
            return tensor.maximum(0., output)
        elif activation == 'tanh':
            return tensor.tanh(output)
        else:
            return output