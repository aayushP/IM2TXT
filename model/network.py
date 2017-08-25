import theano
import theano.tensor as tensor
from utils import TheanoFunctionWrapper as TFW
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy

from FeedForwardLayer import FFLayer
from LSTMLayer import LSTMLayer
from DropOutLayer import DropOutLayer


class Network:
    def __init__(self, options):
        ctx_dim = options['ctx_dim']
        dim = options['dim']
        dim_word = options['dim_word']
        n_words = options['n_words']

        self.scale = 0.01
        self.Wemb = theano.shared((self.scale * numpy.random.randn(n_words, dim_word)).astype('float32'), name='Wemb')
        self.trng = RandomStreams(1234)
        self.use_noise = theano.shared(numpy.float32(0.))

        self.FFInit = FFLayer(shape=[ctx_dim, ctx_dim], name='ff_init')
        self.FFState = FFLayer(shape=[ctx_dim, dim], name='ff_state')
        self.FFMemory = FFLayer(shape=[ctx_dim, dim], name='ff_memory')

        self.LSTMLayer = LSTMLayer(shape=[dim_word, dim, ctx_dim], name='decoder')

        self.FFLSTM = FFLayer(shape=[dim, dim_word], name='ff_logit_lstm')
        self.FFCtx = FFLayer(shape=[ctx_dim, dim_word], name='ff_logit_ctx')
        self.FFLogit = FFLayer(shape=[dim_word, n_words], name='ff_logit')

        self.Layers = [
            self.FFInit, self.FFState, self.FFMemory,
            self.LSTMLayer, self.FFLSTM, self.FFCtx, self.FFLogit]

        self._params = sum([layer.params() for layer in self.Layers],[self.Wemb])

        self.dropOutInit = DropOutLayer(self.use_noise, self.trng)
        self.dropOutLSTM = DropOutLayer(self.use_noise, self.trng)
        self.dropOutLogit = DropOutLayer(self.use_noise, self.trng)

    def params(self):
        return self._params

    def infer_init(self, ctx_mean):
        ctx_mean = self.FFInit(ctx_mean, activation='relu')
        ctx_mean = self.dropOutInit(ctx_mean)

        init_state = self.FFState(ctx_mean, activation='tanh')
        init_memory = self.FFMemory(ctx_mean, activation='tanh')

        return init_state, init_memory

    def infer_main(self, ctx, emb = None, mask =None,
                   init_state=None, init_memory=None,one_step=False):

        output_state = self.LSTMLayer(emb,ctx,init_memory, init_state, one_step, mask)
        output_state_h = self.dropOutLSTM(output_state[0])
        logit = self.FFLSTM(output_state_h, activation = 'linear')
        # prev2out
        logit += emb
        # ctx2out
        logit += self.FFCtx(output_state[3], activation='linear')
        logit = tensor.tanh(logit)
        logit = self.dropOutLogit(logit)
        logit = self.FFLogit(logit, activation='linear')

        return output_state, logit

    def build_training_graph(self, options):
        # description string: #words x #samples,
        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype='float32')
        # context: #samples x #annotations x dim
        ctx = tensor.tensor3('ctx', dtype='float32')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # index into the word embedding matrix, shift it forward in time
        #n_timesteps == caption length. n_samples = number of captions.
        emb = self.Wemb[x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        # initial state/cell [top right on page 4]
        ctx_mean = ctx.mean(1)
        init_state, init_memory = self.infer_init(ctx_mean)

        output_state, logit = self.infer_main(ctx=ctx, emb=emb, mask=mask, init_state=init_state,
                                              init_memory=init_memory, one_step=False)

        logit_shp = logit.shape
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # Index into the computed probability to give the log likelihood
        x_flat = x.flatten()
        p_flat = probs.flatten()
        cost = -tensor.log(p_flat[tensor.arange(x_flat.shape[0])*probs.shape[1]+x_flat]+1e-8)
        cost = cost.reshape([x.shape[0], x.shape[1]])
        masked_cost = cost * mask
        cost = (masked_cost).sum(0)

        alphas = output_state[2]

        return self.use_noise, [x, mask, ctx], alphas, cost


    def infer(self):

        # context: #annotations x dim
        ctx = tensor.matrix('ctx_sampler', dtype='float32')
        x = tensor.vector('x_sampler', dtype='int64')

        # initial state/cell
        ctx_mean = ctx.mean(0)
        init_state, init_memory = self.infer_init(ctx_mean)

        f_init = TFW([ctx],
                     {'context':ctx, 'state':init_state, 'memory':init_memory},
                     name='f_init', profile=False)

        init_state = tensor.matrix('init_state', dtype='float32')
        init_memory = tensor.matrix('init_memory', dtype='float32')

        # for the first word (which is coded with -1), emb should be all zero
        emb = tensor.switch(x[:,None] < 0, tensor.alloc(0., 1, self.Wemb.shape[1]), self.Wemb[x])

        output_state, logit = self.infer_main(ctx=ctx, emb=emb, mask=None, init_state=init_state,
                                              init_memory=init_memory, one_step=True)

        next_probs = tensor.nnet.softmax(logit)
        next_sample = self.trng.multinomial(pvals=next_probs).argmax(1)

        next_state, next_memory = output_state[0], output_state[1]

        f_next = TFW([x, ctx , init_state, init_memory],
                     {'probs':next_probs, 'sample':next_sample, 'state':next_state, 'memory':next_memory},
                     name='f_next', profile=False)

        return f_init, f_next


