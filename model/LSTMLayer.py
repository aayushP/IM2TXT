import numpy
import theano
import theano.tensor as tensor
import sys

sys.path.insert(0,'../')

import utils

class LSTMCell:
   def __init__(self,
               context,
               U,
               Wc,
               Wd_att, U_att, c_tt,
               W_sel, b_sel ):
      self.context = context
      self.U = U
      self.Wc = Wc

      self.Wd_att = Wd_att
      self.U_att = U_att
      self.c_att = c_tt

      self.W_sel = W_sel
      self.b_sel = b_sel

   def __call__(self, m_, x_, h_, c_, a_, ct_, pctx_):
      """
      Each variable is one time slice of the LSTM
      m_ - (mask),
      x_- (previous word),
      h_- (hidden state),
      c_- (lstm memory),
      a_ - (alpha distribution [eq (5)]),
      as_- (sample from alpha dist),
      ct_- (context),
      pctx_ (projected context),
      """

      # attention computation
      # [described in  equations (4), (5), (6) in
      # section "3.1.2 Decoder: Long Short Term Memory Network]
      # f_att(a_i, h_t-1) This part is concatenation between a_i and h_t-1
      pstate_ = tensor.dot(h_, self.Wd_att)
      pctx_ = pctx_ + pstate_[:,None,:]
      pctx_list = [pctx_]
      pctx_ = tensor.tanh(pctx_)

      #equation 4
      alpha = tensor.dot(pctx_, self.U_att)+self.c_att

      #Eqn 5
      alpha = tensor.nnet.softmax(alpha.reshape([alpha.shape[0],alpha.shape[1]])) # softmax
      #Eqn 6
      ctx_ = (self.context * alpha[:,:,None]).sum(1) # current context, aka zt in paper

      #selector
      sel_ = tensor.nnet.sigmoid(tensor.dot(h_, self.W_sel+self.b_sel))
      sel_ = sel_.reshape([sel_.shape[0]])
      ctx_ = sel_[:,None] * ctx_

      #Accumulation to compute different gate outputs
      preact = tensor.dot(h_, self.U)
      preact += x_
      preact += tensor.dot(ctx_, self.Wc)

      dim = self.U.shape[0]

      # Recover the activations to the lstm gates
      # [equation (1)]
      i = utils.slice(preact, 0, dim)
      f = utils.slice(preact, 1, dim)
      o = utils.slice(preact, 2, dim)

      i = tensor.nnet.sigmoid(i)
      f = tensor.nnet.sigmoid(f)
      o = tensor.nnet.sigmoid(o)
      c = tensor.tanh(utils.slice(preact, 3, dim))

      # compute the new memory/hidden state
      # if the mask is 0, just copy the previous state
      #Eqn 2
      c = f * c_ + i * c
      c = m_[:,None] * c + (1. - m_)[:,None] * c_

      #Eqn 3
      h = o * tensor.tanh(c)
      h = m_[:,None] * h + (1. - m_)[:,None] * h_

      return [h, c, alpha, ctx_]

class LSTMLayer:
   # shape[0] - (prev. "nin") word vector dimensionality
   # shape[1] - (prev. "dim") number of LSTM cells i.e. max length of sentence
   # shape[2] - (prev. "ctx_dim"/"dimctx") context vector dimensionality i.e. feature dimension

   def __init__(self, shape, name):

      # input to LSTM, similar to the above, we stack the matricies for compactness, do one
      # dot product, and use the slice function below to get the activations for each "gate"
      self.W = theano.shared(numpy.concatenate(
         [utils.norm_weight(shape[0],shape[1]),
          utils.norm_weight(shape[0],shape[1]),
          utils.norm_weight(shape[0],shape[1]),
          utils.norm_weight(shape[0],shape[1])
          ], axis=1), name=name+"_W")

      # LSTM to LSTM
      self.U = theano.shared(numpy.concatenate(
         [utils.ortho_weight(shape[1]),
          utils.ortho_weight(shape[1]),
          utils.ortho_weight(shape[1]),
          utils.ortho_weight(shape[1])
          ], axis=1), name=name+"_U")

      # bias to LSTM
      self.b = theano.shared(numpy.zeros((4 * shape[1],)).astype('float32').astype('float32'), name=name+"_b")

      # context to LSTM
      self.Wc = theano.shared(utils.norm_weight(shape[2], 4 * shape[1]), name=name+"_Wc")

      # attention: context -> hidden
      self.Wc_att = theano.shared(utils.norm_weight(shape[2], ortho=False),  name=name+"_Wc_att")

      # attention: LSTM -> hidden
      self.Wd_att = theano.shared(utils.norm_weight(shape[1],shape[2]), name=name+"_Wd_att")

      # attention: hidden bias
      self.b_att = theano.shared(numpy.zeros((shape[2],)).astype('float32'), name=name+"_b_att")

      # optional "deep" attention
      self.W_att_1 = theano.shared(utils.ortho_weight(shape[2]), name=name+"_W_att_1")
      self.b_att_1 = theano.shared(numpy.zeros((shape[2],)).astype('float32'), name=name+"_b_att_1")

      # attention:
      self.U_att = theano.shared(utils.norm_weight(shape[2], 1), name=name+"_U_att")
      self.c_att = theano.shared(numpy.zeros((1,)).astype('float32'), name=name+"_c_att")

      # attention: selector
      self.W_sel = theano.shared(utils.norm_weight(shape[1], 1), name=name+"_W_sel")
      self.b_sel = theano.shared(numpy.float32(0.), name=name+"_b_sel")


   def params(self):
      return [self.W, self.U, self.b, self.Wc, self.Wc_att, self.Wd_att, self.b_att, self.W_att_1, self.b_att_1,
              self.U_att, self.c_att, self.W_sel, self.b_sel]


   def __call__(self,
               input,
               context,
               init_memory=None,
               init_state=None,
               one_step=False,
               mask=None):

      assert input, 'Input must be provided'
      assert context, 'Input context must be provided'
      assert init_memory, 'previous memory must be provided'
      assert init_state, 'previous state must be provided'

      """
         input: the Embedding matrix
         n_timesteps:  the depth of the LSTM layer
         batch_size: the number of LSTM cells in each layer
      """

      # depth of the LSTM layer (how many LSTM cells in a sequence
      time_len = input.shape[0]

      if input.ndim == 3:
         batch_size = input.shape[1]
      else: # one step only
         batch_size = 1

      # mask
      if mask is None:
         mask = tensor.alloc(1., input.shape[0], 1)

      # infer lstm dimension
      dim = self.U.shape[0]

      # hidden activation of LSTM h_t is a linear projection of
      # the stochastic context vector z_t followed by tanh non-linearity
      pctx_ = tensor.dot(context, self.Wc_att) + self.b_att
      pctx_ = tensor.dot(pctx_, self.W_att_1) + self.b_att_1
      pctx_ = tensor.tanh(pctx_)

      # embedding is timesteps*num samples by d in training Ey_t-1
      # this is n * d during sampling
      embedding = tensor.dot(input, self.W) + self.b

      lstm_cell =LSTMCell(context,
         self.U,
         self.Wc,
         self.Wd_att, self.U_att, self.c_att,
         self.W_sel, self.b_sel )

      _step0 = lambda m_, x_, h_, c_, a_, ct_, pctx_: \
         lstm_cell(m_, x_, h_, c_, a_, ct_, pctx_)

      if one_step:
         rval = _step0(mask, embedding, init_state, init_memory, None, None, pctx_)
         return rval
      else:
         seqs = [mask, embedding]

         outputs_info = [init_state, init_memory,
                        tensor.alloc(0., batch_size, pctx_.shape[1]), #alpha
                        tensor.alloc(0., batch_size, context.shape[2])]  #ctx_

         #_step 0 function argument is concatenation of seqs, 3 arguments from output_info and non_sequences in that order
         rval, updates = theano.scan(_step0,
                                     sequences=seqs,
                                     outputs_info=outputs_info,
                                     non_sequences=[pctx_],
                                     name='decoder_layers',
                                     n_steps=time_len, profile=False)
         return rval