import numpy
import theano
import theano.tensor as tensor

import cPickle as pkl
from copy import copy
import utils
import time
import warnings
from network import Network
from Optimizers import RMSPropOptimizer
from utils import HomogeneousData
from sklearn.cross_validation import KFold

class Trainer:
   def __init__(self,
                train=None,
                validate=None,
                test=None,
                worddict=None,
                use_noise=None,
                nn_params=None,
                f_init=None,
                f_next=None,
                f_grad=None,
                f_update=None,
                f_probs=None,
                kf_valid=None,
                kf_test=None,
                save_network=None,
                options=None):

      self.__train = train
      self.__validate = validate
      self.__test = test
      self.__worddict = worddict
      self.use_noise=use_noise

      # index 0 and 1 always code for the end of sentence and unknown token
      # wworddict  arranges words/vocab in the order of frequency. highest frequent words come first.
      self.__word_idict = dict()
      for kk, vv in worddict.iteritems():
         self.__word_idict[vv] = kk
         self.__word_idict[0] = '<eos>'
         self.__word_idict[1] = 'UNK'

      self.options=options
      self.nn_params=nn_params

      # we need to pass in reference to model to invoke save_network
      self.save_network=save_network

      # theano functions
      self.kf_valid = kf_valid
      self.kf_test = kf_test
      self.f_init=f_init
      self.f_next = f_next
      self.f_update = f_update
      self.f_grad = f_grad
      self.f_probs= f_probs

      # history_errs is a bare-bones training log that holds the validation and test error
      self.history_errs=[]

      self.best_p = None
      self.bad_counter = 0

      # [See note in section 4.3 of paper]
      self.caption_batch_iterator = HomogeneousData(self.__train,
                                   batch_size=options['batch_size'],
                                   maxlen=options['maxlen'])

      self.update_index=0
      self.epoch_stop = False

   def print_sample(self, x, mask, ctx, captions):
      if numpy.mod(self.update_index, self.options['sampleFreq']) == 0:
         # turn off dropout first
         self.use_noise.set_value(0.)
         x_s = x
         mask_s = mask
         ctx_s = ctx
         # generate and decode the a subset of the current training batch
         for jj in xrange(numpy.minimum(10, len(captions))):
            sample, score = Model.generate_sample(self.f_init, self.f_next, ctx_s[jj], self.options, k=5, maxlen=30)

            # Decode the sample from encoding back to words
            print 'Truth ', jj, ': ',
            for vv in x_s[:, jj]:
               if vv == 0:
                  break
               if vv in self.__word_idict:
                  print self.__word_idict[vv],
               else:
                  print 'UNK',
            print
            for kk, ss in enumerate([sample[0]]):
               print 'Sample (', kk, ') ', jj, ': ',
               for vv in ss:
                  if vv == 0:
                     break
                  if vv in self.__word_idict:
                     print self.__word_idict[vv],
                  else:
                     print 'UNK',
            print

   # Return false to break the loop
   def log_validation_loss(self, epoch_index):
      options = self.options
      if numpy.mod(self.update_index, self.options['validFreq']) == 0:
         self.use_noise.set_value(0.)
         train_err = 0
         valid_err = 0
         test_err = 0

         if self.__validate:
            valid_err = -Model.predict_probs(self.f_probs, self.options, self.__worddict,
                                             utils.extract_input, self.__validate, self.kf_valid).mean()
         if self.__test:
            test_err = -Model.predict_probs(self.f_probs, self.options, self.__worddict,
                                            utils.extract_input, self.__test, self.kf_test).mean()

         self.history_errs.append([valid_err, test_err])

         # the model with the best validation long likelihood is saved seperately with a different name
         if self.update_index == 0 or valid_err <= numpy.array(self.history_errs)[:, 0].min():
            self.best_p = utils.getNNParams(self.nn_params)

            print 'Saving model with best validation ll'
            self.save_network(name='_bestll',best_p=self.best_p,update_index=self.update_index,history_errs=self.history_errs)
            self.bad_counter = 0

         # abort training if perplexity has been increasing for too long
         if epoch_index > options['patience'] and len(self.history_errs) > options['patience'] and valid_err >= numpy.array(self.history_errs)[
                                                                              :-options['patience'], 0].min():
            self.bad_counter += 1
            if self.bad_counter > options['patience']:
               print 'Early Stop!'
               self.epoch_stop = True

         print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

   def iterate(self, epoch_index, captions):
      options = self.options
      # preprocess the caption, recording the
      # time spent to help detect bottlenecks
      x, mask, ctx = utils.extract_input(captions,
                                         self.__train[1],
                                         self.__worddict,
                                         maxlen=options['maxlen'],
                                         n_words=options['n_words'])

      if x is None:
         print 'Minibatch with zero sample under length ', self.options['maxlen']
         return False

      # get the cost for the minibatch, and update the weights
      ud_start = time.time()
      cost = self.f_grad(x, mask, ctx)  # cost here is scalar value.
      self.f_update(options['lrate'])
      ud_duration = time.time() - ud_start  # some monitoring for each mini-batch

      # Numerical stability check
      if numpy.isnan(cost) or numpy.isinf(cost):
         print 'NaN detected'
         self.epoch_stop = True
         return False

      if numpy.mod(self.update_index, options['dispFreq']) == 0:
         print 'Epoch ', epoch_index, 'Update ', self.update_index, 'Cost ', cost, 'Time taken ', ud_duration

      # Checkpoint
      self.save_network(best_p=self.best_p,update_index=self.update_index,history_errs=self.history_errs)

      # Print a generated sample as a sanity check
      self.print_sample(x, mask, ctx, captions)

      # Log validation loss + checkpoint the model with the best validation log likelihood
      self.log_validation_loss(epoch_index)

      return True

   def run(self):
      options = self.options
      for epoch_index in xrange(options['max_epochs']):
         n_samples = 0

         print 'Epoch ', epoch_index

         for captions in self.caption_batch_iterator:
            n_samples += len(captions)
            self.update_index += 1

            # turn on dropout
            self.use_noise.set_value(1.)

            self.iterate(epoch_index, captions)

            if self.epoch_stop:
               break

         print 'Seen %d samples' % n_samples

         if self.epoch_stop:
            break


class Model:
   def __init__(self,
                train=None,
                validate=None,
                test=None,
                worddict=None,
                options=None):
      default_options = {
         "alpha_c":1.0,  # doubly stochastic coeff
         "alpha_entropy_c":0.002,  # hard attn param
         "attn_type":'deterministic',  # [see section 4 from paper]
         "batch_size":64,
         "ctx2out":True,  # Feed attention weighted ctx into logit
         "ctx_dim":512,  # context vector dimensionality
         "dataset":'flickr8k',
         "decay_c":0.0,  # weight decay coeff
         "dictionary":None,  # word dictionary
         "dim":1800,  # the number of LSTM units
         "dim_word":512,  # word vector dimensionality
         "dispFreq":1,
         "lrate":0.01,  # used only for SGD
         "lstm_encoder":False,  # if True, run bidirectional LSTM on input units
         "max_epochs":5000,
         "maxlen":100,  # maximum length of the description
         "n_layers_att":2,  # number of layers used to compute the attention weights
         "n_layers_init":2,  # number of layers to initialize LSTM at time 0
         "n_layers_lstm":1,  # number of lstm layers
         "n_layers_out":1,  # number of layers used to compute logit
         "n_words":10000,  # vocab size
         "optimizer":'rmsprop',
         "patience":10,
         "prev2out":True,  # Feed previous word into logit
         "RL_sumCost":True,  # hard attn param
         "sampleFreq":250,  # generate some samples after every sampleFreq updates
         "save_per_epoch":False, # this saves down the model every epoch
         "saveFreq":1000,  # save the parameters after every saveFreq updates
         "saveto":'caption_model',  # relative path of saved model file
         "semi_sampling_p":0.5,  # hard attn param
         "temperature":1.0,  # hard attn param
         "use_dropout":True,  # setting this true turns on dropout at various points
         "use_dropout_lstm":False,  # dropout on lstm gates
         "valid_batch_size":64,
         "validFreq":2000
      }
      self.__options = default_options
      self.__options.update(options)
      self.validate_options()

      self.__train = train
      self.__validate = validate
      self.__test = test

      self.__worddict = worddict
      self.__worddict[0] = '<eos>'
      self.__worddict[1] = 'UNK'

   @staticmethod
   def generate_sample(f_init, f_next, ctx0, options, k=1, maxlen=30):

      sample = []
      sample_score = []
      dead_k = 0
      prev_samples = [[]] * 1
      prev_scores = numpy.zeros(1).astype('float32')

      init = f_init(ctx0)
      ctx0 = init[0]
      next_state = [init[1]]
      next_memory = [init[2]]

      # reminder: if next_w = -1, the switch statement
      # in build_sampler is triggered -> (empty word embeddings)
      next_w = -1 * numpy.ones((1,)).astype('int64')

      for ii in xrange(maxlen):
         # our "next" state/memory in our previous step is now our "initial" state and memory
         next = f_next(*([next_w, ctx0] + [next_state] + [next_memory]))
         next_p = next[0]
         next_w = next[1]

         # extract all the states and memories
         next_state = next[2]
         next_memory = next[3]

         cand_scores = prev_scores[:, None] - numpy.log(next_p)
         cand_flat = cand_scores.flatten()
         ranks_flat = cand_flat.argsort()[:(k - dead_k)]  # (k-dead_k) numpy array of with min nll

         dict_size = next_p.shape[1]
         trans_indices = ranks_flat / dict_size
         word_indices = ranks_flat % dict_size
         costs = cand_flat[ranks_flat]  # extract costs from top hypothesis

         # a bunch of lists to hold future hypothesis
         curr_samples = []
         curr_scores = []
         curr_states = []
         curr_memories = []

         # get the corresponding hypothesis and append the predicted word
         for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            prev_sample = prev_samples[ti] + [wi]
            if(prev_sample[-1] == 0):
               sample.append(prev_sample)
               sample_score.append(copy(costs[idx]))
               dead_k += 1  # completed sample!
            else:
               curr_samples.append(prev_samples[ti] + [wi])
               curr_scores.append(copy(costs[idx]))
               curr_states.append(copy(next_state[ti]))
               curr_memories.append(copy(next_memory[ti]))

         live_k = k - dead_k
         # generated all the k best samples
         if live_k < 1 or dead_k >= k:
            break

         prev_samples = curr_samples
         prev_scores = numpy.array(curr_scores)
         next_w = numpy.array([w[-1] for w in curr_samples])
         next_state = numpy.array(curr_states)
         next_memory = numpy.array(curr_memories)

      # dump every remaining one
      if live_k > 0:
         for idx in xrange(live_k):
            sample.append(prev_samples[idx])
            sample_score.append(prev_scores[idx])

      return sample, sample_score

   @staticmethod
   def predict_probs(f_probs, options, worddict, prepare_data, data, iterator):
      # Get log probabilities of captions

      n_samples = len(data[0])
      probs = numpy.zeros((n_samples, 1)).astype('float32')

      for _, index in iterator:
         x, mask, ctx = prepare_data([data[0][t] for t in index], data[1],
                                     worddict, maxlen=None, n_words=options['n_words'])

         pred_probs = f_probs(x, mask, ctx)
         probs[index] = pred_probs[:, None]

      return probs

   def validate_options(self):
      # Put friendly reminders here
      if self.__options['dim_word'] > self.__options['dim']:
         warnings.warn('dim_word should only be as large as dim.')


      if self.__options['use_dropout_lstm']:
         warnings.warn('dropout in the lstm seems not to help')

      # Other checks:
      if self.__options['attn_type'] not in ['deterministic']:
         raise ValueError("specified attention type is not correct")

   def build_network(self):
      print 'Building network...'
      self.nn_network = Network(self.__options)
      return self.nn_network.build_training_graph(self.__options)

   def add_l2_regularization(self):
      # add L2 regularization costs
      nn_params = self.nn_network.params()
      self.__decay_c = theano.shared(numpy.float32(self.__options['decay_c']), name='decay_c')
      self.__weight_decay = 0.
      for vv in self.nn_network.params():
         self.__weight_decay += (vv ** 2).sum()
         self.__weight_decay *= self.__decay_c
         self.__cost += self.__weight_decay

   def add_doubly_stochastic_regularization(self, alphas):
      # Doubly stochastic regularization
      self.__alpha_c = theano.shared(numpy.float32(self.__options['alpha_c']), name='alpha_c')
      alpha_reg = self.__alpha_c * ((1. - alphas.sum(0)) ** 2).sum(0).mean()
      self.__cost += alpha_reg

   def optimize(self, input=None):
      # f_grad computes the cost and updates adaptive learning rate variables
      # f_update updates the weights of the model
      nn_params = self.nn_network.params()
      self.__lr = tensor.scalar(name='lr')
      rmsopt = RMSPropOptimizer(nn_params, self.__lr)
      return rmsopt.minimize(input, self.__cost)

   def save_network(self, name ='_snapshot', best_p=None,update_index=0,history_errs=[]):
      if numpy.mod(update_index, self.__options['saveFreq']) == 0:
         print 'Saving...',

         if best_p is not None:
            params = copy(best_p)
         else:
            params = utils.getNNParams(self.nn_network.params())
         numpy.savez(self.__options['saveto'] +name, history_errs=history_errs, **params)
         pkl.dump(self.__options, open('%s.pkl' % self.__options['saveto'], 'wb'))
         print 'Done'

   def train(self):
      options = self.__options
      use_noise, inps, alphas, cost = self.build_network()

      print('Buliding a sample inference')
      f_init, f_next = self.nn_network.infer()

      # we want the cost without any the regularizers
      f_probs = theano.function(inps, -cost, profile=False, updates=None)

      self.__cost = cost.mean()

      if self.__options['decay_c'] > 0.:
         self.add_l2_regularization()

      if self.__options['alpha_c'] > 0.:
         self.add_doubly_stochastic_regularization(alphas)

      # f_grad computes the cost and updates adaptive learning rate variables
      # f_update updates the weights of the model
      f_grad, f_update = self.optimize(input=inps)

      print 'Optimization'
      kf_test=None
      kf_valid=None

      if self.__validate:
         kf_valid = KFold(len(self.__validate[0]),
                          n_folds=len(self.__validate[0]) / options['valid_batch_size'],
                          shuffle=False)
      if self.__test:
         kf_test = KFold(len(self.__test[0]),
                         n_folds=len(self.__test[0]) / options['valid_batch_size'],
                         shuffle=False)

      trainer = Trainer(
         train=self.__train,
         validate=self.__validate,
         test=self.__test,
         worddict=self.__worddict,
         nn_params=self.nn_network.params(),
         use_noise=use_noise,
         f_init=f_init,
         f_next=f_next,
         f_grad=f_grad,
         f_update=f_update,
         f_probs=f_probs,
         kf_valid=kf_valid,
         kf_test=kf_test,
         save_network=self.save_network,
         options=options)

      trainer.run()

   def infer(self, config_path, path, image_files):

      self.build_network()
      f_init, f_next = self.nn_network.infer()

      nn_params = self.nn_network.params()
      params = numpy.load(path+'caption_model_bestll.npz')
      utils.setNNParams(params, nn_params)

      feat_maps = utils.get_feature_maps(config_path, image_files)
      for ctx_s in feat_maps:
         ctx_s = ctx_s.reshape(196, 512)
         sample, score = self.generate_sample(f_init, f_next, ctx_s, self.__options, k=5, maxlen=30)

         for kk, ss in enumerate([sample[0]]):
            print 'Sample (', kk, ') '
            for vv in ss:
               if vv == 0:
                  break
               if vv in self.__worddict:
                  print self.__worddict[vv],
               else:
                  print 'UNK',
