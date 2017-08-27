import cPickle as pkl
import numpy
import copy

import theano

from collections import OrderedDict
import sys
sys.path.insert(0,'data/')

from data_generation import load_config, build_graph, load_images

def ortho_weight(ndim):
    """
    Random orthogonal weights

    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

def get_feature_maps(config_path, image_files):
    _, _, vgg19_model_path, _, _, _, _, _ = load_config(config_path)

    vgg, pl_images = build_graph(vgg19_model_path)

    return load_images(image_files, vgg, pl_images)

def extract_input(caps, features, worddict, maxlen=None, n_words=10000, zero_pad=False):
    # x: a list of sentences
    seqs = []
    feat_list = []
    for cc in caps:
            seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc[0].split()])
            elem = features[cc[1]]
            feat_list.append(elem)

    lengths = [len(s) for s in seqs]

    if maxlen != None and numpy.max(lengths) >= maxlen:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, feat_list):
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    y = numpy.zeros((len(feat_list), feat_list[0].shape[1])).astype('float32')
    for idx, ff in enumerate(feat_list):
        y[idx,:] = numpy.array(ff.todense())

    y = y.reshape([y.shape[0], 196, 512])

    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s #transpose of seqs. why god?
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y

# push parameters to Theano shared variables
def setNNParams(params, nn_params):
    for nn_p in nn_params:
        nn_p.set_value(params[nn_p.name])

def getNNParams(nn_params):
    params = OrderedDict()
    for nn_p in nn_params:
        params[nn_p.name] = nn_p.get_value()
    return params

class HomogeneousData():
    def __init__(self, data, batch_size=128, maxlen=None):
        self.batch_size = 128
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.prepare()
        self.reset()

    def prepare(self):
        self.caps = self.data[0]
        self.feats = self.data[1]

        # find the unique lengths
        self.lengths = [len(cc[0].split()) for cc in self.caps]
        self.len_unique = numpy.unique(self.lengths)
        # remove any overly long captions
        if self.maxlen:
            self.len_unique = [ll for ll in self.len_unique if ll <= self.maxlen]

        # indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])

        # current counter
        self.len_curr_counts = copy.copy(self.len_counts)

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = numpy.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def next(self):
        # randomly choose the length
        count = 0
        while True:
            self.len_idx = numpy.mod(self.len_idx+1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        # get the batch size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
        # get the indices for the current batch
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
        self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size

        caps = [self.caps[ii] for ii in curr_indices]

        return caps

    def __iter__(self):
        return self


class TheanoFunctionWrapper():
    def __init__(self, input_variables, output_variables, **kwargs):
        self.__output_dict = None

        if isinstance(output_variables, dict):
            outputs = []
            self.__output_dict = {}
            for k, v in output_variables.iteritems():
                outputs.append(v)
                self.__output_dict[k] = None
        else:
            outputs = output_variables

        self.theano_function = theano.function(input_variables, outputs, **kwargs)

    def __call__(self, *inputs, **kwargs):
        self.theano_results = self.theano_function(*inputs, **kwargs)
        if self.__output_dict is not None:
            for index, key in enumerate(self.__output_dict.iteritems()):
                self.__output_dict[key] = self.theano_results[index]
            return self.__output_dict
        else:
            return self.theano_results

