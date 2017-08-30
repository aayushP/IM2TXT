import pandas as pd
import numpy as np
import os
import scipy
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import vgg19
import skimage
import skimage.io
from collections import OrderedDict
import json

batch_size = 100
feat_len = 196 * 512

def load_config(config_path=None):
    ##load the parameters for data generation from data_generation_params.json
    with open(config_path+'data_generation_params.json') as data_file:
        data_gen_params = json.load(data_file)
        dataset = data_gen_params["current_dataset"]
        print ('Using '+dataset+' for pkl generation')


    TOTAL_SIZE = data_gen_params[dataset]["total_size"]
    TRAIN_SIZE = data_gen_params[dataset]["training_size"]
    TEST_SIZE = data_gen_params[dataset]["test_size"]
    DEV_SIZE = TOTAL_SIZE-TRAIN_SIZE-TEST_SIZE

    annotation_path = data_gen_params[dataset]["paths"]["annotations"]
    image_path = data_gen_params[dataset]["paths"]["images"]
    pkl_output_path = data_gen_params[dataset]["paths"]["pkl_output"]
    vgg19_model_path = data_gen_params["vgg19_path"]

    return dataset, pkl_output_path, vgg19_model_path, annotation_path, image_path, TRAIN_SIZE, TEST_SIZE, DEV_SIZE

def load_annotations(annotation_path, image_path):

    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
    annotations['image'] = annotations['image'].map(lambda x: os.path.join(image_path,x.split('#')[0]))
    captions = annotations['caption'].values
    images = pd.Series(annotations['image'].unique())
    image_id_dict = pd.Series(np.array(images.index), index=images)#Redundant Code
    caption_image_id = annotations['image'].map(lambda x: image_id_dict[x]).values
    cap = zip(captions, caption_image_id)


    # split up into train, test, and dev
    all_idx = range(len(images))
    np.random.shuffle(all_idx)
    train_idx = all_idx[0:TRAIN_SIZE]
    test_idx = all_idx[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
    dev_idx = all_idx[TRAIN_SIZE+TEST_SIZE:TRAIN_SIZE+TEST_SIZE+DEV_SIZE]

    train_ext_idx = [i for idx in train_idx for i in xrange(idx*5, (idx*5)+5)]
    test_ext_idx = [i for idx in test_idx for i in xrange(idx*5, (idx*5)+5)]
    dev_ext_idx = [i for idx in dev_idx for i in xrange(idx*5, (idx*5)+5)]



    return images, captions, train_idx, test_idx, dev_idx

def create_dictionary_pickle(pkl_output_path, captions):
    ##create dictionary pkl
    vectorizer = CountVectorizer(analyzer=str.split, lowercase=False).fit(captions)
    dictionary = vectorizer.vocabulary_
    dictionary_series = pd.Series(dictionary.values(), index=dictionary.keys()) + 2
    dictionary = dictionary_series.to_dict()
    # Sort dictionary in descending order
    dictionary = OrderedDict(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))
    with open(pkl_output_path+'/'+'dictionary.pkl', 'wb') as f:
        cPickle.dump(dictionary, f, 2)


def load_images(image_files, vgg, pl_images):
  dataset = np.ndarray(shape=(len(image_files), feat_len), dtype=np.float32)
  image_index = 0
  for image in image_files:
    try:
        if not tf.gfile.Exists(image):
            tf.logging.fatal('File does not exist %s', image)
        image_data = skimage.io.imread(image)
        image_data = image_data / 255.0
        batch = np.ndarray(shape=(1, image_data.shape[0], image_data.shape[1], image_data.shape[2]), dtype=np.float32)
        batch[0, :, :, :] = image_data
        feed_dict = {pl_images: batch}

        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                feat = sess.run(vgg.conv5_4, feed_dict=feed_dict)

        feat.resize(feat_len,refcheck=False)
        dataset[image_index, :] = feat
        image_index += 1

    except IOError as e:
      print('Could not read:', image, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:image_index, :]

  print('Full dataset tensor:', dataset.shape)
  return dataset



def create_image_pkl(pkl_output_path, images, captions, set_idx, size, filename, access_mode, vgg, pl_images):

    #select images and captions
    set_ext_idx = [i for idx in set_idx for i in xrange(idx*5, (idx*5)+5)]
    images_set = images[set_idx]
    captions_set = captions[set_ext_idx]

    #reindex the images
    images_set.index = xrange(size)
    image_id_dict_set = pd.Series(np.array(images_set.index), index= images_set)

    #create list of image ids corresponding to each caption
    caption_image_id_set = [image_id_dict_set[img] for img in images_set for i in xrange(5)]
    # Create tuples of caption and image id
    cap_set = zip(captions_set, caption_image_id_set)

    for start, end in zip(range(0, len(images_set) + 100, 100), range(100, len(images_set) + 100, 100)):
        image_files = images_set[start:end]

        feat = load_images(image_files, vgg, pl_images)

        if start == 0:
            feat_flatten_list_set = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
        else:
            feat_flatten_list_set = scipy.sparse.vstack(
                [feat_flatten_list_set, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

        print "processing images %d to %d" % (start, end)

    with open(pkl_output_path+'/'+filename, access_mode) as f:
        cPickle.dump(cap_set, f, 2)
        cPickle.dump(feat_flatten_list_set, f, 2)


def build_graph(vgg19_model_path):
    pl_images = tf.placeholder("float", [1, 224, 224, 3])
    vgg = vgg19.Vgg19(vgg19_model_path)
    with tf.name_scope("content_vgg"):
        vgg.build(pl_images)

    return vgg, pl_images

def create_pickles(dataset, TRAIN_SIZE, TEST_SIZE, DEV_SIZE, train_idx, test_idx, dev_idx, pkl_output_path, images, captions, vgg, pl_images):
    #create dictionary pickle
    create_dictionary_pickle(pkl_output_path, captions)

    ##Create pkl for train, test, dev
    create_image_pkl(pkl_output_path, images, captions, train_idx, TRAIN_SIZE, dataset+'_align.train.pkl', 'wb', vgg, pl_images)
    create_image_pkl(pkl_output_path, images, captions, test_idx, TEST_SIZE, dataset+'_align.test.pkl', 'wb', vgg, pl_images)
    create_image_pkl(pkl_output_path, images, captions, dev_idx, DEV_SIZE, dataset+'_align.dev.pkl', 'wb', vgg, pl_images)

def get_feature_maps(config_path, image_files):
    _, _, vgg19_model_path, _, _, _, _, _ = load_config(config_path)

    vgg, pl_images = build_graph(vgg19_model_path)

    return load_images(image_files, vgg, pl_images)

if __name__ == "__main__":

    dataset, pkl_output_path, vgg19_model_path, annotation_path, \
            image_path, TRAIN_SIZE, TEST_SIZE, DEV_SIZE = load_config()

    images, captions, train_idx, test_idx, dev_idx = load_annotations(annotation_path, image_path)

    vgg, pl_images = build_graph(vgg19_model_path)

    create_pickles(dataset, TRAIN_SIZE, TEST_SIZE, DEV_SIZE, train_idx, test_idx, dev_idx, pkl_output_path, images, captions, vgg, pl_images)