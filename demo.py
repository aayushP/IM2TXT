"""
Example execution script. The dataset parameter can
be modified to coco/flickr30k/flickr8k
"""
import argparse
from model.Model import Model
import DataLoader
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--attn_type",  default="deterministic",
                    help="type of attention mechanism")
parser.add_argument("changes",  nargs="*",
                    help="Changes to default values", default="")
def parse_args():
   # get updates from command line
   args = parser.parse_args()
   args_dict= dict({})
   for change in args.changes:
      args_dict.update(eval("dict({})".format(change)))
      # the supported commandline parameters
   keys = ("model",
           "attn-type",
           "reload",
           "dim-word",
           "ctx-dim",
           "dim",
           "n-layers-att",
           "n-layers-out",
           "n-layers-lstm",
           "n-layers-init",
           "n-words",
           "lstm-encoder",
           "decay-c",
           "alpha-c",
           "prev2out",
           "ctx2out",
           "learning-rate",
           "optimizer",
           "use-dropout",
           "save-per-epoch")

   filtered = dict((key, args_dict[key]) for key in [k for k in keys if k in args_dict])
   params = dict({})

   # convert dashes to underscore used in code
   for k,v in filtered:
      params[k.replace('-', '_')]=filtered[k]

   return params

def main(options):

   print 'Loading data...'
   train, valid, test, worddict = DataLoader.load_data()

   print 'Initializing model...'
   model = Model(train=train,
                 validate=valid,
                 test=test,
                 worddict=worddict,
                 options=options)

   image_files = glob.glob('demo/*.jpg')
   model_path ='./'
   config_path='data/'

   model.infer(config_path, model_path,image_files)

   #_, validerr, _ = model.train(options)
   #print "Final cost: {:.2f}".format(validerr.mean())

if __name__ == "__main__":
    defaults = {}
    main(parse_args())
