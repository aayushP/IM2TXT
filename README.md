# Image to Text using Attention


Implementation of ["Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"](https://arxiv.org/pdf/1502.03044.pdf)

## Setup Environment
This demo uses the [Theano](http://www.deeplearning.net/software/theano/) framework. 

### [Anaconda](https://www.anaconda.com/download/)
We highly recommend using the Anaconda platform to manage all the python dependencies and virtual environments. Otherwise, you will have to manually install each of theano's dependencies.

### [Installing Theano](http://www.deeplearning.net/software/theano/)
After Anaconda is installed, run 

`condo install theano`

## Running the code
### Data
* Download the data and annotations ([Flickr8k](https://illinois.edu/fb/sec/1713398), [Flickr30k](https://illinois.edu/fb/sec/229675), [Coco](http://mscoco.org/dataset/#download), etc)
* Modify data/data_generation_params.json file to indicate the location of annotation file and Image dataset.
* Resize the image to 224x224x3
* Run the data/data_generation.py to generate image and annotation pickles.

### Training
` python train.py`

### Demo

`python demo.py`

### Results
![A group of people stand together .](https://github.com/aayushP/IM2TXT/blob/master/demo/146098876_0d99d7fb98.jpg)
A group of people stand together.


![A girl is in a field .](https://github.com/aayushP/IM2TXT/blob/master/demo/174466741_329a52b2fe.jpg)
A girl is in a field.


![Dog running in field](https://github.com/aayushP/IM2TXT/blob/master/demo/240696675_7d05193aa0.jpg)
Dog running in field



### Acknowledgements:

 * [arctic-captions](https://github.com/kelvinxu/arctic-captions) for code/reference
 * [deep-learning tutorial](http://deeplearning.net/tutorial/code/) for code/reference






