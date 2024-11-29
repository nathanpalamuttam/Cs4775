### Basset
###### Deep convolutional neural networks for DNA sequence analysis.
--------------------------------------------------------------------------------

To train a model, you first need to convert your sequences into HDF5 format for Torch. Check out my tutorials for how to do that; they're linked from the [main page](../README.md).

We'll use the script *basset_train.lua* to train. I'll review the options to that script, and discuss the decisions you need to make for each one.

###### -cuda
First, you need to decide whether to compute on your CPU or GPU. GPU stands for graphics processing unit, which have proven tremendously useful for certain computations that intersect with display...you guessed it- graphics. Here's a [Udacity course](https://www.udacity.com/course/intro-to-parallel-programming--cs344) where you can learn more. Training on the GPU is MUCH faster, and is effectively required for training a complex model on anything beyond 100k sequences. Beyond a million sequences, you'll either need a high end GPU or patience. Move training to your GPU with the -cuda option.

###### -job
Next, you need to design the model architecture, choose optimization parameters, and write them to a table text file. At this stage of rapid artifical neural network research, this is equal parts art and science. I provide the best architecture and optimization parameters that I found for the DNaseI-seq compendium analyzed in the paper [here](../data/models/pretrained_params.txt), which can also serve as an example for the required format. There's no doubt that better parameters exist, so explore away! [Bayesian optimization](http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf) is a fantastic approach for doing so, and [Spearmint](https://github.com/HIPS/Spearmint) can help you do that. Jasper and I can offer more advice here for ambitious users if you send me an email.

###### -stagnant_t
Assuming you carefully set aside some validation data, this option specifies how many training epochs to carry out without improvement in the validation loss before halting. If you expect to overfit, a small value here will save you a lot of unnecessary computation.

###### -save
Basset will save checkpoints and the model corresponding to the best validation loss that it's seen. (These are large so once you know you don't want them, delete them.) Specify the filename prefix with this option.

At this point, you know enough to train. Run as
```
basset_train.lua -cuda -job pretrained_params.txt -stagnant_t 10 all_data_ever.h5
```

###### -restart
Sometimes you'll need to stop your training job, but want to restart it. No problem. Basset writes a checkpoint after every epoch, so just provide the latest with this option.

###### -seed
As I demonstrate in the paper, seeding a training run with the parameters from my pretrained model can be a powerful approach to learn on smaller, new datasets. Provide the model with this option.