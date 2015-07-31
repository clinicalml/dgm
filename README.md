# Deep Generative Model
This contains the Torch Implementation of a Deep Generative Model trained using Stochastic Backpropagation. The model is trained on the GPU with binarized MNIST.

# Requirements

Torch installed with cunn 

Some miscellaneous lua packages installed from:
https://raw.githubusercontent.com/rahulk90/helper-files/master/install_torch_deps.sh

# Learning the Model

Run th train.lua

Uses display (https://github.com/szym/display) to show samples and reconstruction to a browser display

# References:
----------
Auto-Encoding Variational Bayes
http://arxiv.org/abs/1312.6114

Stochastic Backpropagation and Approximate Inference in Deep Generative Models
http://arxiv.org/abs/1401.4082

Some code adopted from:
https://github.com/y0ast/VAE-Torch
