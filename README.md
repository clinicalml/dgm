# Deep Generative Model
This contains the Torch Implementation of a Deep Generative Model trained using Stochastic Backpropagation. The model is trained on the GPU with binarized MNIST.

# Requirements

Torch installed with cunn 

Some miscellaneous lua packages installed from:
https://raw.githubusercontent.com/rahulk90/helper-files/master/install_torch_deps.sh

# Learning the Model

Run with starter.lua

Specify options as: -option optionvalue to modify default settings. 

Creates a folder "checkpoint" with model details and samples every 100 epochs. 

Run: th visualize.lua -path ./checkpoint -file filename to visualize samples stored in filename

# References:
----------
Auto-Encoding Variational Bayes
http://arxiv.org/abs/1312.6114

Stochastic Backpropagation and Approximate Inference in Deep Generative Models
http://arxiv.org/abs/1401.4082

Some code adopted from:
https://github.com/y0ast/VAE-Torch

Also contains code adopted from:
https://github.com/karpathy/char-rnn

