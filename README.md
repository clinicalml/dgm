# Branch Notes:
* `train.lua` = baseline
* `train_planarflow.lua` = normalizing flow
* in both training scripts, make sure to adjust paths and model params accordingly (for examples, see `exp1_baseline.lua`, `exp3_planarflow5.lua`, and `exp5_planarflow5deep.lua`)
* modified utils.lua so that data directory is adjustable
* `analyze_results.ipynb` and `visualize.ipynb` are useful for checking experiment results
* see `check_all.lua` and `check_PlanarFlow.lua` for scripts to verify calculations in `PlanarFlow.lua` and `GaussianReparam_normflow.lua`


# Deep Generative Model
This contains the Torch Implementation of a Deep Generative Model trained using Stochastic Backpropagation. The model is trained on the GPU with binarized MNIST.

# Requirements

Torch installed with cunn 

Some miscellaneous lua packages installed from:
https://raw.githubusercontent.com/rahulk90/helper-files/master/install_torch_deps.sh

# Learning the Model

Run th train.lua

Uses display (https://github.com/szym/display) to show samples and reconstruction to a browser display

# References
Auto-Encoding Variational Bayes
http://arxiv.org/abs/1312.6114

Stochastic Backpropagation and Approximate Inference in Deep Generative Models
http://arxiv.org/abs/1401.4082

See also:
https://github.com/y0ast/VAE-Torch
