# Adapted from the code for "Backpropamine: training self-modifying 
# neural networks with differentiable neuromodulated plasticity" by
# Miconi et al., accessable at https://github.com/uber-research/backpropamine.


import argparse
import pdb
#from line_profiler import LineProfiler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import sys
import pickle
import time
import os
import platform

import numpy as np



# Maybe start with hard coded architecture, later make it general
class FFNetwork(nn.Module):

    def __init__(self, architecture):
        super(FFNetwork, self).__init__()
        self.activ = F.tanh

        self.input_layer_w = torch.nn.Linear(4, 64) #Use activation function?

        self.plastic_layer_w = torch.nn.Parameter(0.001 * torch.rand(64, 64))
        self.plastic_layer_alpha = torch.nn.Parameter(0.001 * torch.rand(64, 64))

        self.q_value_output_layer_w = torch.nn.Linear(64, 2)

        self.neuromod_output_layer_w = torch.nn.Linear(64, 1)
        # NOTE: one can also choose to not do the fanout, and just have the
        # same neuromodulatory signal value for each of the post-synaptic neurons.
        self.neuromod_fanout = torch.nn.Linear(1, 64)

    # Check with standard DQN
    #NOTE: the apparent convention of pytorch is row vectors. That is why
    # e.g. plastic_layer_w has dims nnl2 x nnl3 and not the other way around,
    # because pre_synaptic_activations is a matrix of BatchSize rows of vectors,
    # each with nnl2 dim: (BS x nnl2) x (nnl2 x nnl3) => BS x nnl3.
    def forward(self, inputs, hebb):

        # inputs: BatchSize x input_dims
        # pre_synaptic_activations: BatchSize x num_neurons_layer_2
        pre_synaptic_activations = self.activ(self.input_layer_w(inputs))

        # Batch size 1:
        #post_synaptic_activations = self.activ(torch.matmul((self.plastic_layer_w + 
        #                            torch.mul(self.plastic_layer_alpha, hebb)), pre_synaptic_activations))
        
        # post_synaptic_activations: BatchSize x num_neurons_layer_3
        # plastic_layer_w: num_neurons_layer_2 x num_neurons_layer_3
        # plastic_layer_alpha: num_neurons_layer_2 x num_neurons_layer_3
        # hebb: Batchsize x num_neurons_layer_2 x num_neurons_layer_3
        # Multiplying alpha and hebb means Batchsize number of matrix products,
        # resulting in Batchsize x num_neurons_layer_2 x num_neurons_layer_3.
        # Adding w to that means Batchsize number of additions, resulting in
        # Batchsize x num_neurons_layer_2 x num_neurons_layer_3.
        # pre_synaptic_activations unsqueezed gives BatchSize x 1 x num_neurons_layer_2,
        # which can be multiplied with BatchSize x num_neurons_layer_2 x num_neurons_layer_3
        post_synaptic_activations = self.activ(pre_synaptic_activations.unsqueeze(1).bmm(self.plastic_layer_w + torch.mul(self.plastic_layer_alpha, hebb)).squeeze(1))
        
        # Output the Q-values. BatchSize x num_actions
        q_values = self.q_value_output_layer_w(post_synaptic_activations)

        # Output the neuromodulatory signal. BatchSize x nm_sig_dims x 1
        # Unsqueeze, i.e. insert extra dimension of size 1 at the end,
        # so that broadcasted multiplication with deltahebb is possible later.
        neuromod_signal = self.activ(self.neuromod_output_layer_w(post_synaptic_activations)).unsqueeze(2)

        # Pass the neuromodulatory signal through a vector of fanout weights, one
        # weight for each of the neurons that the plastic weights feed into.
        # BatchSize x 1 x num_neurons_layer_3
        neuromod_signal = self.neuromod_fanout(neuromod_signal)


        # Calculate the Hebbian updates. This is an outer product of the vectors of
        # pre-synaptic and post-synaptic activations, giving a Hebbian term for each
        # combination of pre- and post-synaptic neurons, for each batch element.
        # Unsqueeze is used to make the pre-synaptic activation vectors column vectors,
        # and the post-synaptic activation vectors row vectors, the multiplication of which
        # is an outer product.
        # BatchSize x num_neurons_layer_2 x num_neurons_layer_3
        delta_hebb = torch.bmm(pre_synaptic_activations.unsqueeze(2), post_synaptic_activations.unsqueeze(1))

        # Update the Hebbian traces. Note that the below multiplication is broadcasted,
        # i.e. the same neuromod_signal row vector is element-wise multiplied with
        # each row of delta_hebb, going from BS x 1 x nnl3 and BS x nnl2 x nnl3 to
        # BS x nnl2 x nnl3. In this way, the neuromodulatory signal value for each post-
        # synaptic cell's inputs are the same, while they are different for each pre-
        # synaptic cell's outputs.
        hebb = torch.clamp(hebb + neuromod_signal*delta_hebb, min = -1.0, max = 1.0)

        # Return the estimated Q-values and the updated Hebbian traces.
        return q_values, hebb





        