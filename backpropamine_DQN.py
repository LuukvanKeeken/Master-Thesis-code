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
class BP_FFNetwork(nn.Module):

    def __init__(self, input_dims, pre_plastic_dims, post_plastic_dims, num_actions, seed):
        super(BP_FFNetwork, self).__init__()
        
        # Is all of this really needed?
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.input_dims = input_dims
        self.pre_plastic_dims = pre_plastic_dims
        self.post_plastic_dims = post_plastic_dims

        self.activ = torch.tanh

        self.input_layer_w = torch.nn.Linear(input_dims, pre_plastic_dims) #Use activation function?

        self.plastic_layer_w = torch.nn.Parameter(0.001 * torch.rand(pre_plastic_dims, post_plastic_dims))
        self.plastic_layer_alpha = torch.nn.Parameter(0.001 * torch.rand(pre_plastic_dims, post_plastic_dims))

        self.q_value_output_layer_w = torch.nn.Linear(post_plastic_dims, num_actions)

        self.neuromod_output_layer_w = torch.nn.Linear(post_plastic_dims, 1)
        # NOTE: one can also choose to not do the fanout, and just have the
        # same neuromodulatory signal value for each of the post-synaptic neurons.
        self.neuromod_fanout = torch.nn.Linear(1, post_plastic_dims)

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


    # Initialize the Hebbian traces.
    def initialZeroHebb(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.pre_plastic_dims, self.post_plastic_dims), requires_grad = False)




# RNN with trainable modulated plasticity ("backpropamine")
class BP_RNetwork(nn.Module):
    
    def __init__(self, isize, hsize, num_actions, seed): 
        super(BP_RNetwork, self).__init__()

        # Is all of this really needed?
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.hsize, self.isize  = hsize, isize 

        self.i2h = torch.nn.Linear(isize, hsize)    # Weights from input to recurrent layer
        self.w =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Baseline (non-plastic) component of the plastic recurrent layer
        
        self.alpha =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Plasticity coefficients of the plastic recurrent layer; one alpha coefficient per recurrent connection
        #self.alpha = torch.nn.Parameter(.0001 * torch.rand(1,1,hsize))  # Per-neuron alpha
        #self.alpha = torch.nn.Parameter(.0001 * torch.ones(1))         # Single alpha for whole network

        self.h2mod = torch.nn.Linear(hsize, 1)      # Weights from the recurrent layer to the (single) neurodulator output
        self.modfanout = torch.nn.Linear(1, hsize)  # The modulator output is passed through a different 'weight' for each neuron (it 'fans out' over neurons)

        self.h2o = torch.nn.Linear(hsize, num_actions)    # From recurrent to outputs (action probabilities)


        
    def forward(self, inputs, hidden): # hidden is a tuple containing the h-state (i.e. the recurrent hidden state) and the hebbian trace 
            HS = self.hsize
           
            # hidden[0] is the h-state; hidden[1] is the Hebbian trace
            hebb = hidden[1]


            # Each *column* of w, alpha and hebb contains the inputs weights to a single neuron
            hactiv = torch.tanh( self.i2h(inputs) + hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1)  )  # Update the h-state
            activout = self.h2o(hactiv)  # Pure linear, raw scores - to be softmaxed later, outside the function
        

            # Now computing the Hebbian updates...
            deltahebb = torch.bmm(hidden[0].unsqueeze(2), hactiv.unsqueeze(1))  # Batched outer product of previous hidden state with new hidden state
            
            # We also need to compute the eta (the plasticity rate), wich is determined by neuromodulation
            # Note that this is "simple" neuromodulation.
            myeta = torch.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1
            
            # The neuromodulated eta is passed through a vector of fanout weights, one per neuron.
            # Each *column* in w, hebb and alpha constitutes the inputs to a single cell.
            # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
            # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each 
            # batch element. When multiplying by hebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
            # value for each cell but the same value for all inputs of a cell, as required by fanout concept.
            myeta = self.modfanout(myeta) 
            
            
            # Updating Hebbian traces, with a hard clip (other choices are possible)
            self.clipval = 2.0
            hebb = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)

            hidden = (hactiv, hebb)
            return activout, hidden




    def initialZeroState(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize), requires_grad=False )

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize, self.hsize) , requires_grad=False)
    
    def loadWeights(self, weights):
        self.i2h.weight = torch.nn.Parameter(weights["i2h.weight"])
        self.i2h.bias = torch.nn.Parameter(weights["i2h.bias"])
        self.w = torch.nn.Parameter(weights["w"])
        self.alpha = torch.nn.Parameter(weights["alpha"])
        self.h2mod.weight = torch.nn.Parameter(weights["h2mod.weight"])
        self.h2mod.bias = torch.nn.Parameter(weights["h2mod.bias"])
        self.modfanout.weight = torch.nn.Parameter(weights["modfanout.weight"])
        self.modfanout.bias = torch.nn.Parameter(weights["modfanout.bias"])
        self.h2o.weight = torch.nn.Parameter(weights["h2o.weight"])
        self.h2o.bias = torch.nn.Parameter(weights["h2o.bias"])




# A standard RNN (without backpropamine)
class Standard_RNetwork(nn.Module):
    
    def __init__(self, isize, hsize, num_actions, seed): 
        super(Standard_RNetwork, self).__init__()

        # Is all of this really needed?
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.hsize, self.isize  = hsize, isize 

        self.i2h = torch.nn.Linear(isize, hsize)    # Weights from input to recurrent layer
        self.w =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Baseline (non-plastic) component of the plastic recurrent layer

        self.h2o = torch.nn.Linear(hsize, num_actions)    # From recurrent to outputs (action probabilities)


        
    def forward(self, inputs, hidden): # hidden is a tuple containing the h-state (i.e. the recurrent hidden state) and the hebbian trace 
            HS = self.hsize
           
            # hidden[0] is the h-state; hidden[1] is the Hebbian trace
            hebb = hidden[1]


            # Each *column* of w, alpha and hebb contains the inputs weights to a single neuron
            hactiv = torch.tanh( self.i2h(inputs) + torch.matmul(hidden[0], self.w))  # Update the h-state
            activout = self.h2o(hactiv)  # Pure linear, raw scores - to be softmaxed later, outside the function
    

            hidden = (hactiv, hebb)
            return activout, hidden




    def initialZeroState(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize), requires_grad=False )

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize, self.hsize) , requires_grad=False)
    
    def loadWeights(self, weights):
        self.i2h.weight = torch.nn.Parameter(weights["i2h.weight"])
        self.i2h.bias = torch.nn.Parameter(weights["i2h.bias"])
        self.w = torch.nn.Parameter(weights["w"])
        self.h2o.weight = torch.nn.Parameter(weights["h2o.weight"])
        self.h2o.bias = torch.nn.Parameter(weights["h2o.bias"])








