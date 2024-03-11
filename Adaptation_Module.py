import torch.nn as nn
import random
import torch
import numpy as np


class StandardRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function = "Tanh", seed = 5):
        super(StandardRNN, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.rnn = nn.RNN(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_state = None

        if activation_function == "Tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError("Given activation function not implemented yet.")
        
    
    def forward(self, input):
        rnn_out, self.hidden_state = self.rnn(input, self.hidden_state)
        output = self.linear(rnn_out)
        return output
    

    def reset_hidden_state(self):
        self.hidden_state = None


    
