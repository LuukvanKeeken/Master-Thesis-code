import numpy as np
import torch
import torch.nn as nn
import random
from ncps.torch import CfC, LTC


class LTC_Network(nn.Module):

    def __init__(self, isize, hsize, num_actions, seed, fully_connected = True):
        super(LTC_Network, self).__init__()

        # Is all of this really needed?
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        if fully_connected:
            self.ltc_model = LTC(isize, hsize)
        else:
            raise NotImplementedError
        
        self.h2o = nn.Linear(hsize, num_actions)
        self.h2v = nn.Linear(hsize, 1)

    
    def forward(self, inputs, hidden):

        ltc_output, next_hidden = self.ltc_model(inputs, hidden)

        actions = self.h2o(ltc_output)
        values = self.h2v(ltc_output)

        return actions, values, next_hidden



class CfC_Network(nn.Module):

    def __init__(self, isize, hsize, num_actions, seed, fully_connected = True):
        super(CfC_Network, self).__init__()

        # Is all of this really needed?
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        if fully_connected:
            self.ltc_model = CfC(isize, hsize)
        else:
            raise NotImplementedError
        
        self.h2o = nn.Linear(hsize, num_actions)
        self.h2v = nn.Linear(hsize, 1)

    
    def forward(self, inputs, hidden):

        ltc_output, next_hidden = self.ltc_model(inputs, hidden)

        actions = self.h2o(ltc_output)
        values = self.h2v(ltc_output)

        return actions, values, next_hidden




