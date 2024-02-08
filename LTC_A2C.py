import numpy as np
import torch
import torch.nn as nn
import random
from ncps_time_constant_extraction.ncps.torch import LTC, CfC
from ncps_time_constant_extraction.ncps.wirings import Wiring


class LTC_Network(nn.Module):

    def __init__(self, isize, hsize, num_actions, seed, fully_connected = True, extract_tau_sys = False):
        super(LTC_Network, self).__init__()

        # Is all of this really needed?
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        if fully_connected:
            self.ltc_model = LTC(isize, hsize, track_tau_system=extract_tau_sys)
        else:
            raise NotImplementedError
        
        self.h2o = nn.Linear(hsize, num_actions)
        self.h2v = nn.Linear(hsize, 1)

        self.extract_tau_sys = extract_tau_sys

    
    def forward(self, inputs, hidden):

        if self.extract_tau_sys:
            ltc_output, next_hidden, tau_sys = self.ltc_model(inputs, hidden)
        else:
            ltc_output, next_hidden = self.ltc_model(inputs, hidden)

        actions = self.h2o(ltc_output)
        values = self.h2v(ltc_output)

        if self.extract_tau_sys:
            return actions, values, next_hidden, tau_sys
        else:
            return actions, values, next_hidden



class CfC_Network(nn.Module):

    def __init__(self, isize, hsize, num_actions, seed, fully_connected = True, extract_tau_sys = False, mode = "default"):
        super(CfC_Network, self).__init__()

        # Is all of this really needed?
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        if fully_connected:
            self.cfc_model = CfC(isize, hsize, track_tau_system=extract_tau_sys, mode=mode)
        else:
            raise NotImplementedError
        
        self.h2o = nn.Linear(hsize, num_actions)
        self.h2v = nn.Linear(hsize, 1)

        self.extract_tau_sys = extract_tau_sys

    
    def forward(self, inputs, hidden):

        if self.extract_tau_sys:
            ltc_output, next_hidden, tau_sys = self.cfc_model(inputs, hidden)
        else:
            ltc_output, next_hidden = self.cfc_model(inputs, hidden)

        actions = self.h2o(ltc_output)
        values = self.h2v(ltc_output)

        if self.extract_tau_sys:
            return actions, values, next_hidden, tau_sys
        else:
            return actions, values, next_hidden




