import numpy as np
import torch
import torch.nn as nn
import random
from ncps_time_constant_extraction.ncps.torch import LTC, CfC
from ncps_time_constant_extraction.ncps.wirings import Wiring


class LTC_Network(nn.Module):

    def __init__(self, isize, hsize, num_actions, seed, extract_tau_sys = False, wiring = None):
        super(LTC_Network, self).__init__()

        # Is all of this really needed?
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        if not wiring:
            self.ltc_model = LTC(isize, hsize, track_tau_system=extract_tau_sys)
            self.h2o = nn.Linear(hsize, num_actions)
            self.h2v = nn.Linear(hsize, 1)
        else:
            self.ltc_model = LTC(isize, wiring, track_tau_system=extract_tau_sys)
        
        

        self.extract_tau_sys = extract_tau_sys
        self.wiring = wiring

    
    def forward(self, inputs, hidden):

        if self.extract_tau_sys:
            ltc_output, next_hidden, tau_sys = self.ltc_model(inputs, hidden)
        else:
            ltc_output, next_hidden = self.ltc_model(inputs, hidden)

        if not self.wiring:
            actions = self.h2o(ltc_output)
            values = self.h2v(ltc_output)
        else:
            if ltc_output.dim() == 1:
                actions = ltc_output[:-1]
                values = ltc_output[-1]
            elif ltc_output.dim() == 2:
                actions = ltc_output[:, :-1]
                values = ltc_output[:, -1].unsqueeze(-1)
            else:
                raise ValueError("ltc_output has too many dimensions")

        if self.extract_tau_sys:
            return actions, values, next_hidden, tau_sys
        else:
            return actions, values, next_hidden



class CfC_Network(nn.Module):

    def __init__(self, isize, hsize, num_actions, seed, extract_tau_sys = False, mode = "default", wiring = None):#, neuromod_network_dims = None, neuromod_network_activation = nn.Tanh):
        super(CfC_Network, self).__init__()

        # Is all of this really needed?
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        if not wiring:
            # self.cfc_model = CfC(isize, hsize, track_tau_system=extract_tau_sys, mode=mode, neuromod_network_dims=neuromod_network_dims, neuromod_network_activation=neuromod_network_activation)
            self.cfc_model = CfC(isize, hsize, track_tau_system=extract_tau_sys, mode=mode, batch_first=False)
            self.h2o = nn.Linear(hsize, num_actions)
            self.h2v = nn.Linear(hsize, 1)
        else:
            # self.cfc_model = CfC(isize, wiring, track_tau_system=extract_tau_sys, mode=mode, neuromod_network_dims=neuromod_network_dims, neuromod_network_activation=neuromod_network_activation)
            self.cfc_model = CfC(isize, wiring, track_tau_system=extract_tau_sys, mode=mode, batch_first=False)
        
        

        self.extract_tau_sys = extract_tau_sys
        self.wiring = wiring

    
    def forward(self, inputs, hidden, neuromod_signal = None):

        if self.extract_tau_sys:
            cfc_output, next_hidden, tau_sys = self.cfc_model(inputs, hx=hidden, neuromod_signal=neuromod_signal)
        else:
            cfc_output, next_hidden = self.cfc_model(inputs, hx=hidden, neuromod_signal=neuromod_signal)

        if not self.wiring:
            actions = self.h2o(cfc_output)
            values = self.h2v(cfc_output)
        else:
            if cfc_output.dim() == 1:
                actions = cfc_output[:-1]
                values = cfc_output[-1]
            elif cfc_output.dim() == 2:
                actions = cfc_output[:, :-1]
                values = cfc_output[:, -1].unsqueeze(-1)
            else:
                raise ValueError("cfc_output has too many dimensions")


        if self.extract_tau_sys:
            return actions, values, next_hidden, tau_sys
        else:
            return actions, values, next_hidden




