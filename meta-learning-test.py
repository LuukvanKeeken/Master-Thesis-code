from tqdm import tqdm
from BP_A2C.backpropamine_A2C import BP_RNetwork
# for i in tqdm(range(1000000)):
#     for j in range(100):
#         j


agent_net = BP_RNetwork(4, 64, 2, 42)

# For parameters that are part of the neuromodulation
# subset of the network, set the attribute
# learn_inner_loop to False. This is later used to 
# determine which parameters should be updated in the
# inner loop/meta-train training. At that point we
# could also check whether the name starts with 'nm_'.
# However, this approach is easier for when we might want
# to have additional, non-neuromodulation parameters to
# be frozen in the inner loop.
for name, param in agent_net.named_parameters():
    print(param.requires_grad)
    if (name.startswith('nm_')):
        param.learn_inner_loop = False
        param.learn_outer_loop = True
    else:
        param.learn_inner_loop = True
        param.learn_outer_loop = True

# for name, param in agent_net.named_parameters():
#     print(name, param.learn_inner_loop, param.learn_outer_loop)

