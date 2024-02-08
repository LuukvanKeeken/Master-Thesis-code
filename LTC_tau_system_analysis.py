
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from ncps_time_constant_extraction.ncps.wirings import AutoNCP
import torch

# neuron_type = "LTC"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# results_dir = f"LTC_a2c_result_7_2024130_learningrate_0.0001_selectiomethod_evaluation_gamma_0.99_trainingmethod_standard"
# weights_0 = torch.load(f'Master_Thesis_Code/LTC_A2C/training_results/{results_dir}/checkpoint_{neuron_type}_A2C_0.pt', map_location=torch.device(device))

# model = LTC(4, 32, track_tau_system=True)
# model.load_state_dict(weights_0)

# Create a model
#CHECK OF ER NOG VERSCHIL IS TUSSEN CONTINUOUS EN DISCRETE OUTPUT
wiring = AutoNCP(32, 2, sparsity_level=0.5, seed=5)

#GEBRUIK DEZE FUNCTIE OM TE ZIEN WELKE NEURON WAT IS
for i in range(32):
    print(wiring.get_type_of_neuron(i))

model = CfC_Network(4, 32, 2, 5)


# Generate fake data to feed into model
x = torch.randn(1, 4)
output, hx, tau = model(x)
print(output, hx, tau)