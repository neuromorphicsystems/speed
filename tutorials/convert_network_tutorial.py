import os

from brian2 import PoissonGroup
from brian2 import StateMonitor, SpikeMonitor
from brian2 import Hz, second, ms

from teili import Neurons, Connections, TeiliNetwork
from teili.models.neuron_models import LinearLIF as neuron_model
from teili.models.synapse_models import Exponential as static_synapse_model
from teili.models.synapse_models import ExponentialStdp as plastic_synapse_model

from speed.teili2orca import Speed

# Defining the network
N = 1000
F = 8*Hz

Net = TeiliNetwork()

input = PoissonGroup(N, rates=F)
neurons = Neurons(2, equation_builder=neuron_model(num_inputs=1),
                  name='neurons')

S = Connections(input, neurons,
                equation_builder=plastic_synapse_model(), name='stdp_synapse')


S.connect()
S.weight = 1.0
S.w_plast = 'rand()'
S.dApre = 0.01
S.taupre = 20 * ms
S.taupost = 20 * ms

Net.add(input, neurons, S)

# Simulating the networ
Net.run(10*second, report='text')

# Converting the network
converted_model = Speed(Net)

# Example of how print neuron equation from the original network
neurons.print_equations()

# Print the converted model properties
converted_model.print_network()

# Save the converted model to a file which can be loaded by the ORCA compiler
converted_model.save_to_file(filename='orca_net.p',
                             directory=os.path.expanduser('~'))
