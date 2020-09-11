# A compiler for ORCA a Neuromorphic Signal Processor
This compiler will be used to program the ORCA neuromorphic processor using 
a high level Network Descript Language (NDL) such as `brian2` or `teili`. 
Ultimately we would like to build a proper backend. Before doing this I 
would like to speak to James Knight and/or Thomas Nowotny as they developed 
the `GeNN` backend. I believe it ultimately breaks down to a code generation 
which converts the `brian2's` NDL into a c++/cuda code for `GeNN` and FPGA 
code for `ORCA`.

In the first iteration I will simply build a text generator function which 
extracts all necessary information from a `TeiliNetwork` or `Network` 
object, similar to `teili2ctxctl` as provided in [teili tools](https://code.ini.uzh.ch/ncs/teili/blob/dev-teili2ctxctl/teili/tools/teili2ctxctl.py).
In contrast to the DYNAP interface `ORCA` needs more information about the
`TeiliNetwork` as it has less limitations and higher flexibility and higher
weight precision. A second important difference is that ORCA features on-chip 
learning so we don't need to transfer learned weights, but rather the same 
initialization parameters for the random distribution and the probability of
connection, rather than exact indices.

Ultimately, we can also provide equations to ORCA, but for now the model is
inherently fixed, so that the teili2orca 1.0 focuses on connectivity and
general NDL rather than equations.

## teili2orca
We need to grab the network object and extract the following information.
*  Total number of neurons aka neuron count across population
*  List/dict of neuron popluation
*  Matching list/dict of population sizes
*  List of populations (pre & post) which are connected via synapses
   *  Sign of the synapse
   *  Static vs. plastic
*  Neuron & Synapse parameter dictionaries

## Installation
In order to specify networks you either need to install [brian2](https://brian2.readthedocs.io/en/stable/introduction/install.html) or [teili](https://teili.readthedocs.io/en/latest/scripts/Getting%20started.html#installation).
Both network descriptions work out of the box and can be converted to an
ORCA friendly format. 

To install `speed` simply clone the repository and use pip to install
the module by pointing it to the folder containing the `setup.py`.
```bash
pip install speed/
```

Alternatively, you can set your `$PYTHONPATH` pointing to where you
cloned the repository to.

## Tutorial
For more detailed and functional example please refer to 
`speed/tutorials/convert_network_tutoral.py`,
You can find more tutorials at `speed/tutorials`.

```python
import os

from speed.teili2orca import Speed
from teili import TeiliNetwork

Net = TeiliNetwork()
# Define your network

Net.run(10*second, report='text')


converted_model = Speed(Net)
converted_model.print_network()
converted_model.save_to_file(filename='orca_net.p',
                             directory=os.path.expanduser('~'))
```

When printing the network structure one expect an output similar to:
```bash
n_pop
   test_WTA__n_exc :  50
   test_WTA__n_inh :  12
s_pop
   test_WTA__s_exc_exc :  ['test_WTA__n_exc', 'test_WTA__n_exc']
   test_WTA__s_exc_inh :  ['test_WTA__n_exc', 'test_WTA__n_inh']
   noise_syn :  ['poissongroup', 'test_WTA__n_exc']
   test_WTA__s_inp_exc :  ['test_WTA__spike_gen', 'test_WTA__n_exc']
   test_WTA__s_inh_exc :  ['test_WTA__n_inh', 'test_WTA__n_exc']
n_params
   test_WTA__n_exc
     Vthr :  -50.4 mV
     EL :  -55. mV
     Iexp :  0. A
     Iconst :  0. A
     Cm :  281. pF
     Inoise :  0. A
     gL :  4.3 nS
     Iadapt :  0. A
     Vres :  -70.6 mV
     refP :  2. ms
   test_WTA__n_inh
     Vthr :  -50.4 mV
     EL :  -55. mV
     Iexp :  0. A
     Iconst :  0. A
     Cm :  281. pF
     Inoise :  0. A
     gL :  4.3 nS
     Iadapt :  0. A
     Vres :  -70.6 mV
     refP :  2. ms
s_total 2069
n_total 62
s_params
   test_WTA__s_exc_exc
     w_plast :  1
     baseweight :  1. nA
     kernel :  0. A s^-1
     tausyn :  5. ms
   test_WTA__s_exc_inh
     w_plast :  1
     baseweight :  1. nA
     kernel :  0. A s^-1
     tausyn :  5. ms
   noise_syn
     w_plast :  1
     baseweight :  1. nA
     kernel :  0. A s^-1
     tausyn :  5. ms
   test_WTA__s_inp_exc
     w_plast :  1
     baseweight :  1. nA
     kernel :  0. A s^-1
     tausyn :  5. ms
   test_WTA__s_inh_exc
     w_plast :  1
     baseweight :  1. nA
     kernel :  0. A s^-1
     tausyn :  5. ms
s_tags
   test_WTA__s_exc_exc
     plastic :  False
     p_connection :  0.376
     mean :  1.0
     std :  0.0
   test_WTA__s_exc_inh
     plastic :  False
     p_connection :  0.715
     mean :  1.0
     std :  0.0
   noise_syn
     plastic :  False
     p_connection :  0.0004
     mean :  1.0
     std :  0.0
   test_WTA__s_inp_exc
     plastic :  False
     p_connection :  0.02
     mean :  1.0
     std :  0.0
   test_WTA__s_inh_exc
     plastic :  False
     p_connection :  1.0
     mean :  1.0
     std :  0.0
```
