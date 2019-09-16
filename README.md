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

At the this stage please make sure to set your `$PYTHONPATH` pointing to where you
cloned the repository to. In the near future `speed` can be installed using pip.

## Tutorial
For more detailed and functional example please refer to 
`speed/tutorials/convert_network_tutoral.py`,
You can find more tutorials at `speed/tutorials`.

```python
import os
from teili2orca import Speed
from teili import TeiliNetwork

Net = TeiliNetwork()
# Define your network

Net.run(10*second, report='text')


converted_model = Speed(Net)
converted_model.print_network()
converted_model.save_to_file(filename='orca_net.p',
                             directory=os.path.expanduser('~'))
```
