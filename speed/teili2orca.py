# -*- coding: utf-8 -*-
""" This file contains a class which handles the `teili` / `brian2` side
of the high-level network compiler to the ORCA processor.

The ORCA processor is described [here](https://www.frontiersin.org/articles/10.3389/fnins.2018.00213/full).
A documentation on `teili` can be found [here](https://teili.readthedocs.io/en/latest/)
`Teili` uses `brian2` to simulate Spiking Neural Networks. More information on `brian2`
can be found [here](https://brian2.readthedocs.io/en/stable/).

To install teili with all dependencies please run:
**pip install teili**

for more information please refer to the [installation instructions](https://teili.readthedocs.io/en/latest/scripts/Getting%20started.html#install-python-requirements).
"""
# @Author: schlowm0 (Moritz Milde)
# @AuthorEmail: m.milde@westernsydney.edu.au
# @Date: 12/09/2019

import os
import numpy as np
import pickle
import collections
from brian2 import Network, Synapses, NeuronGroup
from brain2 import SpikeGeneratorGroup, PoissonGroup

from teili import TeiliNetwork, Connections, Neurons


class teili2orca(TeiliNetwork, Network):
    """This class provides the first iteration of a high-level
    interface to the ORCA Neuromorphic Signal Processor (NSP) developed
    at ICNS, Western Sydney University.
    The ORCA NSP a fully digital FPGA neural network simulator for very-large
    scale event-based Spiking Neural Networks suited to operate on event-based
    data as provided by the Dynamic Vision Sensor.

    Attributes:
        conn_groups (dict): Contains a dictionary of all synaptic groups
            which are present in the network. Each group has a unique ID.
        neuron_groups (dict): Contains a dictionary of all neuron groups
            which are present in the network. Each group has a unique ID.
        spikegen_groups (dict): Contains a dictionary of all
            `SpikeGeneratorGroup`s which are present in the network.
        poisson_groups (dict): Contains a dictionary of all
            `PoissonGroup`s which are present in the network.
        total_num_neurons (int): Total number of neurons in the network.
        total_num_synapses (int): Total number of synapses in the network.
        neuron_populations (dict): {'unique population ID': N}
        neuron_params (dict): {'unique popluation ID': parameters}
        synapse_populations (dict): {'unique synapse ID': [ID_pre, ID_post]}
        synapse_params (dict): {'unique synapse ID': parameters}
        synapse_tags (dict): Tags help to identify synaptic properties. Tags
            consist of
            {'sign', 'target_sign', 'p_connection', 'plastic', 'mean', 'std'}
            where
            *  sign         (str)  : 'exc' | 'inh'
            *  target_sign  (str)  : 'exc' | 'inh'
            *  p_connection (float): [0, 1]
            *  plastic      (bool) : True | False
            *  mean         (float): [0, 1]
            *  std          (float): [0, 1]
        net_dict (dict): Dictionary containing all necessary information
            to program the ORCA processor. The dictionary is structured
            as following:
            *  total_n (int): Total number of neurons
            *  total_s (int): Total number of synapses
            *  n_pop (dict): {'unique population ID': N}
            *  s_pop (dict): {'unique synapse ID': [ID_pre, ID_post]}
            *  s_tags (dict): {'unique synapse ID': synapse_tags}
            *  n_params (dict): {'unique population ID': parameters}
            *  s_params (dict): {'unique synapse ID': parameters}

    """

    def __init__(self, net):
        """Summary

        Args:
            net (TYPE): Description
        """
        self.neuron_groups = {att.name: att for att in net.__dict__['objects']
                              if type(att) == Neurons or type(att) == NeuronGroup}
        self.conn_groups = {att.name: att for att in net.__dict__['objects']
                            if type(att) == Connections or type(att) == Synapses}
        self.poisson_groups = {att.name: att for att in net.__dict__['objects']
                               if type(att) == PoissonGroup}
        self.spikegen_groups = {att.name: att for att in net.__dict__['objects']
                                if type(att) == SpikeGeneratorGroup}

        self.total_num_neurons = 0
        self.total_num_synapses = 0

        for group_key in self.neuron_groups:
            self.total_num_neurons += self.neuron_groups[group_key].N

        for group_key in self.conn_groups:
            self.total_num_synapses += len(self.conn_groups[group_key])

        self.net_dict = collections.OrderedDict()
        self.net_dict = {'n_total': self.total_num_neurons,
                         's_total': self.total_num_synapses,
                         'n_pop': self.extract_neuron_groups(),
                         's_pop': self.extract_synapse_groups(),
                         's_tags': self.extract_synapse_tags(),
                         'n_params': self.extract_neuron_parameters(),
                         's_params': self.extract_synapse_parameters(),
                         }

    def extract_neuron_groups(self):
        """ This function extracts all present `NeuronGroup`/`Neurons` from
        a provided network.

        Returns:
            dict: {'unique population ID': N}
        """
        self.neuron_populations = {}
        for group_key in self.neuron_groups:
            num_neurons = self.neuron_groups[group_key].N
            self.neuron_populations.update({group_key: num_neurons})

        return self.neuron_populations

    def extract_synapse_groups(self):
        """This function returns dictionary that contains the name synapse
        identifier and corresponding pre, post population identifier.

        Returns:
            dict: {'unique synapse ID': [ID_pre, ID_post]}
        """
        self.synapse_populations = {}
        for group_key in self.conn_groups:
            pre_post = [self.conn_groups[group_key].source.name,
                        self.conn_groups[group_key].target.name]
            self.synapse_populations.update({group_key: pre_post})

        return self.synapse_populations

    def extract_synapse_tags(self):
        """This function collects impartant meta information of a given
        synapse group, given the synapse identifier, and returns a
        dictionary

        Returns:
            dict: Tags help to identify synaptic properties.
                Tags consist of
                {'sign', 'target_sign', 'p_connection', 'plastic', 'mean', 'std'}
                where
                *  sign         (str)  : 'exc' | 'inh'
                *  target_sign  (str)  : 'exc' | 'inh'
                *  p_connection (float): [0, 1]
                *  plastic      (bool) : True | False
                *  mean         (float): [0, 1]
                *  std          (float): [0, 1]
        """
        self.synapse_tags = {}
        for group_key in self.conn_groups:
            current_tags = self.conn_groups[group_key]._tags
            current_tags.pop('mismatch', None)
            current_tags.pop('noise', None)
            current_tags.pop('level', None)
            current_tags.pop('num_inputs', None)
            current_tags.pop('bb_type', None)
            current_tags.pop('group_type', None)
            current_tags.pop('connection_type', None)

            if 'taupre' in self.conn_groups[group_key]._init_parameters:
                current_tags.update({'plastic': True})
            else:
                current_tags.update({'plastic': False})

            p_connection = len(self.conn_groups[group_key]) / \
                (self.conn_groups[group_key].source.N *
                 self.conn_groups[group_key].target.N)

            current_tags.update({'p_connection': p_connection})
            current_tags.update({'mean':
                                 np.round(np.mean(
                                     self.conn_groups
                                     [group_key].w_plast), 4)})
            current_tags.update({'std':
                                 np.round(np.std(
                                     self.conn_groups
                                     [group_key].w_plast), 4)})

            self.synapse_tags.update({group_key: current_tags})
        return self.synapse_tags

    def extract_neuron_parameters(self):
        """ This function extracts the initial neuron paramter.
        At the moment we assume that the network **does not** have
        heterogeneous parameters.

        Returns:
            dict: {'unique popluation ID': parameters}
        """
        self.neuron_params = {}
        for group_key in self.neuron_groups:
            self.neuron_params.update({group_key:
                                       self.neuron_groups
                                       [group_key]._init_parameters})
        return self.neuron_params

    def extract_synapse_parameters(self):
        """This function extracts the initial neuron paramter.
        At the moment we assume that the network **does not** have
        heterogeneous parameters.

        Returns:
            dict: {'unique synapse ID': parameters}
        """
        self.synapse_params = {}
        for group_key in self.conn_groups:
            self.synapse_params.update({group_key:
                                        self.conn_groups
                                        [group_key]._init_parameters})
        return self.synapse_params

    def save_to_file(self, filename, directory=None):
        """Simple wrapper function to save network description to a
        pickle file.

        Args:
            filename (str): Desired name to store the network dictionary as
            directory (str, optional): Desired directory to save the network file.
                If no directory is provided the file is saved to:
                `~/teiliApps/output/`
        """

        if directory is None:
            directory = os.path.join(os.path.expanduser('~'),
                                     'teiliApps',
                                     'output')
            if not os.path.exists(directory):
                os.makedirs(directory)

        if filename[-2:] == '.p' or filename[-7:] == '.pickle':
            filename = os.path.join(directory,
                                    filename)
        else:
            filename = filename + '.p'
            filename = os.path.join(directory,
                                    filename)

        with open(filename, 'wb') as handle:
            pickle.dump(self.net_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

            print('Network description saved to: \n {}'
                  .format(filename))

    def load_from_file(self, filename):
        """ Wrapper function to load previously exported network.

        Args:
            filename (str): Filename with full path and file extension.
        """
        with open(filename, 'rb') as handle:
            self.net_dict = pickle.load(handle)

    def print_network(self):
        """Simple print function for quick check
        """
        for k,v in self.net_dict.items():
            if type(v) == dict:
                print(k)
                for kk, vv in v.items():
                    if type(vv) == dict:
                        print('  ', kk)
                        for kkk, vvv in vv.items():
                            print('    ', kkk, ': ', vvv)
                    else:
                        print('  ', kk, ': ', vv)
            else:
                print(k, v)
