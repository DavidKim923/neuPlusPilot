# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:00:20 2024

@author: David Kim
"""

import sys,os

import numpy as np
import matplotlib.pyplot as plt
from datetime import date

import neuralSim.parameters as param
import neuralSim.compiler as compiler
import neuralSim.synapticTable as synTable
import neuralSim.inputSpikesGen as inSpike
import neuralSim.eventAnalysis as eventAnalysis
import neuralSim.poissonSpike as spikes

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf, suppress=True)

parameters = {}
parameters['pre_trace'] = 12.
parameters['post_trace'] = 4.
parameters['neuron_tau'] = 20.
parameters['synapse_tau'] = 20.
parameters['refractory'] = 0.
parameters['threshold'] = 192.
parameters['wmax'] = 32.



# %% Hardware experiment setup
inputSpikeFilename = "testByte.nam"
synTableFilePrefix = "SynTableWrite"

fname = "testExpConf.exp"
nfname = "testNeuronConf.nac"
conffname = "neuplusconf.txt"
synfname = "testRead.dat"

testSet = compiler.expSetupCompiler(fname, nfname)

#Experiment setup
testSet.setExperimentConf("EVENT_SAVE_MODE", True)
testSet.setExperimentConf("EXP_TIME", 1000)
testSet.setExperimentConf("TIME_ACCEL_MODE", True)
testSet.setExperimentConf("INTERNAL_ROUTING_MODE", True)
testSet.setExperimentConf("INPUT_SPIKES_FILE", inputSpikeFilename)
testSet.setExperimentConf("SYN_TABLE_FILE_PREFIX", synTableFilePrefix)
testSet.setExperimentConf("SYN_TABLE_READ_FILE", synfname)
testSet.setExperimentConf("SYN_TABLE_READ_START", 0)
testSet.setExperimentConf("SYN_TABLE_READ_COUNT", 1024)

# Neuron array
testSet.setNeuronCoreConf([0], [parameters['neuron_tau']], [parameters['refractory']], [parameters['threshold']], [0], [0])  # Core Number, Membrane time constant, Refractory period, Threshold, Synaptic Gain, Stochasticity
testSet.setNeuronCoreConf([1], [parameters['neuron_tau']], [parameters['refractory']], [parameters['threshold']], [0], [0])

# FPGA configuration
testSet.setFPGAConfiguration("CHIP_CLOCK", 150)
# testSet.setFPGAConfiguration("ARRAY_OUTPUT_OFF",  0)
# testSet.setFPGAConfiguration("ROUTER_SYN_OUTPUT_DISABLE", 0)
testSet.genExpConfFile(conffname)

# %% Learning Engine Configuration

testSet.setLearningEngine("LC_TIMECONSTANT", 0, parameters['synapse_tau'])
testSet.setLearningEngine("MAX_SYN_WEIGHT", 0, parameters['wmax'])
testSet.setLearningEngine("TRACE_UPDATE_AMOUNT", 0, 0, parameters['pre_trace'])
testSet.setLearningEngine("TRACE_UPDATE_AMOUNT", 0, 1, parameters['post_trace'])
# testSet.setLearningEngine("LEARNING_RULE", 0, 1, 1, 0, 1, 1)
testSet.setLearningEngine("LEARNING_RULE", 0, 1, 1, 0, 1, 1)
testSet.setLearningEngine("MOD_AREA_DEFINE", 0, 1, 1024) # [0]: [1]:num_of_neurons [2]:start_neuron_num

# %%
##################################
# Experimental variable settings #
##################################
pattern_duration = 0.5      # [sec]
simulation_duration = 100   # [sec]

################
# Network size #
################
cortex_num = 200
str1_num = 20           # Direct striatum pathway neuron
str2_num = 20           # Indirect striatum pathway neuron
snc1_num = 20
snc2_num = 20
# strf_num = 84           # Striatum fast spiking neuron
# gpe_num = 46            # Globus pallidus external
# gpi_num = 27            # Globus pallidus internal

#########################
# Input spike generator #
#########################
# 1. noise
noise_firing_rate = 3 # [Hz]
noise_spike_idx = np.random.randint(0, cortex_num, size=noise_firing_rate*simulation_duration*cortex_num)
noise_spike_time = np.round(np.sort(np.random.rand(noise_firing_rate*simulation_duration*cortex_num)*simulation_duration), 3)

# 2. pattern
pattern_firing_rate = 15 # [Hz]

a_pattern_window = np.sort(np.random.choice(simulation_duration, simulation_duration//2, replace=False))
b_pattern_window = [i for i in range(simulation_duration) if i not in a_pattern_window]

a_pattern_neuron_list = np.sort(np.random.choice(cortex_num, cortex_num//4, replace=False))
b_pattern_neuron_list = np.sort(np.random.choice(cortex_num, cortex_num//4, replace=False))
np.random.shuffle(a_pattern_neuron_list)
np.random.shuffle(b_pattern_neuron_list)

a_pattern_spike_idx = np.array([])
for i in range(round(pattern_firing_rate*pattern_duration)):
    random_neuron_list = np.random.permutation(a_pattern_neuron_list)
    a_pattern_spike_idx = np.append(a_pattern_spike_idx, random_neuron_list)

b_pattern_spike_idx = np.array([])
for i in range(round(pattern_firing_rate*pattern_duration)):
    random_neuron_list = np.random.permutation(b_pattern_neuron_list)
    b_pattern_spike_idx = np.append(b_pattern_spike_idx, random_neuron_list)
 
a_pattern_spike_time = np.round(np.sort(np.random.rand(round(pattern_firing_rate*pattern_duration)*cortex_num//4)*pattern_duration), 3)
b_pattern_spike_time = np.round(np.sort(np.random.rand(round(pattern_firing_rate*pattern_duration)*cortex_num//4)*pattern_duration), 3)

input_spike_time = np.array([])
input_spike_idx = np.array([])

input_spike_time = np.append(input_spike_time, noise_spike_time)
input_spike_idx = np.append(input_spike_idx, noise_spike_idx)

for i in range(simulation_duration):
    if i in a_pattern_window:
        input_spike_time = np.append(input_spike_time, a_pattern_spike_time+i)
        input_spike_idx = np.append(input_spike_idx, a_pattern_spike_idx)
    
    elif i in b_pattern_window:
        input_spike_time = np.append(input_spike_time, b_pattern_spike_time+i)
        input_spike_idx = np.append(input_spike_idx, b_pattern_spike_idx)
    else:
        pass

for i in range(1, len(input_spike_time)):
    if input_spike_idx[i] == input_spike_idx[i-1]:
        if input_spike_time[i] - input_spike_time[i-1] <= 0.001:
            input_spike_time[i] = input_spike_time[i-1]+0.001

sorted_indices = np.argsort(input_spike_time)
input_spike_time = input_spike_time[sorted_indices]
input_spike_idx = input_spike_idx[sorted_indices]

plt.figure(dpi=1200)
plt.plot(input_spike_time, input_spike_idx, '.k', markersize=2)
plt.plot(noise_spike_time, noise_spike_idx, '.r', markersize=1)
plt.xlim([0, 10])
plt.savefig('input_raster_plot.png', dpi=1200)



# 3. dopamine 
dopamine_firing_rate = 4 # [Hz]
a_dopamine_spike_time = np.array([])
b_dopamine_spike_time = np.array([])
a_dopamine_spike_idx = np.array([])
b_dopamine_spike_idx = np.array([])

for time in a_pattern_window:
    for idx in range(snc1_num):
        random_uniform_list = np.random.rand(round(dopamine_firing_rate*pattern_duration))*pattern_duration+time
        # a_dopamine_spike_time = np.append(a_dopamine_spike_time, random_uniform_list)
        a_dopamine_spike_time = np.append(a_dopamine_spike_time, np.ones(round(dopamine_firing_rate*pattern_duration))*time)
        a_dopamine_spike_time = np.append(a_dopamine_spike_time, np.ones(round(dopamine_firing_rate*pattern_duration))*(time+0.25))
        
        a_dopamine_spike_idx = np.append(a_dopamine_spike_idx, np.ones(round(dopamine_firing_rate*pattern_duration*2))*(idx))
        #

for time in b_pattern_window: 
    for idx in range(snc2_num):
        random_uniform_list = np.random.rand(round(dopamine_firing_rate*pattern_duration))*pattern_duration+time
        # b_dopamine_spike_time = np.append(b_dopamine_spike_time, random_uniform_list)
        b_dopamine_spike_time = np.append(b_dopamine_spike_time, np.ones(round(dopamine_firing_rate*pattern_duration))*time)
        b_dopamine_spike_time = np.append(b_dopamine_spike_time, np.ones(round(dopamine_firing_rate*pattern_duration))*(time+0.25))
        #
        # b_dopamine_spike_idx = np.append(b_dopamine_spike_idx, np.ones(round(dopamine_firing_rate*pattern_duration))*(idx+str1_num))
        b_dopamine_spike_idx = np.append(b_dopamine_spike_idx, np.ones(round(dopamine_firing_rate*pattern_duration*2))*(idx+snc1_num))
        #

reward_time = np.append(a_dopamine_spike_time, b_dopamine_spike_time)
reward_idx = np.append(a_dopamine_spike_idx, b_dopamine_spike_idx)
sorted_indices = np.argsort(reward_time)
reward_time = reward_time[sorted_indices]
reward_idx = reward_idx[sorted_indices]
reward_core = np.zeros(len(reward_time))
reward_value = np.ones(len(reward_time))*127

for i in range(1, len(reward_time)):
    if reward_time[i] - reward_time[i-1] <= 0.001:
        reward_time[i] = reward_time[i-1]+0.001

plt.figure(dpi=1200)
plt.plot(reward_time, reward_idx, '.r', markersize=2)
plt.xlim([0, 10])
plt.title('Dopamine ratser plot')
plt.xlabel('Time [sec]')
plt.ylabel('Modulator index [#]')
plt.savefig('dopamine_raster_plot1.png', dpi=1200)

plt.figure(dpi=1200)
plt.plot(reward_time, reward_idx, '.r', markersize=2)
plt.xlim([simulation_duration-10, simulation_duration])
plt.title('Dopamine ratser plot')
plt.xlabel('Time [sec]')
plt.ylabel('Modulator index [#]')
plt.savefig('dopamine_raster_plot2.png', dpi=1200)

# %%
cortex_idx = np.arange(0, cortex_num)
str1_idx = np.arange(1024, 1024+str1_num)
str2_idx = np.arange(1024+str1_num, 1024+str1_num+str2_num)

events = [input_spike_time, input_spike_idx]

inputSpike = inSpike.MakeNeuPLUSByteFile(deltat=1000)

rewards = [reward_time, reward_core, reward_idx, reward_value]  # [time, core, address, value]
# rewards = []

inputSpike.events_to_bytefile(events=events, 
                              byte_file_name=inputSpikeFilename, 
                              conf_events=rewards, 
                              weight_single=248.)

testSet.genExpConfFile(conffname)

SynTable = synTable.synapticTable(2, 1024, 1024)

cortex_str1_weight = np.random.rand(cortex_num*str1_num)*parameters['wmax']
cortex_str2_weight = np.random.rand(cortex_num*str1_num)*parameters['wmax']

# cortex_str1_weight = np.ones(cortex_num*str1_num)*parameters['wmax']/2
# cortex_str2_weight = np.ones(cortex_num*str2_num)*parameters['wmax']/2

SynTable.createConnections(source = np.sort(np.tile(cortex_idx, str1_num)), 
                           destination = np.tile(str1_idx, cortex_num), 
                           mode = 'one', 
                           probability = 1, 
                           weight = cortex_str1_weight, 
                           wmax = parameters['wmax'], 
                           delay = 0, 
                           dmax = 0, 
                           synType = 'exc', 
                           trainable = True)

SynTable.createConnections(source = np.sort(np.tile(cortex_idx, str2_num)), 
                            destination = np.tile(str2_idx, cortex_num), 
                            mode = 'one', 
                            probability = 1, 
                            weight = cortex_str2_weight, 
                            wmax = parameters['wmax'], 
                            delay = 0, 
                            dmax = 0, 
                            synType = 'exc', 
                            trainable = True)

SynTable.createConnections(source = str1_idx, 
                            destination = str2_idx, 
                            mode = 'all', 
                            probability = 20/str1_num, 
                            weight = 6, 
                            wmax = parameters['wmax'], 
                            delay = 0, 
                            dmax = 0, 
                            synType = 'inh', 
                            trainable = False)

SynTable.createConnections(source = str2_idx, 
                            destination = str1_idx, 
                            mode = 'all', 
                            probability = 20/str2_num, 
                            weight = 6, 
                            wmax = parameters['wmax'], 
                            delay = 0, 
                            dmax = 0, 
                            synType = 'inh', 
                            trainable = False)

SynTable.createConnections(source = str1_idx, 
                            destination = str1_idx, 
                            mode = 'all', 
                            probability = 20/str1_num, 
                            weight = 1.0625, 
                            wmax = parameters['wmax'], 
                            delay = 0, 
                            dmax = 0, 
                            synType = 'inh', 
                            trainable = False)

SynTable.createConnections(source = str2_idx, 
                            destination = str2_idx, 
                            mode = 'all', 
                            probability = 20/str2_num, 
                            weight = 1.0625, 
                            wmax = parameters['wmax'], 
                            delay = 0, 
                            dmax = 0, 
                            synType = 'inh', 
                            trainable = False)

SynMapCompiled = compiler.synMapCompiler(synTableFilePrefix)
SynMapCompiled.generateSynMap(SynTable, [range(3072)])

# %% Run on NeuPLUS system
elapsed_time = param.NeuPLUSRun('-GUIModeOff', '-conf', conffname)
print("Result time : ", elapsed_time)


# %% Event analysis
event_results = eventAnalysis.eventAnalysis(param.NeuPLUSResultFolder + '2024' + date.today().strftime("%m%d") + "/")

output_spike_idx = np.array(event_results.spikes[1])
output_spike_time  = np.array(event_results.spikes[0])

# %% Synapse table read
synMapDecompile     = compiler.synMapDecompiler(synfname)
synTable_read       = synTable.synapticTable()
synDecompiledList   = synMapDecompile.decompileSynMap()

# 1. Filtering spike data
str1_idx = np.where((output_spike_idx >= 1024) & (output_spike_idx < 1024+str1_num))[0]
str1_time = output_spike_time[str1_idx]
before_str1_idx = np.where(str1_time < 10)[0]
before_str1_time = str1_time[before_str1_idx]
after_str1_idx = np.where(str1_time > simulation_duration-10)[0]
after_str1_time = str1_time[after_str1_idx]

str2_idx = np.where((output_spike_idx >= 1024+str1_num) & (output_spike_idx < 1024+str1_num+str2_num))[0]
str2_time = output_spike_time[str2_idx]
before_str2_idx = np.where(str2_time < 10)[0]
before_str2_time = str2_time[before_str2_idx]
after_str2_idx = np.where(str2_time > simulation_duration-10)[0]
after_str2_time = str2_time[after_str2_idx]

cortex_idx = np.where(output_spike_idx<cortex_num)[0]
cortex_time = output_spike_time[cortex_idx]

# 2. Visualization
plt.figure(dpi=1200)
plt.plot(cortex_time, output_spike_idx[cortex_idx], '.k', markersize=1)
plt.xlabel('Time')
plt.ylabel('Neuron index')
plt.savefig('cortex_raster_plot.png', dpi=1200)

# fig, (ax1, ax2) = plt.subplots(2, 1, dpi=1200)
# ax1.plot(str1_time, output_spike_idx[str1_idx], '.k', markersize=1)
# ax1.set_xlim([0, simulation_duration])
# ax1.set_xlabel('Time [sec]')
# ax1.set_ylabel('Neuron index')
# ax1.set_title('STR1')
# ax2.plot(str2_time, output_spike_idx[str2_idx], '.k', markersize=1)
# ax2.set_xlim([0, simulation_duration])
# ax2.set_xlabel('Time [sec]')
# ax2.set_ylabel('Neuron index')
# ax2.set_title('STR2')
# plt.tight_layout()
# plt.savefig('str_raster_plot.png', dpi=1200)

fig, (ax1, ax2) = plt.subplots(2, 1, dpi=1200)
ax1.plot(before_str2_time, output_spike_idx[str2_idx[before_str2_idx]], '.k', markersize=1)
ax1.set_xlim([0, 10])
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Neuron index')
ax1.set_title('Before training STR2')
ax2.plot(before_str1_time, output_spike_idx[str1_idx[before_str1_idx]], '.k', markersize=1)
ax2.set_xlim([0, 10])
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Neuron index')
ax2.set_title('Before training STR1')
plt.tight_layout()
plt.savefig('before_raster_plot.png', dpi=1200)

fig, (ax1, ax2) = plt.subplots(2, 1, dpi=1200)
ax1.plot(after_str2_time, output_spike_idx[str2_idx[after_str2_idx]], '.k', markersize=1)
ax1.set_xlim([simulation_duration-10, simulation_duration])
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Neuron index')
ax1.set_title('After training STR2')
ax2.plot(after_str1_time, output_spike_idx[str1_idx[after_str1_idx]], '.k', markersize=1)
ax2.set_xlim([simulation_duration-10, simulation_duration])
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Neuron index')
ax2.set_title('After training STR1')
plt.tight_layout()
plt.savefig('after_raster_plot.png', dpi=1200)

bins = np.arange(0, parameters['wmax']+2, 2)
before_weight = np.append(cortex_str1_weight, cortex_str2_weight)
plt.figure(dpi=1200)
plt.hist(before_weight, bins=bins, edgecolor='black')
plt.title('weight distribution before training')
plt.savefig('before_weight_distribution.png', dpi=1200)

plt.figure(dpi=1200)
plt.hist(synDecompiledList[2], bins=bins, edgecolor='black')
plt.title('weight distribution after training')
plt.savefig('after_weight_distribution.png', dpi=1200)

# print('before weight')
# print(parameters['winit'])
# print('after weight')
# print(synDecompiledList[2])

