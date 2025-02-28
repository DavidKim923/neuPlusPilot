# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:46:15 2024

@author: David Kim
"""

import sys,os
# os.chdir(r"C:\Users\user\Desktop\")
# sys.path.append(os.getcwd())


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
parameters['threshold'] = 200.
parameters['wmax'] = 40.
parameters['mod_num'] = 64


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
testSet.genExpConfFile(conffname)

# %% Learning Engine Configuration

testSet.setLearningEngine("LC_TIMECONSTANT", 0, parameters['synapse_tau'])
testSet.setLearningEngine("MAX_SYN_WEIGHT", 0, parameters['wmax'])
testSet.setLearningEngine("TRACE_UPDATE_AMOUNT", 0, 0, parameters['pre_trace'])
testSet.setLearningEngine("TRACE_UPDATE_AMOUNT", 0, 1, parameters['post_trace'])
testSet.setLearningEngine("LEARNING_RULE", 0, 1, 1, 0, 1, 1)
if parameters['mod_num'] == 1:
    testSet.setLearningEngine("MOD_AREA_DEFINE", 0, parameters['mod_num'], 1024) # [0]: [1]:num_of_neurons [2]:start_neuron_num
else:
    testSet.setLearningEngine("MOD_AREA_DEFINE", 0, parameters['mod_num'], 0)

# %%
##################################
# Experimental variable settings #
##################################
pattern_duration = 0.5      # [sec]
simulation_duration = 500   # [sec]
pattern_num = 10
pattern_mode = 2                    # 0 : iteration, 1 : random order, 2 : random duration (Unit of 20% of pattern_duration.)
reward_mode = 0                     # 0 : constant interval, 1 : random interval 

################
# Network size #
################
cortex_num = 300
str_num = 64
snc_num = 1


#########################
# Input spike generator #
#########################
# 1. noise
noise_firing_rate = 2 # [Hz]
noise_spike_idx = np.random.randint(0, cortex_num, size=noise_firing_rate*simulation_duration*cortex_num)
noise_spike_time = np.round(np.sort(np.random.rand(noise_firing_rate*simulation_duration*cortex_num)*simulation_duration), 3)

# 2. pattern
pattern_firing_rate = 15 # [Hz]
if pattern_mode == 0:
    pattern_time_windows = np.random.permutation(pattern_num)
    
    pattern_time_window = {}
    for i in range(pattern_num):
        pattern_time_window[str(i)] = np.arange(np.where(pattern_time_windows==i)[0][0], simulation_duration, pattern_num)
    
    pattern_neuron_idx = {}
    for i in range(pattern_num):
        pattern_neuron_idx[str(i)] = np.array([])
        pattern_neuron_idx[str(i)] = np.append(pattern_neuron_idx[str(i)], np.random.choice(range(cortex_num), cortex_num//6, replace=False))
        np.random.shuffle(pattern_neuron_idx[str(i)])
    
    pattern_spike_idx = {}
    pattern_spike_time = {}
    for i in range(pattern_num):
        pattern_spike_idx[str(i)] = np.array([])
        pattern_spike_time[str(i)] = np.round(np.sort(np.random.rand(round(pattern_firing_rate*pattern_duration)*cortex_num//6)*pattern_duration), 3)
        for j in range(round(pattern_firing_rate*pattern_duration)):
            pattern_spike_idx[str(i)] = np.append(pattern_spike_idx[str(i)], np.random.permutation(pattern_neuron_idx[str(i)]))
    
    iteration_spike_time = np.array([])
    iteration_spike_idx = np.array([])
    
    for i in range(pattern_num):
        iteration_spike_time = np.append(iteration_spike_time, pattern_spike_time[str(np.where(pattern_time_windows==i)[0][0])]+i)
        iteration_spike_idx = np.append(iteration_spike_idx, pattern_spike_idx[str(np.where(pattern_time_windows==i)[0][0])])
    
    input_spike_time = np.array([])
    input_spike_idx = np.array([])
    for i in range(simulation_duration//pattern_num):
        input_spike_time = np.append(input_spike_time, iteration_spike_time+i*(pattern_num))
        input_spike_idx = np.append(input_spike_idx, iteration_spike_idx)
    
    cortex_spike_time = np.array([])
    cortex_spike_idx = np.array([])
    cortex_spike_time = np.append(input_spike_time, noise_spike_time)
    cortex_spike_idx = np.append(input_spike_idx, noise_spike_idx)
    sorted_indices = np.argsort(cortex_spike_time)
    cortex_spike_time = cortex_spike_time[sorted_indices]
    cortex_spike_idx = cortex_spike_idx[sorted_indices]
    
    for i in range(1, len(cortex_spike_time)):
        if cortex_spike_time[i] - cortex_spike_time[i-1] <= 0.0005:
            cortex_spike_time[i] = cortex_spike_time[i-1]+0.0005
    
elif pattern_mode == 1:
    
    shuffled_indices = np.random.permutation(simulation_duration)
    pattern_time_windows = np.array_split(shuffled_indices, pattern_num)
    
    pattern_time_window = {}
    for i in range(pattern_num):
        pattern_time_window[str(i)] = np.sort(pattern_time_windows[i])
    
    pattern_neuron_idx = {}
    for i in range(pattern_num):
        pattern_neuron_idx[str(i)] = np.array([])
        pattern_neuron_idx[str(i)] = np.append(pattern_neuron_idx[str(i)], np.random.choice(range(cortex_num), cortex_num//6, replace=False))
        np.random.shuffle(pattern_neuron_idx[str(i)])
    
    pattern_spike_idx = {}
    pattern_spike_time = {}
    for i in range(pattern_num):
        pattern_spike_idx[str(i)] = np.array([])
        pattern_spike_time[str(i)] = np.round(np.sort(np.random.rand(round(pattern_firing_rate*pattern_duration)*cortex_num//6)*pattern_duration), 3)
        for j in range(round(pattern_firing_rate*pattern_duration)):
            pattern_spike_idx[str(i)] = np.append(pattern_spike_idx[str(i)], np.random.permutation(pattern_neuron_idx[str(i)]))
    
    input_spike_time = np.array([])
    input_spike_idx = np.array([])
    for t in range(simulation_duration):
        for i in range(pattern_num):
            if t in pattern_time_window[str(i)]:
                input_spike_time = np.append(input_spike_time, pattern_spike_time[str(i)]+t)
                input_spike_idx = np.append(input_spike_idx, pattern_spike_idx[str(i)])
                    
    cortex_spike_time = np.array([])
    cortex_spike_idx = np.array([])
    cortex_spike_time = np.append(input_spike_time, noise_spike_time)
    cortex_spike_idx = np.append(input_spike_idx, noise_spike_idx)
    sorted_indices = np.argsort(cortex_spike_time)
    cortex_spike_time = cortex_spike_time[sorted_indices]
    cortex_spike_idx = cortex_spike_idx[sorted_indices]
    
    for i in range(1, len(cortex_spike_time)):
        if cortex_spike_time[i] - cortex_spike_time[i-1] <= 0.0005:
            cortex_spike_time[i] = cortex_spike_time[i-1]+0.0005
    
elif pattern_mode == 2:
    max_time = 0
    pattern_neuron_idx = {}
    for i in range(pattern_num):
        pattern_neuron_idx[str(i)] = np.array([])
        pattern_neuron_idx[str(i)] = np.append(pattern_neuron_idx[str(i)], np.random.choice(range(cortex_num), cortex_num//6, replace=False))
        np.random.shuffle(pattern_neuron_idx[str(i)])
    
    pattern_spike_idx = {}
    pattern_spike_time = {}
    for i in range(pattern_num):
        pattern_spike_idx[str(i)] = np.array([])
        pattern_spike_time[str(i)] = np.round(np.sort(np.random.rand(round(pattern_firing_rate*pattern_duration)*cortex_num//6)*pattern_duration), 3)
        for j in range(round(pattern_firing_rate*pattern_duration)):
            pattern_spike_idx[str(i)] = np.append(pattern_spike_idx[str(i)], np.random.permutation(pattern_neuron_idx[str(i)]))
    
    pattern_array = np.array([])
    max_time_array = np.array([])
    input_spike_time = np.array([])
    input_spike_idx = np.array([])
    dopamine_spike_time = {}
    dopamine_spike_idx = {}
    random_pattern = np.array([])
    random_resting = np.array([])
    for i in range(pattern_num):
        dopamine_spike_time[str(i)] = np.array([])
        dopamine_spike_idx[str(i)] = np.array([])
    while max_time <= simulation_duration:
        max_time_array = np.append(max_time_array, max_time)
        random_pattern_duration = (np.random.randint(1, 6)/5)*pattern_duration
        random_resting_duration = (np.random.randint(1, 6)/5)*pattern_duration
        random_pattern = np.append(random_pattern, random_pattern_duration)
        random_resting = np.append(random_resting, random_resting_duration)
        selected_pattern = np.random.randint(0, 10)
        pattern_array = np.append(pattern_array, selected_pattern)
        selected_pattern_time = pattern_spike_time[str(selected_pattern)][np.where(pattern_spike_time[str(selected_pattern)]<random_pattern_duration)[0]]
        selected_pattern_idx = pattern_spike_idx[str(selected_pattern)][np.where(pattern_spike_time[str(selected_pattern)]<random_pattern_duration)[0]]
        input_spike_time = np.append(input_spike_time, selected_pattern_time+max_time)
        input_spike_idx = np.append(input_spike_idx, selected_pattern_idx)
        
        if parameters['mod_num'] == str_num & parameters['mod_num'] != 1:
            dopamine_spike_time[str(selected_pattern)] = np.append(dopamine_spike_time[str(selected_pattern)], np.arange(0, random_pattern_duration, 0.1)+max_time)
            dopamine_spike_idx[str(selected_pattern)] = np.append(dopamine_spike_idx[str(selected_pattern)], np.ones(len(np.arange(0, random_pattern_duration, 0.1)))*(selected_pattern+1024//parameters['mod_num']))
        else:
            print('Dopamine error')
        max_time += random_pattern_duration
        max_time += random_resting_duration
    cortex_spike_time = np.array([])
    cortex_spike_idx = np.array([])
    cortex_spike_time = np.append(input_spike_time, noise_spike_time)
    cortex_spike_idx = np.append(input_spike_idx, noise_spike_idx)
    sorted_indices = np.argsort(cortex_spike_time)
    cortex_spike_time = cortex_spike_time[sorted_indices]
    cortex_spike_idx = cortex_spike_idx[sorted_indices]
    
    for i in range(1, len(cortex_spike_time)):
        if cortex_spike_time[i] - cortex_spike_time[i-1] <= 0.0005:
            cortex_spike_time[i] = cortex_spike_time[i-1]+0.0005
else:
    pass

# 3. dopamine 

if pattern_mode != 2:
    if reward_mode == 0:
        dopamine_firing_rate = 10 # [Hz]
    elif reward_mode == 1:
        dopamine_firing_rate = 10 # [Hz]
    else:
        pass
    
    dopamine_spike_time = {}
    dopamine_spike_idx = {}
    for i in range(pattern_num):
        dopamine_spike_time[str(i)] = np.array([])
        dopamine_spike_idx[str(i)] = np.array([])
        
    for t in range(simulation_duration):
        for i in range(pattern_num):
            if t in pattern_time_window[str(i)]:
                for j in range(snc_num):
                    if reward_mode == 0:
                        dopamine_spike_time[str(i)] = np.append(dopamine_spike_time[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration))*t)
                        dopamine_spike_time[str(i)] = np.append(dopamine_spike_time[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration))*t+0.1)
                        dopamine_spike_time[str(i)] = np.append(dopamine_spike_time[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration))*t+0.2)
                        dopamine_spike_time[str(i)] = np.append(dopamine_spike_time[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration))*t+0.3)
                        dopamine_spike_time[str(i)] = np.append(dopamine_spike_time[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration))*t+0.4)
                        if parameters['mod_num'] == 1:
                            dopamine_spike_idx[str(i)] = np.append(dopamine_spike_idx[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration*5))*(i*snc_num+j))
                        else:
                            dopamine_spike_idx[str(i)] = np.append(dopamine_spike_idx[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration*5))*(i*snc_num+j+1024//parameters['mod_num']))
                    elif reward_mode == 1:
                        dopamine_spike_time[str(i)] = np.append(dopamine_spike_time[str(i)], np.random.rand(round(dopamine_firing_rate*pattern_duration))*pattern_duration+t)
                        if parameters['mod_num'] == 1:
                            dopamine_spike_idx[str(i)] = np.append(dopamine_spike_idx[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration))*(i*snc_num+j))
                        else:
                            dopamine_spike_idx[str(i)] = np.append(dopamine_spike_idx[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration))*(i*snc_num+j+1024//parameters['mod_num']))
                    else:
                        pass
                    
    
    
    
    for i in range(pattern_num):
        sorted_indices = np.argsort(dopamine_spike_time[str(i)])
        dopamine_spike_time[str(i)] = dopamine_spike_time[str(i)][sorted_indices]
        dopamine_spike_idx[str(i)] = dopamine_spike_idx[str(i)][sorted_indices]
        if reward_mode == 1:
            for u in range(1, len(dopamine_spike_time[str(i)])):
                if dopamine_spike_time[str(i)][u] - dopamine_spike_time[str(i)][u-1] <= 0.0005:
                    dopamine_spike_time[str(i)][u] = dopamine_spike_time[str(i)][u-1]+0.0005
        else:
            pass

reward_time_arrays = list(dopamine_spike_time.values())
reward_time = np.concatenate(reward_time_arrays)
reward_idx_arrays = list(dopamine_spike_idx.values())
reward_idx = np.concatenate(reward_idx_arrays)
sorted_indices = np.argsort(reward_time)
reward_time = reward_time[sorted_indices]
reward_idx = reward_idx[sorted_indices]
reward_core = np.zeros(len(reward_time))
reward_value = np.ones(len(reward_time))*127

plt.figure(figsize=(6, 3), dpi=1200)
plt.plot(noise_spike_time, noise_spike_idx, '|', color='black', markersize=2, alpha=0.3)
plt.plot(input_spike_time, input_spike_idx, '|', color='black', markersize=2)
plt.xlim([290, 295])
plt.ylim([0, 300])
plt.xticks(np.arange(290, 295.1, 0.5), [''] * len(np.arange(290, 295.1, 0.5)))
plt.yticks(np.arange(0, 301, 50), [''] * len(np.arange(0, 301, 50)))
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig('input_raster_plot.png', dpi=1200, transparent=True)

fig, axes = plt.subplots(pattern_num, 1, figsize=(8, 2*pattern_num), dpi=600)
for ax, key in zip(axes, dopamine_spike_time):
    times = dopamine_spike_time[key]
    indices = dopamine_spike_idx[key]
    ax.plot(times, indices, '.r', markersize=2)
    ax.set_title(f'snc{key} dopamine')
    # ax.set_xlim([0, simulation_duration/10])
    ax.set_xlim([0, 3])
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Index [#]')
plt.tight_layout()
plt.savefig('snc_initial_raster_plot.png', dpi=600)

fig, axes = plt.subplots(pattern_num, 1, figsize=(8, 2*pattern_num), dpi=600)
for ax, key in zip(axes, dopamine_spike_time):
    times = dopamine_spike_time[key]
    indices = dopamine_spike_idx[key]
    ax.plot(times, indices, '.r', markersize=2)
    ax.set_title(f'snc{key} dopamine')
    # ax.set_xlim([simulation_duration-simulation_duration/10, simulation_duration])
    ax.set_xlim([simulation_duration-3, simulation_duration])
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Index [#]')
plt.tight_layout()
plt.savefig('snc_final_raster_plot.png', dpi=600)

start = 497
end = 500

selected_indices = np.where((reward_time >= start) & (reward_time < end))[0]
reward_fg_idx = reward_idx[selected_indices]-1024//parameters['mod_num']
reward_fg_time = reward_time[selected_indices]-0.1

plt.figure(figsize=(3, 0.5), dpi=1200)
plt.plot(reward_fg_time, reward_fg_idx, '.', color='red', markersize=4)
plt.xlim([start, end])
plt.ylim([-0.5, 9.5])
plt.xticks(np.arange(start, end+0.1, 0.5), [''] * len(np.arange(start, end+0.1, 0.5)))
# plt.yticks(np.arange(0, 9.1, 9), [''] * len(np.arange(0, 9.1, 9)))
plt.yticks([])
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig('snc_raster_plot.png', dpi=1200, transparent=True)




# %%
snn_obj = {}
snn_weight = {}
snn_obj['cortex'] = np.arange(0, cortex_num)
for i in range(pattern_num):
    snn_obj['str'+str(i)] = np.arange(int(1024+str_num*i), int(1024+str_num*(i+1)))

events = [cortex_spike_time, cortex_spike_idx]
inputSpike = inSpike.MakeNeuPLUSByteFile(deltat=500)
rewards = [reward_time, reward_core, reward_idx, reward_value]  # [time, core, address, value]

inputSpike.events_to_bytefile(events=events, 
                              byte_file_name=inputSpikeFilename, 
                              conf_events=rewards, 
                              weight_single=248.)

testSet.genExpConfFile(conffname)

SynTable = synTable.synapticTable(2, 1024, 1024)

for i in range(pattern_num):
    snn_weight['cortex-str'+str(i)] = np.random.rand(cortex_num*str_num)*parameters['wmax']
    
    SynTable.createConnections(source = np.sort(np.tile(snn_obj['cortex'], str_num)), 
                               destination = np.tile(snn_obj['str'+str(i)], cortex_num), 
                               mode = 'one', 
                               probability = 1, 
                               weight = snn_weight['cortex-str'+str(i)], 
                               wmax = parameters['wmax'], 
                               delay = 0, 
                               dmax = 0, 
                               synType = 'exc', 
                               trainable = True)
    
for i in range(pattern_num):
    for j in range(pattern_num):
        if i != j:
            SynTable.createConnections(source = snn_obj['str'+str(i)], 
                                        destination = snn_obj['str'+str(j)], 
                                        mode = 'all', 
                                        probability = 1, # 20/str_num, 
                                        weight = 6*20/str_num, 
                                        wmax = parameters['wmax'], 
                                        delay = 0, 
                                        dmax = 0, 
                                        synType = 'inh', 
                                        trainable = False)

# for i in range(pattern_num):
#     SynTable.createConnections(source = snn_obj['str'+str(i)], 
#                                 destination = snn_obj['str'+str(i)], 
#                                 mode = 'all', 
#                                 probability = 1, # 20/str_num, 
#                                 weight = 1.0625, 
#                                 wmax = parameters['wmax'], 
#                                 delay = 0, 
#                                 dmax = 0, 
#                                 synType = 'inh', 
#                                 trainable = False)


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
str_results_it = {}
str_results_ft = {}
str_results_ii = {}
str_results_fi = {}

img_test = [0]

start = 497
end = 500

# for i in range(pattern_num):
for i in img_test:
    str_idx = np.where((output_spike_idx >= 1024+i*str_num) & (output_spike_idx < 1024+(i+1)*str_num))[0]
    str_time = output_spike_time[str_idx]
    selected_initial_indices = np.where(str_time <simulation_duration/10)[0]
    str_results_it['str'+str(i)] = str_time[selected_initial_indices]
    str_results_ii['str'+str(i)] = output_spike_idx[str_idx[selected_initial_indices]]
    selected_final_indices = np.where(str_time > simulation_duration-simulation_duration/10)[0] 
    str_results_ft['str'+str(i)] = str_time[selected_final_indices]
    str_results_fi['str'+str(i)] = output_spike_idx[str_idx[selected_final_indices]]

for i in img_test:
    str_idx = np.where((output_spike_idx >= 1024+i*str_num) & (output_spike_idx < 1024+(i+1)*str_num))[0]
    str_time = output_spike_time[str_idx]
    selected_initial_indices = np.where((str_time >start) & (str_time <end))[0]
    str_results_it['str'+str(i)] = str_time[selected_initial_indices]
    str_results_ii['str'+str(i)] = output_spike_idx[str_idx[selected_initial_indices]]

plt.figure(figsize=(3, 0.5), dpi=1200)
plt.plot(str_results_it['str'+str(img_test[0])], str_results_ii['str'+str(img_test[0])]-(1024+img_test[0]*64), '|', color='black', markersize=2)
plt.xlim([start, end])
plt.ylim([0, 9])
plt.xticks(np.arange(start, end+0.1, 0.5), [''] * len(np.arange(start, end+0.1, 0.5)))
# plt.yticks(np.arange(0, 9.1, 9), [''] * len(np.arange(0, 9.1, 9)))
plt.yticks([])
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig('str_raster_plot.png', dpi=1200, transparent=True)

plt.figure(figsize=(3, 2), dpi=1200)
plt.plot(noise_spike_time, noise_spike_idx, '|', color='black', markersize=2, alpha=0.3)
plt.plot(input_spike_time, input_spike_idx, '|', color='black', markersize=2)
plt.xlim([start, end])
plt.ylim([0, 300])
plt.xticks(np.arange(start, end+0.1, 0.5), [''] * len(np.arange(start, end+0.1, 0.5)))
# plt.yticks(np.arange(0, 301, 50), [''] * len(np.arange(0, 301, 50)))
plt.yticks([])
ax = plt.gca()
ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig('input_raster_plot.png', dpi=1200, transparent=True)


cortex_idx = np.where(output_spike_idx<cortex_num)[0]
cortex_time = output_spike_time[cortex_idx]

# 2. Visualization
# plt.figure(dpi=600)
# plt.plot(cortex_time, output_spike_idx[cortex_idx], '.k', markersize=1)
# plt.xlabel('Time')
# plt.ylabel('Neuron index')
# plt.savefig('cortex_raster_plot.png', dpi=600)

fig, axes = plt.subplots(pattern_num, 1, figsize=(8, 2*pattern_num*str_num/32), dpi=600)
# fig, axes = plt.subplots(len(img_test), 1, figsize=(8, 2*len(img_test)*str_num/32), dpi=600)
for i, (ax, key) in enumerate(zip(axes, str_results_it)):
    times = str_results_it[key]
    indices = str_results_ii[key]
    ax.plot(times, indices+1, '.k', markersize=1.5)
    # ax.set_xlim([0, simulation_duration/10])
    ax.set_xlim([0, 3])
    ax.set_yticks(np.arange(np.min(indices), np.max(indices)+2, 16))
    ax.set_yticklabels([''] * len(np.arange(np.min(indices), np.max(indices)+2, 16)))
    
    if i == len(str_results_it) - 1:
        ax.xaxis.set_visible(True)
        # ax.set_xticks(np.arange(0, simulation_duration/10 + 1, 5))
        # ax.set_xticklabels([''] * len(np.arange(0, simulation_duration/10 + 1, 5)))
        ax.set_xticks(np.arange(0, 3 + 0.1, 0.5))
        ax.set_xticklabels([''] * len(np.arange(0, 3 + 0.1, 0.5)))
    else:
        ax.xaxis.set_visible(False)

plt.tight_layout()
plt.savefig('str_initial_raster_plot.png', dpi=600, transparent=False)


fig, axes = plt.subplots(pattern_num, 1, figsize=(8, 2*pattern_num*str_num/32), dpi=600)
# fig, axes = plt.subplots(len(img_test), 1, figsize=(8, 2*len(img_test)*str_num/32), dpi=600)
for i, (ax, key) in enumerate(zip(axes, str_results_it)):
    times = str_results_ft[key]
    indices = str_results_fi[key]
    ax.plot(times, indices+1, '.k', markersize=1.5)
    # ax.set_xlim([simulation_duration-simulation_duration/10, simulation_duration])
    ax.set_xlim([simulation_duration-3, simulation_duration])
    # ax.set_yticks(np.arange(np.min(indices), np.max(indices)+2, 16))
    # ax.set_yticklabels([''] * len(np.arange(np.min(indices), np.max(indices)+2, 16)))
    ax.yaxis.set_visible(False)
    if i == len(str_results_it) - 1:
        ax.xaxis.set_visible(True)
        # ax.set_xticks(np.arange(simulation_duration-simulation_duration/10, simulation_duration + 1, 5))        
        # ax.set_xticklabels([''] * len(np.arange(simulation_duration-simulation_duration/10, simulation_duration + 1, 5)))
        ax.set_xticks(np.arange(simulation_duration-3, simulation_duration + 0.1, 0.5))
        ax.set_xticklabels([''] * len(np.arange(simulation_duration-3, simulation_duration + 0.1, 0.5)))
    else:
        ax.xaxis.set_visible(False)

plt.tight_layout()
plt.savefig('str_final_raster_plot.png', dpi=600, transparent=False)


bins = np.arange(0, parameters['wmax']+2, 2)
before_weight_arrays = list(snn_weight.values())
before_weights = np.concatenate(before_weight_arrays)
plt.figure(dpi=600)
plt.hist(before_weights, bins=bins, edgecolor='black')
# plt.title('weight distribution before training')
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.savefig('initial_weight_distribution.png', dpi=600, transparent=True)

plt.figure(dpi=600)
plt.hist(synDecompiledList[2], bins=bins, edgecolor='black')
# plt.title('weight distribution after training')
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.savefig('final_weight_distribution.png', dpi=600, transparent=True)

# %% Evaluation
if pattern_mode == 0:
    acc_hist = np.zeros(int(simulation_duration//(pattern_duration*2*pattern_num)))
    avg_rate = np.zeros(pattern_num)
    answer_cnt = 0
    idx = 0
    for t in range(int(simulation_duration//(pattern_duration*2))):
        for i in range(pattern_num):
            str_idx = np.where((output_spike_idx >= 1024+i*str_num) & (output_spike_idx < 1024+(i+1)*str_num))[0]
            str_time = output_spike_time[str_idx]
            selected_initial_indices = np.where((str_time >= t*(pattern_duration*2)) & (str_time < (t+1)*(pattern_duration*2)))[0]
            avg_rate[i] = len(str_time[selected_initial_indices])/str_num
            if t in pattern_time_window[str(i)]:
                answer = i
            else:
                pass
                # raise Exception(f'An error occurred due to missing input label at time {t}~{t+1} sec!!!')
        if np.random.choice(np.where(avg_rate==np.max(avg_rate))[0]) == answer:
            answer_cnt += 1
        else:
            pass
        if (t+1)%pattern_num == 0:
            acc_hist[idx] = answer_cnt/pattern_num
            idx += 1
            answer_cnt = 0
    print(acc_hist)
    
if pattern_mode == 2:
    acc_hist = np.array([])
    avg_rate = np.zeros(pattern_num)
    answer_hist = np.array([])
    for i in range(len(pattern_array)):
        for j in range(pattern_num):
            str_idx = np.where((output_spike_idx >= 1024+j*str_num) & (output_spike_idx < 1024+(j+1)*str_num))[0]
            str_time = output_spike_time[str_idx]
            
            selected_initial_indices = np.where((str_time >= max_time_array[i]) & (str_time < max_time_array[i]+random_pattern[i]))[0]
            avg_rate[j] = len(str_time[selected_initial_indices])/str_num
            answer = pattern_array[i]
    
        if np.random.choice(np.where(avg_rate==np.max(avg_rate))[0]) == answer:
            answer_hist = np.append(answer_hist, 1)
        else:
            answer_hist = np.append(answer_hist, 0)
    
        if len(answer_hist) >= 100:
            acc_hist = np.append(acc_hist, np.sum(answer_hist[len(answer_hist)-100:len(answer_hist)])/100)
        else:
            # acc_hist = np.append(acc_hist, 0)
            acc_hist = np.append(acc_hist, np.sum(answer_hist/100))
    
    print(acc_hist)
x = max_time_array+random_pattern
y = acc_hist
# selected_x_idx = np.where(y > 0.1)[0]
# x = x[selected_x_idx]
# y = y[selected_x_idx]
plt.figure(figsize=(9.6, 3), dpi=1200)
plt.plot(x, y, color='black', linewidth=2)
plt.xlim([-10, 510])
plt.ylim([0, 1.01])
plt.xticks(np.arange(0, 500.1, 100), ['']*len(np.arange(0, 500.1, 100)))
plt.yticks(np.arange(0, 1.001, 0.2), ['']*len(np.arange(0, 1.001, 0.2)))
ax = plt.gca()
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
plt.savefig('accuracy.png', dpi=1200, transparent=True)

import h5py

with h5py.File('./result/exp2.h5', 'w') as f:
    f.create_dataset('accuracy', data=acc_hist)
    f.create_dataset('max_time_array', data=max_time_array)
    f.create_dataset('random_pattern', data=random_pattern)
    
# acc_hist_dic = {}
# max_time_dic = {}
# random_pattern_dic = {}
# for i in range(10):
#     acc_hist_dic[str(i)] = np.array([])
#     max_time_dic[str(i)] = np.array([])
#     random_pattern_dic[str(i)] = np.array([])
    
#     with h5py.File('./result/exp'+str(i)+'.h5', 'r') as f:
#         acc_hist_dic[str(i)] = np.append(acc_hist_dic[str(i)], f['accuracy'][:])
#         max_time_dic[str(i)] = np.append(max_time_dic[str(i)], f['max_time_array'][:])
#         random_pattern_dic[str(i)] = np.append(random_pattern_dic[str(i)], f['random_pattern'][:])

# plt.figure(figsize=(9.6, 2.8), dpi=1200)
# for i in range(10):
#     plt.plot(max_time_dic[str(i)]+random_pattern_dic[str(i)], acc_hist_dic[str(i)], linewidth=2)
# plt.xlim([-10, 510])
# plt.ylim([0, 1.01])
# plt.xticks(np.arange(0, 500.1, 100), ['']*len(np.arange(0, 500.1, 100)))
# plt.yticks(np.arange(0, 1.001, 0.2), ['']*len(np.arange(0, 1.001, 0.2)))
# ax = plt.gca()
# plt.savefig('accuracy.png', dpi=1200, transparent=True)