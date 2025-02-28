# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:51:13 2024

@author: David Kim
"""

# import sys, os
# os.chdir(r"C:\Users\user\Desktop\")
# sys.path.append(os.getcwd())

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt 

np.set_printoptions(suppress=True, threshold=np.inf, precision=4)

##################################
# Experimental variable settings #
##################################
defaultclock.dt = 0.1*ms
pattern_duration = 0.5      # [sec]
pattern_num = 10
simulation_duration = 200   # [sec]
reward_dt = 0.1             # [sec]
reward_mode = 1             # 0 : single 1 : multi
################
# Neuron model #
################
neuron_model = '''
dv/dt = -v/taum : 1 (unless refractory)
tau : second  
'''
neuron_param = {}
neuron_param['thr'] = 224       # Threshold
neuron_param['res'] = 0         # Reset
neuron_param['ref'] = 0*ms      # Refractory
neuron_param['taum'] = 20*ms     # Time constant

#################
# Synapse model #
#################
synapse_model = '''
w : 1
'''
synapse_onpre = '''
v_post += w
'''

synapse_pstdp_model = '''
dpre/dt = -pre/taua : 1 (event-driven)
dpost/dt = -post/taua : 1 (event-driven)
dd/dt = -d/taud : 1 (event-driven)
lpre : 1
w : 1
'''
synapse_pstdp_onpre = '''
pre += apre
v_post += w
con1 = d>1
w = clip(w+lpre+post*con1, 0, wmax)
'''
synapse_pstdp_onpost = '''
post += apost
con2 = d>1
lpre = pre*con2
'''

synapse_param = {}
synapse_param['taua'] = 20*ms                                                       # Neural trace time constant
synapse_param['taud'] = 20*ms
synapse_param['wmax'] = 16                                                          # Maximum synaptic weight 24
synapse_param['lr'] = 0.10                                                          # Learning rate
synapse_param['gamma'] = -1.0                                                       # Ratio of the anti-causal to causal update rate
synapse_param['apre'] = synapse_param['lr']*synapse_param['wmax']                   # Presynaptic neural trace update rate
synapse_param['apost'] = synapse_param['gamma']*synapse_param['apre']               # Postsynaptic neural trace update rate
synapse_param['ad'] = 1024

################
# Network size #
################
cortex_num = 300
str_num = 20           
snc_num = 20

#########################
# Input spike generator #
#########################
# 1. noise
noise_firing_rate = 3 # [Hz]
noise_spike_idx = np.random.randint(0, cortex_num, size=noise_firing_rate*simulation_duration*cortex_num)
noise_spike_time = np.round(np.sort(np.random.rand(noise_firing_rate*simulation_duration*cortex_num)*simulation_duration), 4)

# 2. pattern
pattern_firing_rate = 15 # [Hz]
shuffled_indices = np.random.permutation(simulation_duration)
pattern_time_windows = np.array_split(shuffled_indices, pattern_num)

pattern_time_window = {}
for i in range(pattern_num):
    pattern_time_window[str(i)] = np.sort(pattern_time_windows[i])

pattern_neuron_idx = {}
for i in range(pattern_num):
    pattern_neuron_idx[str(i)] = np.array([])
    pattern_neuron_idx[str(i)] = np.append(pattern_neuron_idx[str(i)], np.random.choice(range(cortex_num), cortex_num//4, replace=False))
    np.random.shuffle(pattern_neuron_idx[str(i)])

pattern_spike_idx = {}
pattern_spike_time = {}
for i in range(pattern_num):
    pattern_spike_idx[str(i)] = np.array([])
    pattern_spike_time[str(i)] = np.round(np.sort(np.random.rand(round(pattern_firing_rate*pattern_duration)*cortex_num//4)*pattern_duration), 4)
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
    if cortex_spike_time[i] - cortex_spike_time[i-1] <= 0.0001:
        cortex_spike_time[i] = cortex_spike_time[i-1]+0.0001

# 3. dopamine 
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
                    
                elif reward_mode == 1:
                    dopamine_spike_time[str(i)] = np.append(dopamine_spike_time[str(i)], np.random.rand(round(dopamine_firing_rate*pattern_duration))*pattern_duration+t)
                else:
                    pass
                dopamine_spike_idx[str(i)] = np.append(dopamine_spike_idx[str(i)], np.ones(round(dopamine_firing_rate*pattern_duration))*(j))


if reward_mode == 1:
    for i in range(pattern_num):
        sorted_indices = np.argsort(dopamine_spike_time[str(i)])
        dopamine_spike_time[str(i)] = dopamine_spike_time[str(i)][sorted_indices]
        dopamine_spike_idx[str(i)] = dopamine_spike_idx[str(i)][sorted_indices]
        for u in range(1, len(dopamine_spike_time[str(i)])):
            if dopamine_spike_time[str(i)][u] - dopamine_spike_time[str(i)][u-1] <= 0.0001:
                dopamine_spike_time[str(i)][u] = dopamine_spike_time[str(i)][u-1]+0.0001
else:
    pass

# 4. input & reward spike pattern visualization
plt.figure(dpi=600)
plt.title('Cortex spike raster plot')
plt.plot(input_spike_time, input_spike_idx, '.r', markersize=1, label='pattern')
plt.plot(noise_spike_time, noise_spike_idx, '.k', markersize=1, label='noise')
plt.legend()
plt.xlim([0, 10])
plt.savefig('input_raster_plot.png', dpi=600)

fig, axes = plt.subplots(pattern_num, 1, figsize=(8, 2*pattern_num), dpi=600)
for ax, key in zip(axes, dopamine_spike_time):
    times = dopamine_spike_time[key]
    indices = dopamine_spike_idx[key]
    ax.plot(times, indices, '.r', markersize=2)
    ax.set_title(f'str{key} dopamine')
    ax.set_xlim([0, simulation_duration/10])
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Index [#]')
plt.tight_layout()
plt.savefig('dopamine_rastor_plot1.png', dpi=600)

fig, axes = plt.subplots(pattern_num, 1, figsize=(8, 2*pattern_num), dpi=600)
for ax, key in zip(axes, dopamine_spike_time):
    times = dopamine_spike_time[key]
    indices = dopamine_spike_idx[key]
    ax.plot(times, indices, '.r', markersize=2)
    ax.set_title(f'str{key} dopamine')
    ax.set_xlim([simulation_duration-simulation_duration/10, simulation_duration])
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Index [#]')
plt.tight_layout()
plt.savefig('dopamine_rastor_plot2.png', dpi=600)

##########################
# Spiking neural network #
##########################
snn_obj = {}
snn_weight = {}

# 1. Creating layers
snn_obj['cortex'] = SpikeGeneratorGroup(cortex_num, cortex_spike_idx, cortex_spike_time*second)
snn_obj['cortex_mon'] = SpikeMonitor(snn_obj['cortex'])
for i in range(pattern_num):
    snn_obj['snc'+str(i)] = SpikeGeneratorGroup(snc_num, dopamine_spike_idx[str(i)], dopamine_spike_time[str(i)]*second)
    snn_obj['snc'+str(i)+'_mon'] = SpikeMonitor(snn_obj['snc'+str(i)])
    snn_obj['str'+str(i)] = NeuronGroup(N=str_num, model=neuron_model, threshold='v>thr', reset='v=res', refractory=neuron_param['ref'], namespace=neuron_param, method='exact')
    snn_obj['str'+str(i)+'_mon'] = SpikeMonitor(snn_obj['str'+str(i)])

# 2. Connecting layers
for i in range(pattern_num):
    snn_weight['cortex-str'+str(i)] = np.random.rand(cortex_num*str_num)*synapse_param['wmax']
    snn_obj['cortex-str'+str(i)] = Synapses(source=snn_obj['cortex'], target=snn_obj['str'+str(i)], model=synapse_pstdp_model, on_pre=synapse_pstdp_onpre, on_post=synapse_pstdp_onpost, namespace=synapse_param, method='exact')
    snn_obj['cortex-str'+str(i)].connect(condition=True)
    snn_obj['cortex-str'+str(i)].w = snn_weight['cortex-str'+str(i)]

for i in range(pattern_num):
    snn_obj['snc'+str(i)+'-str'+str(i)] = Synapses(source=snn_obj['snc'+str(i)], target=snn_obj['cortex-str'+str(i)], model='''''', on_pre='d_post+=ad', namespace=synapse_param, method='exact')
    snn_obj['snc'+str(i)+'-str'+str(i)].connect(condition='i==j%20')
    
for i in range(pattern_num):
    for j in range(pattern_num):
        if i != j:
            snn_obj['str'+str(i)+'-str'+str(j)] = Synapses(source=snn_obj['str'+str(i)], target=snn_obj['str'+str(j)], model=synapse_model, on_pre=synapse_onpre, namespace=synapse_param, method='exact')
            snn_obj['str'+str(i)+'-str'+str(j)].connect(condition=True)
            snn_obj['str'+str(i)+'-str'+str(j)].w = -6
            
for i in range(pattern_num):
    snn_obj['str'+str(i)+'-str'+str(i)] = Synapses(source=snn_obj['str'+str(i)], target=snn_obj['str'+str(i)], model=synapse_model, on_pre=synapse_onpre, namespace=synapse_param, method='exact')
    snn_obj['str'+str(i)+'-str'+str(i)].connect(condition='i!=j')
    snn_obj['str'+str(i)+'-str'+str(i)].w = -1.0625

# 5. Configure a spiking neural network and run it
network = Network()
for obj in snn_obj.keys():
    network.add(snn_obj[obj])
network.run(simulation_duration*second, report='stdout')

######################
# Simulation results #
######################
snc_results_bt = {}
snc_results_bi = {}
snc_results_at = {}
snc_results_ai = {}
str_results_bt = {}
str_results_bi = {}
str_results_at = {}
str_results_ai = {}

# 1. Filtering spike data
for i in range(pattern_num):
    snc_results_b_indices = [index for index, value in enumerate(snn_obj['snc'+str(i)+'_mon'].t/second) if value < simulation_duration/10]
    snc_results_bt['snc'+str(i)] = [snn_obj['snc'+str(i)+'_mon'].t[index] for index in snc_results_b_indices]
    snc_results_bi['snc'+str(i)] = [snn_obj['snc'+str(i)+'_mon'].i[index] for index in snc_results_b_indices]
    
    snc_results_a_indices = [index for index, value in enumerate(snn_obj['snc'+str(i)+'_mon'].t/second) if value > simulation_duration-simulation_duration/10]
    snc_results_at['snc'+str(i)] = [snn_obj['snc'+str(i)+'_mon'].t[index] for index in snc_results_a_indices]
    snc_results_ai['snc'+str(i)] = [snn_obj['snc'+str(i)+'_mon'].i[index] for index in snc_results_a_indices]
    
    str_results_b_indices = [index for index, value in enumerate(snn_obj['str'+str(i)+'_mon'].t/second) if value < simulation_duration/10]
    str_results_bt['str'+str(i)] = [snn_obj['str'+str(i)+'_mon'].t[index] for index in str_results_b_indices]
    str_results_bi['str'+str(i)] = [snn_obj['str'+str(i)+'_mon'].i[index] for index in str_results_b_indices]
    
    str_results_a_indices = [index for index, value in enumerate(snn_obj['str'+str(i)+'_mon'].t/second) if value > simulation_duration-simulation_duration/10]
    str_results_at['str'+str(i)] = [snn_obj['str'+str(i)+'_mon'].t[index] for index in str_results_a_indices]
    str_results_ai['str'+str(i)] = [snn_obj['str'+str(i)+'_mon'].i[index] for index in str_results_a_indices]

# 2. Visualization
fig, axes = plt.subplots(pattern_num, 1, figsize=(8, 2*pattern_num), dpi=600)
for ax, key in zip(axes, str_results_bt):
    times = str_results_bt[key]
    indices = str_results_bi[key]
    ax.plot(times, indices, '.k', markersize=2)
    ax.set_title(f'{key} initial spike raster plot')
    ax.set_xlim([0, simulation_duration/10])
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Index [#]')
plt.tight_layout()
plt.savefig(f'{key}_initial_rastor_plot.png', dpi=600)

fig, axes = plt.subplots(pattern_num, 1, figsize=(8, 2*pattern_num), dpi=600)
for ax, key in zip(axes, str_results_at):
    times = str_results_at[key]
    indices = str_results_ai[key]
    ax.plot(times, indices, '.k', markersize=2)
    ax.set_title(f'{key} final spike raster plot')
    ax.set_xlim([simulation_duration-simulation_duration/10, simulation_duration])
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Index [#]')
plt.tight_layout()
plt.savefig(f'{key}_final_rastor_plot.png', dpi=600)

bins = np.arange(0, synapse_param['wmax']+1, 1)
before_weight_arrays = list(snn_weight.values())
before_weights = np.concatenate(before_weight_arrays)
plt.figure(dpi=1200)
plt.hist(before_weights, bins=bins, edgecolor='black')
plt.title('initial weight distribution')
plt.savefig('initial_weight_distribution.png', dpi=600)

after_weights = np.array([])
for i in range(pattern_num):
    after_weights = np.append(after_weights, snn_obj['cortex-str'+str(i)].w)
plt.figure(dpi=1200)
plt.hist(after_weights, bins=bins, edgecolor='black')
plt.title('final weight distribution')
plt.savefig('final_weight_distribution.png', dpi=600)