# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:21:50 2024

@author: David Kim
"""

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt 

np.set_printoptions(suppress=True, threshold=np.inf, precision=4)

##################################
# Experimental variable settings #
##################################
defaultclock.dt = 0.1*ms

pattern_duration = 0.5      # [sec]
simulation_duration = 100   # [sec]
reward_dt = 0.1             # [sec]

################
# Neuron model #
################
neuron_model = '''
dv/dt = -v/tau : 1 (unless refractory)
tau : second  
'''
neuron_param = {}
neuron_param['thr'] = 200       # Threshold
neuron_param['res'] = 0         # Reset
neuron_param['ref'] = 0*ms      # Refractory
neuron_param['tau'] = 20*ms     # Time constant

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
synapse_param['lr'] = 0.20                                                          # Learning rate
synapse_param['gamma'] = -1.0                                                       # Ratio of the anti-causal to causal update rate
synapse_param['apre'] = synapse_param['lr']*synapse_param['wmax']                   # Presynaptic neural trace update rate
synapse_param['apost'] = synapse_param['gamma']*synapse_param['apre']               # Postsynaptic neural trace update rate
synapse_param['ad'] = 1024

################
# Network size #
################
cortex_num = 200
# cortex2_num = 2
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

noise_spike_time = np.round(np.sort(np.random.rand(noise_firing_rate*simulation_duration*cortex_num)*simulation_duration), 4)
noise_spike_time = np.array(noise_spike_time)

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
 
a_pattern_spike_time = np.round(np.sort(np.random.rand(round(pattern_firing_rate*pattern_duration)*cortex_num//4)*pattern_duration), 4)
b_pattern_spike_time = np.round(np.sort(np.random.rand(round(pattern_firing_rate*pattern_duration)*cortex_num//4)*pattern_duration), 4)

input_spike_time = np.array([])
input_spike_idx = np.array([])

for i in range(simulation_duration):
    if i in a_pattern_window:
        input_spike_time = np.append(input_spike_time, a_pattern_spike_time+i)
        input_spike_idx = np.append(input_spike_idx, a_pattern_spike_idx)
    
    elif i in b_pattern_window:
        input_spike_time = np.append(input_spike_time, b_pattern_spike_time+i)
        input_spike_idx = np.append(input_spike_idx, b_pattern_spike_idx)
    else:
        pass

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

cortex_spike_time = np.array(cortex_spike_time) * second

# 3. dopamine 
dopamine_firing_rate = 2 # [Hz]
a_dopamine_spike_time = np.array([])
b_dopamine_spike_time = np.array([])
a_dopamine_spike_idx = np.array([])
b_dopamine_spike_idx = np.array([])

for time in a_pattern_window:
    for idx in range(snc1_num):
        random_uniform_list = np.random.rand(round(dopamine_firing_rate*pattern_duration))*pattern_duration+time
        # a_dopamine_spike_time = np.append(a_dopamine_spike_time, random_uniform_list)
        a_dopamine_spike_time = np.append(a_dopamine_spike_time, np.ones(round(dopamine_firing_rate*pattern_duration))*time)
        # a_dopamine_spike_time = np.append(a_dopamine_spike_time, np.ones(round(dopamine_firing_rate*pattern_duration))*(time+0.25))
        #
        a_dopamine_spike_idx = np.append(a_dopamine_spike_idx, np.ones(round(dopamine_firing_rate*pattern_duration))*(idx))
        #
sorted_indices = np.argsort(a_dopamine_spike_time)
a_dopamine_spike_time = a_dopamine_spike_time[sorted_indices]
a_dopamine_spike_idx = a_dopamine_spike_idx[sorted_indices]
for i in range(1, len(a_dopamine_spike_time)):
    if a_dopamine_spike_time[i] - a_dopamine_spike_time[i-1] <= 0.001:
        a_dopamine_spike_time[i] = a_dopamine_spike_time[i-1]+0.001
for time in b_pattern_window: 
    for idx in range(snc2_num):
        random_uniform_list = np.random.rand(round(dopamine_firing_rate*pattern_duration))*pattern_duration+time
        # b_dopamine_spike_time = np.append(b_dopamine_spike_time, random_uniform_list)
        b_dopamine_spike_time = np.append(b_dopamine_spike_time, np.ones(round(dopamine_firing_rate*pattern_duration))*time)
        # b_dopamine_spike_time = np.append(b_dopamine_spike_time, np.ones(round(dopamine_firing_rate*pattern_duration))*(time+0.25))
        #
        b_dopamine_spike_idx = np.append(b_dopamine_spike_idx, np.ones(round(dopamine_firing_rate*pattern_duration))*(idx))
        #
sorted_indices = np.argsort(b_dopamine_spike_time)
b_dopamine_spike_time = b_dopamine_spike_time[sorted_indices]
b_dopamine_spike_idx = b_dopamine_spike_idx[sorted_indices]    
for i in range(1, len(b_dopamine_spike_time)):
    if b_dopamine_spike_time[i] - b_dopamine_spike_time[i-1] <= 0.001:
        b_dopamine_spike_time[i] = b_dopamine_spike_time[i-1]+0.001


# 3. dopamine 
# dopamine_firing_rate = 10 # [Hz]
# a_dopamine_spike_time = np.array([])
# b_dopamine_spike_time = np.array([])
# a_dopamine_spike_idx = np.array([])
# b_dopamine_spike_idx = np.array([])

# for time in a_pattern_window:
#     for idx in range(snc1_num):
#         random_uniform_list = np.random.rand(round(dopamine_firing_rate*pattern_duration))*pattern_duration+time
#         a_dopamine_spike_time = np.append(a_dopamine_spike_time, random_uniform_list)
#         a_dopamine_spike_idx = np.append(a_dopamine_spike_idx, np.ones(snc1_num*round(dopamine_firing_rate*pattern_duration))*idx)
# sorted_indices = np.argsort(a_dopamine_spike_time)
# a_dopamine_spike_time = a_dopamine_spike_time[sorted_indices]
# a_dopamine_spike_idx = a_dopamine_spike_idx[sorted_indices]

# for i in range(1, len(a_dopamine_spike_time)):
#     if a_dopamine_spike_time[i] - a_dopamine_spike_time[i-1] <= 0.0001:
#         a_dopamine_spike_time[i] = a_dopamine_spike_time[i-1]+0.0001
# a_dopamine_spike_time = np.array(a_dopamine_spike_time) * second

# for time in b_pattern_window:
#     for idx in range(snc2_num):
#         random_uniform_list = np.random.rand(round(dopamine_firing_rate*pattern_duration))*pattern_duration+time
#         b_dopamine_spike_time = np.append(b_dopamine_spike_time, random_uniform_list)
#         b_dopamine_spike_idx = np.append(b_dopamine_spike_idx, np.ones(snc2_num*round(dopamine_firing_rate*pattern_duration))*idx)
# sorted_indices = np.argsort(b_dopamine_spike_time)
# b_dopamine_spike_time = b_dopamine_spike_time[sorted_indices]
# b_dopamine_spike_idx = b_dopamine_spike_idx[sorted_indices]     

# for i in range(1, len(b_dopamine_spike_time)):
#     if b_dopamine_spike_time[i] - b_dopamine_spike_time[i-1] <= 0.0001:
#         b_dopamine_spike_time[i] = b_dopamine_spike_time[i-1]+0.0001
# b_dopamine_spike_time = np.array(b_dopamine_spike_time) * second

plt.figure(dpi=1200)
plt.title('Cortex spike raster plot')
plt.plot(input_spike_time, input_spike_idx, '.r', markersize=1, label='pattern')
plt.plot(noise_spike_time, noise_spike_idx, '.k', markersize=1, label='noise')
plt.legend()
plt.xlim([0, 10])
plt.savefig('input_raster_plot.png', dpi=1200)

fig, (ax1, ax2) = plt.subplots(2, 1, dpi=1200)
ax1.plot(a_dopamine_spike_time, a_dopamine_spike_idx, '.r', markersize=2)
ax1.set_title('STR1 dopamine')
ax1.set_xlim([0, 10])
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Modulator index [#]')
ax2.plot(b_dopamine_spike_time, b_dopamine_spike_idx+snc1_num, '.r', markersize=2)
ax2.set_title('STR2 dopamine')
ax2.set_xlim([0, 10])
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Modulator index [#]')
plt.tight_layout()
plt.savefig('dopamine_raster_plot1.png', dpi=1200)

fig, (ax1, ax2) = plt.subplots(2, 1, dpi=1200)
ax1.plot(a_dopamine_spike_time, a_dopamine_spike_idx, '.r', markersize=2)
ax1.set_title('STR1 dopamine')
ax1.set_xlim([simulation_duration-10, simulation_duration])
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Modulator index [#]')
ax2.plot(b_dopamine_spike_time, b_dopamine_spike_idx+snc1_num, '.r', markersize=2)
ax2.set_title('STR2 dopamine')
ax2.set_xlim([simulation_duration-10, simulation_duration])
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Modulator index [#]')
plt.tight_layout()
plt.savefig('dopamine_raster_plot2.png', dpi=1200)
##########################
# Spiking neural network #
##########################
snn_obj = {}

# 1. Generating input spikes
snn_obj['cortex'] = SpikeGeneratorGroup(cortex_num, cortex_spike_idx, cortex_spike_time)
snn_obj['snc1'] = SpikeGeneratorGroup(snc1_num, a_dopamine_spike_idx, a_dopamine_spike_time*second)
snn_obj['snc2'] = SpikeGeneratorGroup(snc2_num, b_dopamine_spike_idx, b_dopamine_spike_time*second)

# 2. Create a layer consist of spiking neurons
snn_obj['str1'] = NeuronGroup(N=str1_num, model=neuron_model, threshold='v>thr', reset='v=res', refractory=neuron_param['ref'], namespace=neuron_param, method='exact')
snn_obj['str1'].tau = 20*ms
snn_obj['str2'] = NeuronGroup(N=str2_num, model=neuron_model, threshold='v>thr', reset='v=res', refractory=neuron_param['ref'], namespace=neuron_param, method='exact')
snn_obj['str2'].tau = 20*ms

# 3. Creating a spiking neuron monitor
snn_obj['cortex_monitor'] = SpikeMonitor(snn_obj['cortex'])

snn_obj['snc1_monitor'] = SpikeMonitor(snn_obj['snc1'])
snn_obj['snc2_monitor'] = SpikeMonitor(snn_obj['snc2'])    
snn_obj['str1_monitor'] = SpikeMonitor(snn_obj['str1'])
snn_obj['str2_monitor'] = SpikeMonitor(snn_obj['str2'])

# 4. Connecting layers with synapses
cortex_str1_weight = np.random.rand(cortex_num*str1_num)*synapse_param['wmax']
snn_obj['cortex-str1'] = Synapses(source=snn_obj['cortex'], target=snn_obj['str1'], model=synapse_pstdp_model, on_pre=synapse_pstdp_onpre, on_post=synapse_pstdp_onpost, namespace=synapse_param, method='exact')
snn_obj['cortex-str1'].connect(condition=True)
snn_obj['cortex-str1'].w = cortex_str1_weight

cortex_str2_weight = np.random.rand(cortex_num*str2_num)*synapse_param['wmax']
snn_obj['cortex-str2'] = Synapses(source=snn_obj['cortex'], target=snn_obj['str2'], model=synapse_pstdp_model, on_pre=synapse_pstdp_onpre, on_post=synapse_pstdp_onpost, namespace=synapse_param, method='exact')
snn_obj['cortex-str2'].connect(condition=True)
snn_obj['cortex-str2'].w = cortex_str2_weight

snn_obj['snc1-str1'] = Synapses(source=snn_obj['snc1'], target=snn_obj['cortex-str1'], model='''''', on_pre='d_post+=ad', namespace=synapse_param, method='exact')
snn_obj['snc1-str1'].connect(condition='i==j%20')
snn_obj['snc2-str2'] = Synapses(source=snn_obj['snc2'], target=snn_obj['cortex-str2'], model='''''', on_pre='d_post+=ad', namespace=synapse_param, method='exact')
snn_obj['snc2-str2'].connect(condition='i==j%20')

snn_obj['str1-str2'] = Synapses(source=snn_obj['str1'], target=snn_obj['str2'], model=synapse_model, on_pre=synapse_onpre, namespace=synapse_param, method='exact')
snn_obj['str1-str2'].connect(condition=True)
snn_obj['str1-str2'].w = -6

snn_obj['str2-str1'] = Synapses(source=snn_obj['str2'], target=snn_obj['str1'], model=synapse_model, on_pre=synapse_onpre, namespace=synapse_param, method='exact')
snn_obj['str2-str1'].connect(condition=True)
snn_obj['str2-str1'].w = -6

snn_obj['str1-str1'] = Synapses(source=snn_obj['str1'], target=snn_obj['str1'], model=synapse_model, on_pre=synapse_onpre, namespace=synapse_param, method='exact')
snn_obj['str1-str1'].connect(condition='i!=j')
snn_obj['str1-str1'].w = -1.0625

snn_obj['str2-str2'] = Synapses(source=snn_obj['str2'], target=snn_obj['str2'], model=synapse_model, on_pre=synapse_onpre, namespace=synapse_param, method='exact')
snn_obj['str2-str2'].connect(condition='i!=j')
snn_obj['str2-str2'].w = -1.0625

# 4-1. Creating a synapse monitor
# snn_obj['cortex-str1_monitor'] = StateMonitor(snn_obj['cortex-str1'], ['d', 'lpre', 'pre'], record=True)
# snn_obj['snc2-str2_monitor'] = StateMonitor(snn_obj['snc2-str2'], ['d', 'lpre'], record=True)

# 5. Configure a spiking neural network and run it
network = Network()
for obj in snn_obj.keys():
    network.add(snn_obj[obj])
network.run(simulation_duration*second, report='stdout')


# fig, (ax1, ax2) = plt.subplots(2, 1, dpi=1200)
# ax1.plot(snn_obj['cortex-str1_monitor'].t, snn_obj['cortex-str1_monitor'].d[0], linewidth=1)
# ax1.set_xlim([0, simulation_duration])
# ax1.set_xlabel('Time [sec]')
# ax1.set_title('Dopamine')
# ax2.plot(snn_obj['cortex-str1_monitor'].t, snn_obj['cortex-str1_monitor'].pre[0], linewidth=1, label='pre trace')
# ax2.plot(snn_obj['cortex-str1_monitor'].t, snn_obj['cortex-str1_monitor'].lpre[0], linewidth=1, label='lpre')
# ax2.plot(snn_obj['str1_monitor'].t[np.where(snn_obj['str1_monitor'].i==0)[0]], snn_obj['str1_monitor'].i[np.where(snn_obj['str1_monitor'].i==0)[0]], '.k',markersize=3, label='post spike')
# ax2.plot(snn_obj['snc1_monitor'].t, snn_obj['snc1_monitor'].i+1, '.r',markersize=3, label='dopamine spike')
# ax2.set_xlim([0, simulation_duration])
# ax2.set_xlabel('Time [sec]')
# ax2.set_title('lpre')
# plt.legend()
######################
# Simulation results #
######################

# 1. Filtering spike data
before_snc1_indices = [index for index, value in enumerate(snn_obj['snc1_monitor'].t/second) if value < 10]
before_snc1_t = [snn_obj['snc1_monitor'].t[index] for index in before_snc1_indices]
before_snc1_i = [snn_obj['snc1_monitor'].i[index] for index in before_snc1_indices]

before_snc2_indices = [index for index, value in enumerate(snn_obj['snc2_monitor'].t/second) if value < 10]
before_snc2_t = [snn_obj['snc2_monitor'].t[index] for index in before_snc2_indices]
before_snc2_i = [snn_obj['snc2_monitor'].i[index] for index in before_snc2_indices]

after_snc1_indices = [index for index, value in enumerate(snn_obj['snc1_monitor'].t/second) if value > simulation_duration-10]
after_snc1_t = [snn_obj['snc1_monitor'].t[index] for index in before_snc1_indices]
after_snc1_i = [snn_obj['snc1_monitor'].i[index] for index in before_snc1_indices]

after_snc2_indices = [index for index, value in enumerate(snn_obj['snc2_monitor'].t/second) if value > simulation_duration-10]
after_snc2_t = [snn_obj['snc2_monitor'].t[index] for index in before_snc2_indices]
after_snc2_i = [snn_obj['snc2_monitor'].i[index] for index in before_snc2_indices]

before_str1_indices = [index for index, value in enumerate(snn_obj['str1_monitor'].t/second) if value < 10]
before_str1_t = [snn_obj['str1_monitor'].t[index] for index in before_str1_indices]
before_str1_i = [snn_obj['str1_monitor'].i[index] for index in before_str1_indices]

before_str2_indices = [index for index, value in enumerate(snn_obj['str2_monitor'].t/second) if value < 10]
before_str2_t = [snn_obj['str2_monitor'].t[index] for index in before_str2_indices]
before_str2_i = [snn_obj['str2_monitor'].i[index] for index in before_str2_indices]

after_str1_indices = [index for index, value in enumerate(snn_obj['str1_monitor'].t/second) if value > simulation_duration-10]
after_str1_t = [snn_obj['str1_monitor'].t[index] for index in after_str1_indices]
after_str1_i = [snn_obj['str1_monitor'].i[index] for index in after_str1_indices]

after_str2_indices = [index for index, value in enumerate(snn_obj['str2_monitor'].t/second) if value > simulation_duration-10]
after_str2_t = [snn_obj['str2_monitor'].t[index] for index in after_str2_indices]
after_str2_i = [snn_obj['str2_monitor'].i[index] for index in after_str2_indices]


# 2. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, dpi=1200)
ax1.plot(before_str1_t, before_str1_i, '.k', markersize=1)
ax1.set_xlim([0, 10])
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Neuron index')
ax1.set_title('Before training STR1')
ax2.plot(before_str2_t, before_str2_i, '.k', markersize=1)
ax2.set_xlim([0, 10])
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Neuron index')
ax2.set_title('Before training STR2')
plt.tight_layout()
plt.savefig('before_raster_plot.png', dpi=1200)

fig, (ax1, ax2) = plt.subplots(2, 1, dpi=1200)
ax1.plot(after_str1_t, after_str1_i, '.k', markersize=1)
ax1.set_xlim([simulation_duration-10, simulation_duration])
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Neuron index')
ax1.set_title('After training STR1')
ax2.plot(after_str2_t, after_str2_i, '.k', markersize=1)
ax2.set_xlim([simulation_duration-10, simulation_duration])
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Neuron index')
ax2.set_title('After training STR2')
plt.tight_layout()
plt.savefig('after_raster_plot.png', dpi=1200)

bins = np.arange(0, synapse_param['wmax']+1, 1)
before_weight = np.append(cortex_str1_weight, cortex_str2_weight)
plt.figure(dpi=1200)
plt.hist(before_weight, bins=bins, edgecolor='black')
plt.title('weight distribution before training')
plt.savefig('before_weight_distribution.png', dpi=1200)

after_weight = np.append(snn_obj['cortex-str1'].w, snn_obj['cortex-str2'].w)
plt.figure(dpi=1200)
plt.hist(after_weight, bins=bins, edgecolor='black')
plt.title('weight distribution after training')
plt.savefig('after_weight_distribution.png', dpi=1200)


