# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:11:15 2023

@author: David Kim
"""

import sys,os

os.chdir(r"D:\KIST\Project\neuPlus_LE_project")
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime

import neuralSim.parameters as param
import neuralSim.compiler as compiler
import neuralSim.synapticTable as synTable
import neuralSim.inputSpikesGen as inSpike
import neuralSim.eventAnalysis as eventAnalysis
import neuralSim.poissonSpike as spikes

####################################################################
# Default value of test
# refractory time = 5ms
# time constant = 20ms
####################################################################

output = {}

threshold = [32, 64, 128, 256]
firing_rate = [500, 1000, 1500, 2000, 2500, 3000]
firing_rate_delta = 500
tc = 0
ref = 0

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf, suppress=True)
for th in threshold:
    output[str(th)] = np.array([])
    for f in firing_rate:
    
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
        testSet.setExperimentConf("INTERNAL_ROUTING_MODE", False)
        testSet.setExperimentConf("INPUT_SPIKES_FILE", inputSpikeFilename)
        testSet.setExperimentConf("SYN_TABLE_FILE_PREFIX", synTableFilePrefix)
        testSet.setExperimentConf("SYN_TABLE_READ_FILE", synfname)
        testSet.setExperimentConf("SYN_TABLE_READ_START", 0)
        testSet.setExperimentConf("SYN_TABLE_READ_COUNT", 1024)
        
        # Neuron array
        testSet.setNeuronCoreConf([0], [tc], [ref], [th], [0], [0])
        
        # FPGA configuration
        testSet.setFPGAConfiguration("CHIP_CLOCK", 150)
        
        testSet.genExpConfFile(conffname)
        
        
        # %% Input setup
        inputSpike = inSpike.MakeNeuPLUSByteFile(deltat=50)
        rate = np.linspace(f-firing_rate_delta, f-1, firing_rate_delta)
        events = spikes.poissonSpike.PoissonSpikeTrains_Time_Neuron(neuronNum=np.linspace(0, firing_rate_delta-1, firing_rate_delta), 
                                                                    rate=rate, 
                                                                    dt=0.0001, 
                                                                    period=5, 
                                                                    offset=0)
        
        inputSpike.events_to_bytefile(events=events, 
                                      byte_file_name=inputSpikeFilename, 
                                      conf_events=[], 
                                      weight_single=14)
        
        
        # %% RUN
        elapsed_time = param.NeuPLUSRun('-GUIModeOff', '-conf', conffname)
        print("Result time : ", elapsed_time)
        
        
        # %% Analysis
        event_results = eventAnalysis.eventAnalysis(param.NeuPLUSResultFolder + "2025" +date.today().strftime("%m%d") + "/")
        freq = np.array(event_results.getAverageFiringRate())
        
        
        output[str(th)] = np.append(output[str(th)], freq[:500])
        output[str(th)].reshape(-1)

    with open('./results/a_results/th'+str(th)+'.txt', 'w') as file:
        for i in range(len(output[str(th)])):
            file.write(str(output[str(th)][i])+'\n')

#%% visualization

import matplotlib.pyplot as plt
import numpy as np
threshold = [32, 64, 128, 256]
output = {}
for th in threshold:
    output[str(th)] = np.array([])
    f = open('./results/a_results/th'+str(th)+'.txt', 'r')
    lines = f.readlines()
    for line in lines:
        output[str(th)] = np.append(output[str(th)], np.float(line))

plt.figure(dpi=1200, figsize=(6.4, 4.8))
plt.plot(np.linspace(0, 999, 1000), output['32'], 'o', markersize=4)
plt.plot(np.linspace(0, 999, 1000), output['64'], 'o', markersize=4)
plt.plot(np.linspace(0, 999, 1000), output['128'], 'o', markersize=4)
plt.plot(np.linspace(0, 999, 1000), output['256'], 'o', markersize=4)
plt.gca().set_xticklabels([])
plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000], ['', '', '', '', '', '', ''])
plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180], ['', '', '', '', '', '', '', '', '', ''])
plt.xlim([0, 3000.1])
plt.ylim([0, 180.1])
plt.grid(True)
plt.savefig('./figures/f_in_f_out.png', transparent=True, dpi=1200)