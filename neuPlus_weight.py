# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:11:15 2023

@author: David Kim
"""

import sys,os

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime
import h5py

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
rate = 1500
ref = 5
tc = 20

threshold = [32, 64, 128, 256]

weight = [  1.0625   ,   1.125   ,   1.1875   ,   1.25    ,   1.3125   ,  1.375   ,   
            1.4375   ,   1.5     ,   1.5625   ,   1.625   ,   1.6875   ,  1.75    ,   
            1.8125   ,   1.875   ,   1.9375   ,   2.      ,   2.125    ,  2.25    ,   
            2.375    ,   2.5     ,   2.625    ,   2.75    ,   2.875    ,  3.      ,   
            3.125    ,   3.25    ,   3.375    ,   3.5     ,   3.625    ,  3.75    ,  
            3.875    ,   4.      ,   4.25     ,   4.5     ,   4.75     ,  5.      ,   
            5.25     ,   5.5     ,   5.75     ,   6.      ,   6.25     ,  6.5     ,   
            6.75     ,   7.      ,   7.25     ,   7.5     ,   7.75     ,  8.      ,   
            8.5      ,   9.      ,   9.5      ,   10.     ,   10.5     ,  11.     ,   
            11.5     ,   12.     ,   12.5     ,   13.     ,   13.5     ,  14.     ,   
            14.5     ,   15.     ,   15.5     ,   16.     ,   17.      ,  18.     ,   
            19.      ,   20.     ,   21.      ,   22.     ,   23.      ,  24.     ,   
            25.      ,   26.     ,   27.      ,   28.     ,   29.      ,  30.     ,   
            31.      ,   32.     ,   34.      ,   36.     ,   38.      ,  40.     ,   
            42.      ,   44.     ,   46.      ,   48.     ,   50.      ,  52.     ,   
            54.      ,   56.     ,   58.      ,   60.     ,   62.      ,  64.     ,   
            68.      ,   72.     ,   76.      ,   80.     ,   84.      ,  88.     ,   
            92.      ,   96.     ,   100.     ,   104.    ,   108.     ,  112.    ,   
            116.     ,   120.    ,   124.     ,   128.    ,   136.     ,  144.    ,   
            152.     ,   160.    ,   168.     ,   176.    ,   184.     ,  192.    ,   
            200.     ,   208.    ,   216.     ,   224.    ,   232.     ,  240.    ,   248.]

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf, suppress=True)
for th in threshold:
    output[str(th)] = np.array([])
    for w in weight:
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
        testSet.setExperimentConf("EXP_TIME", 10000)
        testSet.setExperimentConf("TIME_ACCEL_MODE", True)
        testSet.setExperimentConf("INTERNAL_ROUTING_MODE", False)
        testSet.setExperimentConf("INPUT_SPIKES_FILE", inputSpikeFilename)
        testSet.setExperimentConf("SYN_TABLE_FILE_PREFIX", synTableFilePrefix)
        testSet.setExperimentConf("SYN_TABLE_READ_FILE", synfname)
        testSet.setExperimentConf("SYN_TABLE_READ_START", 0)
        testSet.setExperimentConf("SYN_TABLE_READ_COUNT", 1024)
        
        # Neuron array
        testSet.setNeuronCoreConf([0], [tc], [ref], [th], [0], [0])
        # testSet.setNeuronCoreConf([1], [lc], [ref], [th], [0], [0])
        # testSet.setNeuronCoreConf([2], [lc], [ref], [th], [0], [0])
        
        # FPGA configuration
        testSet.setFPGAConfiguration("CHIP_CLOCK", 150)
        
        testSet.genExpConfFile(conffname)
        
        
        # %% Input setup
        inputSpike = inSpike.MakeNeuPLUSByteFile(deltat=100)
        events = spikes.poissonSpike.PoissonSpikeTrains_Time_Neuron(neuronNum=np.linspace(0, 9, 10), 
                                                                    rate=rate, 
                                                                    dt=0.0001, 
                                                                    period=1, 
                                                                    offset=0)
        
    
        inputSpike.events_to_bytefile(events=events, 
                                      byte_file_name=inputSpikeFilename, 
                                      conf_events=[], 
                                      weight_single=w)
        
        
        
        # %% RUN
        elapsed_time = param.NeuPLUSRun('-GUIModeOff', '-conf', conffname)
        print("Result time : ", elapsed_time)
        
        
        # %% Analysis
        event_results = eventAnalysis.eventAnalysis(param.NeuPLUSResultFolder + "2024" +date.today().strftime("%m%d") + "/")
        # spike = event_results.spikes[1]
        # print('spike number : ', len(np.where(spike == 1024)))
        if len(event_results.spikes[1]) == 0:
            output[str(th)] = np.append(output[str(th)], 0)
            with open('./results/b_results/th'+str(th)+'_w'+str(w)+'.txt', 'w') as file:
                file.write('0')
        else:
            freq = np.array(event_results.getAverageFiringRate())
            output[str(th)] = np.append(output[str(th)], np.sum(freq[:10])/10)
            with open('./results/b_results/th'+str(th)+'_w'+str(w)+'.txt', 'w') as file:
                file.write(str(np.sum(freq[:10])/10))
        
        print(output[str(th)])
        print(f'threhsold : {th}, weight : {w}')
        output[str(th)].reshape(-1)
        


#%%
import numpy as np
import matplotlib.pyplot as plt

for th in threshold:
    output[str(th)] = np.array([])
    for w in weight:
        f = open('./results/b_results/th'+str(th)+'_w'+str(w)+'.txt', 'r')
        line = f.read()
        output[str(th)] = np.append(output[str(th)], np.float(line))

plt.figure(dpi=1200, figsize=(6.4, 4.8))
plt.plot(weight, output['32'], 'o', markersize=4)
plt.plot(weight, output['64'], 'o', markersize=4)
plt.plot(weight, output['128'], 'o', markersize=4)
plt.plot(weight, output['256'], 'o', markersize=4)
plt.xscale('log')
plt.gca().set_xticklabels([])
plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180], ['', '', '', '', '', '', '', '', '', ''])
# plt.yticks([0, 200, 400, 600, 800, 1000], ['', '', '', '', '', ''])
plt.xlim([1, 248.1])
plt.ylim([0, 180.1])
# plt.ylim([0, 1000.1])
plt.grid(True)
plt.savefig('./figures/b.png', transparent=True, dpi=1200)