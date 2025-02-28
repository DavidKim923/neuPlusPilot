# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:33:04 2024

@author: David Kim
"""


import sys,os

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

firing_rate = [500, 1000, 1500, 2000, 2500, 3000]
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
for w in weight:
    output[str(w)] = np.array([])
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
        testSet.setNeuronCoreConf([0], [20], [5], [256], [0], [0])
        
        # FPGA configuration
        testSet.setFPGAConfiguration("CHIP_CLOCK", 150)
        
        testSet.genExpConfFile(conffname)
        
        
        # %% Input setup
        inputSpike = inSpike.MakeNeuPLUSByteFile(deltat=50)
        # default_strength_integer = 52 (weight = 10)
        rate = np.linspace(f-500, f-1, 500)
        events = spikes.poissonSpike.PoissonSpikeTrains_Time_Neuron(neuronNum=np.linspace(0, 499, 500), 
                                                                    rate=rate, 
                                                                    dt=0.0001, 
                                                                    period=10, 
                                                                    offset=0)
        
        inputSpike.events_to_bytefile(events=events, 
                                      byte_file_name=inputSpikeFilename, 
                                      conf_events=[], 
                                      weight_single=w)
        
        
        # %% RUN
        elapsed_time = param.NeuPLUSRun('-GUIModeOff', '-conf', conffname)
        print("Result time : ", elapsed_time)
        
        
        # %% Analysis
        event_results = eventAnalysis.eventAnalysis(param.NeuPLUSResultFolder + "2025" +date.today().strftime("%m%d") + "/")
        if len(event_results.spikes[1]) == 0:
            freq = np.zeros(500)
        else:
            freq = np.array(event_results.getAverageFiringRate())
        output[str(w)] = np.append(output[str(w)], freq[:500])
        output[str(w)].reshape(-1)
        
    with open('./results/c_results/w'+str(w)+'.txt', 'w') as file:
        for i in range(len(output[str(w)])):
            file.write(str(output[str(w)][i])+'\n')

#%% visualization
# import numpy as np
# import matplotlib.pyplot as plt
# firing_rate = [500, 1000, 1500, 2000, 2500, 3000]
# weight = [  1.0625   ,   1.125   ,   1.1875   ,   1.25    ,   1.3125   ,  1.375   ,   
#             1.4375   ,   1.5     ,   1.5625   ,   1.625   ,   1.6875   ,  1.75    ,   
#             1.8125   ,   1.875   ,   1.9375   ,   2.      ,   2.125    ,  2.25    ,   
#             2.375    ,   2.5     ,   2.625    ,   2.75    ,   2.875    ,  3.      ,   
#             3.125    ,   3.25    ,   3.375    ,   3.5     ,   3.625    ,  3.75    ,  
#             3.875    ,   4.      ,   4.25     ,   4.5     ,   4.75     ,  5.      ,   
#             5.25     ,   5.5     ,   5.75     ,   6.      ,   6.25     ,  6.5     ,   
#             6.75     ,   7.      ,   7.25     ,   7.5     ,   7.75     ,  8.      ,   
#             8.5      ,   9.      ,   9.5      ,   10.     ,   10.5     ,  11.     ,   
#             11.5     ,   12.     ,   12.5     ,   13.     ,   13.5     ,  14.     ,   
#             14.5     ,   15.     ,   15.5     ,   16.     ,   17.      ,  18.     ,   
#             19.      ,   20.     ,   21.      ,   22.     ,   23.      ,  24.     ,   
#             25.      ,   26.     ,   27.      ,   28.     ,   29.      ,  30.     ,   
#             31.      ,   32.     ,   34.      ,   36.     ,   38.      ,  40.     ,   
#             42.      ,   44.     ,   46.      ,   48.     ,   50.      ,  52.     ,   
#             54.      ,   56.     ,   58.      ,   60.     ,   62.      ,  64.     ,   
#             68.      ,   72.     ,   76.      ,   80.     ,   84.      ,  88.     ,   
#             92.      ,   96.     ,   100.     ,   104.    ,   108.     ,  112.    ,   
#             116.     ,   120.    ,   124.     ,   128.    ,   136.     ,  144.    ,   
#             152.     ,   160.    ,   168.     ,   176.    ,   184.     ,  192.    ,   
#             200.     ,   208.    ,   216.     ,   224.    ,   232.     ,  240.    ,   248.]


datas = []
for w in weight:
    data = []
    f = open('./results/c_results/w'+str(w)+'.txt', 'r')
    lines = f.readlines()
    for line in lines:
        data.append(line)
    datas.append(data)

datas = np.random.rand(127, 3000) * 200

plt.figure(dpi=1200, figsize=(6.4, 4.8))
fig, ax = plt.subplots()
cax = ax.imshow(datas, interpolation='nearest', cmap='jet', aspect='auto', extent=[0, 3000, 1, 248])

ax.set_yscale('log')
plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000], ['', '', '', '', '', '', ''])
plt.gca().set_yticklabels([])

cbar = fig.colorbar(cax)
cbar.set_ticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
cbar.ax.set_yticklabels([''] * 9)

plt.savefig('./figures/c.png', transparent=True, dpi=1200)
