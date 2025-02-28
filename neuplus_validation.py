# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:12:05 2022

@author: jongkil

This script is for testing a pre-synaptic event driven learning module.
"""

import sys,os
# os.chdir("C:/Users/User/Desktop/LJP/NeuPlus/neurosynaptic-comp-main")
# sys.path.append(os.getcwd())

import h5py
import pickle
import neuralSim.parameters as param
import numpy as np
import neuralSim.compiler as compiler

import neuralSim.synapticTable as synTable
import neuralSim.inputSpikesGen as inSpike
import neuralSim.eventAnalysis as eventAnalysis

import matplotlib.pyplot as plt
import neuralSim.poissonSpike as spikes
import neuralSim.MNISTdataset as MNIST
import time as t

from datetime import date
import datetime
import brian2 as br
import random
np.set_printoptions(threshold=sys.maxsize, linewidth=145)


def checkAccurateTimeConst(timeConstant, refractory, timeAccelTimestep, timeAccelResolution):
    unitTimestep = timeAccelTimestep # ms
    updateCycle = 2**timeAccelResolution # count from MSB of mantissa, max 10 
    
    ChipPrescaler = (timeConstant * 0.693147 / updateCycle / unitTimestep) - 1
    if ChipPrescaler <= 0:
        ChipPrescaler = 0.1
    
    ActualTimeconst = (int(ChipPrescaler) + 1) * unitTimestep * updateCycle / 0.693147
    ActualArrayTimer = unitTimestep * (int(ChipPrescaler) + 1) / (1024/updateCycle)
    ActualRefractory = (int((refractory / ActualArrayTimer)) >> (10 - timeAccelResolution)) * 2**(10-timeAccelResolution) * ActualArrayTimer
    
    propUnitstep = timeConstant * 0.693147 / updateCycle
    
    return ActualTimeconst, ActualRefractory

def Real2Fp34(realVal, realValmax=999):
    
    mentisaBit = 4
    DecimalPoint = 0
    
    realVal = np.clip(realVal, 0, realValmax)
    
    if (realVal >= 1.0625):
        realVal_positional = float('{:0.5f}'.format(realVal.item()))
        floatSplit = str(realVal_positional).split('.')
        
        a = np.array(floatSplit, dtype=np.int64).T[0]
        b = np.array(floatSplit, dtype=np.int64).T[1]
        
        bLen = len(floatSplit[1])
        
        Exponent = len("{0:b}".format(a.item()))-1
        
        tempMentisa = 0
        for i in range(mentisaBit):
            b = b * 2
            flag = (np.array((b / (10**bLen)), dtype=np.int64) == 1) + 0
            
            tempMentisa = tempMentisa + 2**(mentisaBit - i - 1) * flag
            b = b - (10**bLen) * flag
        
        Mentisa = (((a - 2**Exponent) << mentisaBit) + np.array(tempMentisa, dtype=np.int64)) >> Exponent
        Mentisa = 0 if (Mentisa < 0) else Mentisa
        
        if type(Exponent) != np.ndarray:
            Exponent = np.array([Exponent])
        
        if type(Mentisa) != np.ndarray:
            Mentisa = np.array([Mentisa])
            
        zeroIndex = np.where(Exponent + Mentisa == 0)[0]
        
        for i in range(mentisaBit):
            DecimalPoint += ((Mentisa >> i) % 2) * (1/(2**(mentisaBit - i)))
            
        realVal = (1 + DecimalPoint) * (2**Exponent)
        
        if len(zeroIndex) > 0:
            realVal[zeroIndex] = 0
    else :
        realVal = 0
    
        
    return realVal

def getNearestQValue(realVal, rounding='ceil'):
    
    reference = np.array([  0.      ,   1.0625   ,   1.125   ,   1.1875   ,   1.25    ,   1.3125   ,
                            1.375   ,   1.4375   ,   1.5     ,   1.5625   ,   1.625   ,   1.6875   ,
                            1.75    ,   1.8125   ,   1.875   ,   1.9375   ,   2.      ,   2.125    ,
                            2.25    ,   2.375    ,   2.5     ,   2.625    ,   2.75    ,   2.875    ,
                            3.      ,   3.125    ,   3.25    ,   3.375    ,   3.5     ,   3.625    ,
                            3.75    ,   3.875    ,   4.      ,   4.25     ,   4.5     ,   4.75     ,
                            5.      ,   5.25     ,   5.5     ,   5.75     ,   6.      ,   6.25     ,
                            6.5     ,   6.75     ,   7.      ,   7.25     ,   7.5     ,   7.75     ,
                            8.      ,   8.5      ,   9.      ,   9.5      ,   10.     ,   10.5     ,
                            11.     ,   11.5     ,   12.     ,   12.5     ,   13.     ,   13.5     ,
                            14.     ,   14.5     ,   15.     ,   15.5     ,   16.     ,   17.      ,
                            18.     ,   19.      ,   20.     ,   21.      ,   22.     ,   23.      ,
                            24.     ,   25.      ,   26.     ,   27.      ,   28.     ,   29.      ,
                            30.     ,   31.      ,   32.     ,   34.      ,   36.     ,   38.      ,
                            40.     ,   42.      ,   44.     ,   46.      ,   48.     ,   50.      ,
                            52.     ,   54.      ,   56.     ,   58.      ,   60.     ,   62.      ,
                            64.     ,   68.      ,   72.     ,   76.      ,   80.     ,   84.      ,
                            88.     ,   92.      ,   96.     ,   100.     ,   104.    ,   108.     ,
                            112.    ,   116.     ,   120.    ,   124.     ,   128.    ,   136.     ,
                            144.    ,   152.     ,   160.    ,   168.     ,   176.    ,   184.     ,
                            192.    ,   200.     ,   208.    ,   216.     ,   224.    ,   232.     ,
                            240.    ,   248.                                                          ])
    
    if (type(realVal) is not np.ndarray):
        
        floor = reference[np.where((reference - realVal) >= 0)[0][0]]
        ceil  = reference[np.where((realVal - reference) >= 0)[0][-1]]
        
        # Assume realVal as scalar
        if (rounding == 'random'):
            # Random package required
            return random.choice([floor, ceil])
        
        elif (rounding == 'floor'):
            return floor
        
        elif (rounding == 'ceil'):
            return ceil
            
    else:
        # Assume realVal as vector which has size
        
        floor = np.zeros_like(realVal)
        ceil  = np.zeros_like(realVal)
        rand  = np.zeros_like(realVal)
        
        for i in range(len(realVal)):
            floor[i] = reference[np.where((reference - realVal[i]) >= 0)[0][0]]
            ceil[i]  = reference[np.where((realVal[i] - reference) >= 0)[0][-1]]
            rand[i]  = random.choice([floor[i], ceil[i]])
        
        if (rounding == 'random'):
            return rand
        
        elif (rounding == 'floor'):
            return floor
        
        elif (rounding == 'ceil'):
            return ceil

# %%

parameters                      = {}
parameters['ApreMax']           = 16.
parameters['causalRate']        = -4.0

# parameters['idxArr_010']        = np.array([0, 1, 0])
# parameters['idxArr_101']        = np.array([1, 0, 1])
# parameters['idxArr_0011']       = np.array([0, 0, 1, 1])
# parameters['idxArr_0101']       = np.array([0, 1, 0, 1])
# parameters['idxArr_0110']       = np.array([0, 1, 1, 0])
# parameters['idxArr_1001']       = np.array([1, 0, 0, 1])
# parameters['idxArr_1100']       = np.array([1, 1, 0, 0])
# parameters['idxArr_1010']       = np.array([0, 0, 1, 0])


# parameters['timeArr_3spike_ex1'] = np.array([50,    80,     110])
# parameters['timeArr_3spike_ex2'] = np.array([40,    70,     120])
# parameters['timeArr_3spike_ex3'] = np.array([20,    75,     100])
# parameters['timeArr_3spike_ex4'] = np.array([10,    100,    110])
# parameters['timeArr_3spike_ex5'] = np.array([30,    40,     150])

# parameters['timeArr_4spike_ex1'] = np.array([55, 66, 85, 95])
# # parameters['timeArr_4spike_ex1'] = np.array([58,    415,     834,    849])
# parameters['timeArr_4spike_ex2'] = np.array([20,    40,     70,     120])
# parameters['timeArr_4spike_ex3'] = np.array([5,     20,     75,     100])
# parameters['timeArr_4spike_ex4'] = np.array([10,    100,    110,    150])
# parameters['timeArr_4spike_ex5'] = np.array([58,    415,     834,    849])    




randomSize = 20
parameters['timeArr_random']    = np.sort(np.copy(np.random.randint(0, 200, size=(randomSize), dtype=int)), axis=0)*10
parameters['timeArr_random']    = np.unique(parameters['timeArr_random'])
parameters['idxArr_random']     = np.round(np.random.rand(len(parameters['timeArr_random']))).astype(int)

# %% Input Spike Pattern Selection

# parameters['idxArr']            = parameters['idxArr_1010']
# parameters['timeArr']           = parameters['timeArr_4spike_ex1']

parameters['idxArr']            = np.array([0, 1, 0, 0])
parameters['timeArr']           = np.array([55, 70, 80, 95])

# %% 

parameters['idxArrNeu']         = parameters['idxArr']*(1)
parameters['timeArrBr']         = parameters['timeArr']*br.ms
parameters['timeArrNeu']        = parameters['timeArr']*0.001

parameters['sourceNeuronIdx']   = np.array([0])
parameters['destNeuronIdx']     = np.array([1])

parameters['taum']              = 20
parameters['tauLC']             = 20.
parameters['tref']              = 0.
parameters['vth']               = 100
parameters['wmax']              = 248.
parameters['winit']             = 64.
parameters['modulation']        = 0.

# %%

br.defaultclock.dt              = (0.5)*br.ms
inputSpikeFilename              = "testByte.nam"
synTableFilePrefix              = "SynTableWrite"
fname                           = "testExpConf.exp"
nfname                          = "testNeuronConf.nac"
conffname                       = "neuplusconf.txt"
synfname                        = "testRead.dat"

# %% Experiment Configuration
testSet = compiler.expSetupCompiler(fname, nfname)

#Experiment setup
testSet.setExperimentConf("EVENT_SAVE_MODE",            True)
testSet.setExperimentConf("EXP_TIME",                   100)
testSet.setExperimentConf("TIME_ACCEL_MODE",            True)
testSet.setExperimentConf("INTERNAL_ROUTING_MODE",      True)
testSet.setExperimentConf("INPUT_SPIKES_FILE",          inputSpikeFilename)
testSet.setExperimentConf("SYN_TABLE_FILE_PREFIX",      synTableFilePrefix)
testSet.setExperimentConf("SYN_TABLE_READ_FILE",        synfname)
testSet.setExperimentConf("SYN_TABLE_READ_START",       0)
testSet.setExperimentConf("SYN_TABLE_READ_COUNT",       1024)

####################################################################
# %% FPGA Configuration

testSet.setFPGAConfiguration("CHIP_CLOCK",                      150)
testSet.setFPGAConfiguration("ROUTER_SYN_OUTPUT_DISABLE", 0)

####################################################################
# %% Learning Engine Configuration

testSet.setLearningEngine("LC_TIMECONSTANT",                0,                                              parameters['tauLC'])
testSet.setLearningEngine("LC_TIMECONSTANT",                1,                                              parameters['tauLC'])

testSet.setLearningEngine("MAX_SYN_WEIGHT",                 0,                                              parameters['wmax'])
testSet.setLearningEngine("MAX_SYN_WEIGHT",                 1,                                              parameters['wmax'])

testSet.setLearningEngine("TRACE_UPDATE_AMOUNT",            0,          0,                                  parameters['ApreMax'])
testSet.setLearningEngine("TRACE_UPDATE_AMOUNT",            1,          0,                                  parameters['ApreMax'])

testSet.setLearningEngine("POST_TRACE_SCALE",               0,          0,                                  parameters['causalRate'])
testSet.setLearningEngine("POST_TRACE_SCALE",               1,          0,                                  parameters['causalRate'])

testSet.setLearningEngine("LEARNING_RULE",                  0,          parameters['modulation'],           0,          0,        0,        0)
testSet.setLearningEngine("LEARNING_RULE",                  1,          parameters['modulation'],           0,          0,        0,        0)

####################################################################
# %% input spike pattern generation

inputSpike          = inSpike.MakeNeuPLUSByteFile(deltat=1000)
start_time          = datetime.datetime.now()

input_spike_time    = parameters['timeArrNeu']
input_spike_idx     = parameters['idxArrNeu']
events              = [input_spike_time, input_spike_idx]


inputSpike.events_to_bytefile(events=events, 
                              byte_file_name=inputSpikeFilename, 
                              conf_events=[], 
                              weight_single=parameters['vth']+10)

end_time = datetime.datetime.now()

inputSetupTime = end_time - start_time
####################################################################
# %% Neuron Core Configuration

testSet.setNeuronCoreConf([0],                          # Core Number
                          [parameters['taum']],         # Taum
                          [parameters['tref']],         # Refractory
                          [parameters['vth']],          # Vth
                          [0],                          # Synaptic Gain
                          [0])                          # Stochasticity

testSet.setNeuronCoreConf([1],                          # Core Number
                          [parameters['taum']],         # Taum
                          [parameters['tref']],         # Refractory
                          [parameters['vth']],          # Vth
                          [0],                          # Synaptic Gain
                          [0])                          # Stochasticity

testSet.genExpConfFile(conffname)

SynTable = synTable.synapticTable(1, 1024, 1024)

SynTable.createConnections(     source          =   parameters['sourceNeuronIdx'],
                                destination     =   parameters['destNeuronIdx'],
                                mode            =   'one',
                                probability     =   1,
                                weight          =   parameters['winit'],
                                wmax            =   parameters['wmax'],
                                delay           =   0,
                                dmax            =   0,
                                synType         =   'exc',
                                trainable       =   True)

SynMapCompiled = compiler.synMapCompiler(synTableFilePrefix)
SynMapCompiled.generateSynMap(synTable=SynTable, 
                              inputNeurons=[range(2*1024)])

####################################################################
# %% Run on NeuPLUS system
elapsed_time = param.NeuPLUSRun('-GUIModeOff', '-conf', conffname)
print("Result time : ", elapsed_time)
####################################################################
# %% Event analysis

event_results = eventAnalysis.eventAnalysis(param.NeuPLUSResultFolder + '2024' + date.today().strftime("%m%d") + "/")

spike = event_results.spikes[1]
time  = event_results.spikes[0]

####################################################################
# %% Synapse table read

synMapDecompile     = compiler.synMapDecompiler(synfname)
synTable_read       = synTable.synapticTable()
synDecompiledList   = synMapDecompile.decompileSynMap()
synTable_read       = synTable_read.createFromSynapseList(synDecompiledList)

# %% Neuron Parameters

neuron_param                        = {}

neuron_param['timeAccelTimestep']   = 0.5
neuron_param['timeAccelResolution'] = 5

neuron_param['taum']                = parameters['taum']
neuron_param['refractory']          = parameters['tref']

neuron_param['taum']                = checkAccurateTimeConst(neuron_param['taum'],
                                                              neuron_param['refractory'],
                                                              neuron_param['timeAccelTimestep'],
                                                              neuron_param['timeAccelResolution'])[0]
neuron_param['refractory']          = checkAccurateTimeConst(neuron_param['taum'],
                                                              neuron_param['refractory'],
                                                              neuron_param['timeAccelTimestep'],
                                                              neuron_param['timeAccelResolution'])[1]

neuron_param['taum']                *= br.ms
neuron_param['refractory']          *= br.ms

neuron_param['Vreset']              = 'v=0'
neuron_param['Vth']                 = parameters['vth']
neuron_param['Vth']                 = 'v>'+str(neuron_param['Vth'])

neuron_param['model']               = '''
                                      dv/dt       = -v / taum : 1 (unless refractory)
                                      taum        : second
                                      '''

# %% Synapse Parameters

synapse_param                       = {}
synapse_param['taupre']             = neuron_param['taum']
synapse_param['taupost']            = neuron_param['taum']
synapse_param['wmax']               = parameters['wmax']
synapse_param['winit']              = parameters['winit']
synapse_param['lr']                 = 1

synapse_param['ApreMax']            = parameters['ApreMax']
synapse_param['post_trace_scale']   = parameters['causalRate']

if (synapse_param['post_trace_scale'] < 0):    
    synapse_param['ApostMax']   = Real2Fp34(synapse_param['ApreMax']*(1/abs(synapse_param['post_trace_scale'])))
else :
    synapse_param['ApostMax']   = Real2Fp34(synapse_param['ApreMax']*(synapse_param['post_trace_scale']))

# Ground truth synapse model
synapse_param['model']      = '''
                              w : 1
                              wmax : 1
                              taupre : second
                              taupost : second
                              ApreMax : 1
                              ApostMax : 1
                              dApre/dt = -Apre / taupre : 1 (clock-driven)
                              dApost/dt = -Apost / taupost : 1 (clock-driven)
                              lpre : 1
                              '''
                              
synapse_param['on_pre']     =   '''
                                w = clip(w-Apost+lpre, 0, wmax)
                                lpre = 0
                                Apre = Apre + ApreMax  
                                '''
                              
synapse_param['on_post']    =   '''
                                lpre = Apre
                                Apost = Apost + ApostMax
                                '''

# Floating Point 3.4 Formatted synapse model
synapse_param['Fp34_model'] =         '''
                                      wmax : 1
                                      taupre : second
                                      taupost : second
                                      ApreMax : 1
                                      ApostMax : 1
                                      
                                      dApre/dt = -Apre / taupre : 1 (clock-driven)
                                      dApost/dt = -Apost / taupost : 1 (clock-driven)
                                      
                                      Real2Fp34_qApre : 1
                                      Real2Fp34_qApost : 1
                                      Real2Fp34_lpre : 1
                                      Real2Fp34_w : 1
                                      
                                      is_onpre : 1
                                      is_onpost : 1
                                      
                                      getNearest_qApre : 1
                                      getNearest_qApost : 1
                                      getNearest_lpre : 1
                                      getNearest_w : 1
                                      '''
                              
synapse_param['Fp34_on_pre_trace'] =    '''
                                        is_onpost = 0
                                        is_onpre = 1
                                        '''

synapse_param['Fp34_on_post_trace'] =   '''
                                        is_onpre = 0
                                        is_onpost = 1
                                        '''
                                        
# %% SNN Network Configuration - Neurons

input_spike_idx                 = parameters['idxArr']
input_spike_time                = parameters['timeArrBr']
snn_obj                         = {}
snn_obj['background']           = br.SpikeGeneratorGroup(N          = 2,
                                                          indices    = input_spike_idx,
                                                          times      = input_spike_time)

snn_obj['spk_mon_background']   = br.SpikeMonitor(source=snn_obj['background'],
                                                  record=[0, 1])

                                      
snn_obj['Fp34_neurons']         = br.NeuronGroup(N          =   2,
                                                  model      =   neuron_param['model'],
                                                  method     =   'exact',
                                                  threshold  =   neuron_param['Vth'],
                                                  reset      =   neuron_param['Vreset'],
                                                  refractory =   neuron_param['refractory'],
                                                  name       =   "Fp34_neurons"
                                                  )

snn_obj['neurons']              = br.NeuronGroup(N          =   2,
                                                  model      =   neuron_param['model'],
                                                  method     =   'exact',
                                                  threshold  =   neuron_param['Vth'],
                                                  reset      =   neuron_param['Vreset'],
                                                  refractory =   neuron_param['refractory'],
                                                  name       =   "neurons"
                                                  )

snn_obj['Fp34_neurons'].taum    = neuron_param['taum']
snn_obj['Fp34_neurons'].v = 0
snn_obj['spk_mon_Fp34_neurons'] = br.SpikeMonitor(source=snn_obj['Fp34_neurons'], record='True')
snn_obj['st_mon_Fp34_neurons']  = br.StateMonitor(source=snn_obj['Fp34_neurons'], variables=['v'], record=[0, 1])

snn_obj['neurons'].taum         = neuron_param['taum']
snn_obj['neurons'].v = 0
snn_obj['spk_mon_neurons']      = br.SpikeMonitor(source=snn_obj['neurons'], record='True')
snn_obj['st_mon_neurons']       = br.StateMonitor(source=snn_obj['neurons'], variables=['v'], record=[0, 1])

# %% SNN Network Configuration - Synaptic connections

snn_obj['syn_background']       = br.Synapses(source  =   snn_obj['background'],
                                              target  =   snn_obj['neurons'],
                                              model   =   '''
                                                          s : 1
                                                          ''',
                                              on_pre  =   '''
                                                          v_post += s
                                                          ''')

snn_obj['Fp34_syn_background'] = br.Synapses(source  =   snn_obj['background'],
                                              target  =   snn_obj['Fp34_neurons'],
                                              model   =   '''
                                                          s : 1
                                                          ''',
                                                    
                                              on_pre  =   '''
                                                          v_post += s
                                                          ''')

snn_obj['syn_background'].connect(condition='i==j')
snn_obj['syn_background'].s = parameters['wmax']

snn_obj['Fp34_syn_background'].connect(condition='i==j')
snn_obj['Fp34_syn_background'].s = parameters['wmax']


snn_obj['Fp34_synapse'] = br.Synapses(source    =   snn_obj['Fp34_neurons'],
                                      target    =   snn_obj['Fp34_neurons'],
                                      model     =   synapse_param['Fp34_model'],
                                      on_pre    =   synapse_param['Fp34_on_pre_trace'],
                                      on_post   =   synapse_param['Fp34_on_post_trace'],
                                      method    =   "exact",
                                      name      =   'Fp34_synapse')


snn_obj['Fp34_synapse'].connect(i=0, j=1)
snn_obj['Fp34_synapse'].taupre          = synapse_param['taupre']
snn_obj['Fp34_synapse'].taupost         = synapse_param['taupost']
snn_obj['Fp34_synapse'].ApreMax         = synapse_param['ApreMax']
snn_obj['Fp34_synapse'].ApostMax        = synapse_param['ApostMax']
snn_obj['Fp34_synapse'].wmax            = synapse_param['wmax']

snn_obj['Fp34_synapse'].Real2Fp34_w               = synapse_param['winit']
snn_obj['Fp34_synapse'].Real2Fp34_lpre            = 0
snn_obj['Fp34_synapse'].Real2Fp34_qApre           = 0
snn_obj['Fp34_synapse'].Real2Fp34_qApost          = 0

snn_obj['Fp34_synapse'].getNearest_w              = synapse_param['winit']
snn_obj['Fp34_synapse'].getNearest_lpre           = 0
snn_obj['Fp34_synapse'].getNearest_qApre          = 0
snn_obj['Fp34_synapse'].getNearest_qApost         = 0

snn_obj['Fp34_synapse'].is_onpre        = 0
snn_obj['Fp34_synapse'].is_onpost       = 0

snn_obj['st_mon_Fp34_syn_neurons']      = br.StateMonitor(source    =   snn_obj['Fp34_synapse'],
                                                          variables =   ['Real2Fp34_w',  'Real2Fp34_lpre',  'Real2Fp34_qApre',  'Real2Fp34_qApost',
                                                                          'getNearest_w', 'getNearest_lpre', 'getNearest_qApre', 'getNearest_qApost'],
                                                          record    =   True)

# %%
snn_obj['synapse'] = br.Synapses(source    =   snn_obj['neurons'],
                                  target    =   snn_obj['neurons'],
                                  model     =   synapse_param['model'],
                                  on_pre    =   synapse_param['on_pre'],
                                  on_post   =   synapse_param['on_post'],
                                  method    =   "exact",
                                  name      =   'synapse')

snn_obj['synapse'].connect(i=0, j=1)
snn_obj['synapse'].taupre       = synapse_param['taupre']
snn_obj['synapse'].taupost      = synapse_param['taupost']
snn_obj['synapse'].ApreMax      = synapse_param['ApreMax']
snn_obj['synapse'].ApostMax     = synapse_param['ApostMax']
snn_obj['synapse'].wmax         = synapse_param['wmax']
snn_obj['synapse'].w            = synapse_param['winit']
snn_obj['synapse'].lpre         = 0

snn_obj['st_mon_syn_neurons']   = br.StateMonitor(source        =   snn_obj['synapse'],
                                                  variables     =   ['w', 'Apre', 'Apost'],
                                                  record        =   True)

# %% Network Wrapping

def Real2Fp34_EveryTimestep():
    
    # Using Real2Fp34 Function
    # snn_obj['Fp34_synapse'].Real2Fp34_qApre             = Real2Fp34(snn_obj['Fp34_synapse'].Apre)
    # snn_obj['Fp34_synapse'].Real2Fp34_qApost             = Real2Fp34(snn_obj['Fp34_synapse'].Apost)
    snn_obj['Fp34_synapse'].Real2Fp34_qApre             = Real2Fp34(snn_obj['Fp34_synapse'].Apre*np.exp(-(0.5)/(neuron_param['taum']/br.ms)))
    snn_obj['Fp34_synapse'].Real2Fp34_qApost            = Real2Fp34(snn_obj['Fp34_synapse'].Apost*np.exp(-(0.5)/(neuron_param['taum']/br.ms)))
    
    # Using getNearestQValue Function
    # snn_obj['Fp34_synapse'].getNearest_qApre            = getNearestQValue(snn_obj['Fp34_synapse'].Apre)
    # snn_obj['Fp34_synapse'].getNearest_qApost           = getNearestQValue(snn_obj['Fp34_synapse'].Apost)
    snn_obj['Fp34_synapse'].getNearest_qApre            = getNearestQValue(snn_obj['Fp34_synapse'].Apre*np.exp(-(0.5)/(neuron_param['taum']/br.ms)))
    snn_obj['Fp34_synapse'].getNearest_qApost           = getNearestQValue(snn_obj['Fp34_synapse'].Apost*np.exp(-(0.5)/(neuron_param['taum']/br.ms)))
    
    if (snn_obj['Fp34_synapse'].is_onpre == 1):
        
        # Using Real2Fp34 Function
        # snn_obj['Fp34_synapse'].Real2Fp34_w             =   Real2Fp34(snn_obj['Fp34_synapse'].Real2Fp34_w + snn_obj['Fp34_synapse'].Real2Fp34_lpre)
        snn_obj['Fp34_synapse'].Real2Fp34_w             =   Real2Fp34(snn_obj['Fp34_synapse'].Real2Fp34_w + snn_obj['Fp34_synapse'].Real2Fp34_lpre)
        snn_obj['Fp34_synapse'].Real2Fp34_w             =   Real2Fp34(snn_obj['Fp34_synapse'].Real2Fp34_w - snn_obj['Fp34_synapse'].Real2Fp34_qApost)
        snn_obj['Fp34_synapse'].Real2Fp34_w             =   np.clip(snn_obj['Fp34_synapse'].Real2Fp34_w, 0, snn_obj['Fp34_synapse'].wmax)
        snn_obj['Fp34_synapse'].Real2Fp34_lpre          =   0
        
        # Using getNearestQValue Function
        # snn_obj['Fp34_synapse'].getNearest_w            =   getNearestQValue(snn_obj['Fp34_synapse'].getNearest_w + snn_obj['Fp34_synapse'].getNearest_lpre)
        snn_obj['Fp34_synapse'].getNearest_w            =   getNearestQValue(snn_obj['Fp34_synapse'].getNearest_w + snn_obj['Fp34_synapse'].getNearest_lpre)
        snn_obj['Fp34_synapse'].getNearest_w            =   getNearestQValue(snn_obj['Fp34_synapse'].getNearest_w - snn_obj['Fp34_synapse'].getNearest_qApost)
        snn_obj['Fp34_synapse'].getNearest_w            =   np.clip(snn_obj['Fp34_synapse'].getNearest_w, 0, snn_obj['Fp34_synapse'].wmax)
        snn_obj['Fp34_synapse'].getNearest_lpre         =   0
        
        # Trace value update
        snn_obj['Fp34_synapse'].Apre                    +=  snn_obj['Fp34_synapse'].ApreMax
        snn_obj['Fp34_synapse'].is_onpre                =   0
    
    elif (snn_obj['Fp34_synapse'].is_onpost == 1):
        
        # Using Real2Fp34 Function
        snn_obj['Fp34_synapse'].Real2Fp34_lpre          =   snn_obj['Fp34_synapse'].Real2Fp34_qApre
        
        # Using getNearestQValue Function
        snn_obj['Fp34_synapse'].getNearest_lpre         =   snn_obj['Fp34_synapse'].getNearest_qApre

        # Trace value update
        snn_obj['Fp34_synapse'].Apost                   +=  snn_obj['Fp34_synapse'].ApostMax
        snn_obj['Fp34_synapse'].is_onpost               =   0
        
snn_obj_network = br.Network()
snn_obj_network_op = br.NetworkOperation(Real2Fp34_EveryTimestep, dt=0.5*br.ms)

for key in snn_obj.keys():
    snn_obj_network.add(snn_obj[key])
    
snn_obj_network.add(snn_obj_network_op)

# %% Simulation Begin

snn_obj_network.run(np.max(input_spike_time)+10*br.ms, report='stdout')

# %% Variable Declaration for Spike Monitoring

input_spk_time          = snn_obj['spk_mon_background'].t[:]
input_spk_idx           = snn_obj['spk_mon_background'].i[:]

real_neu_Vm_t           = snn_obj['st_mon_neurons'].t[:]
real_neu_Vm_v           = snn_obj['st_mon_neurons'].v[:]

Fp34_neu_Vm_t           = snn_obj['st_mon_Fp34_neurons'].t[:]
Fp34_neu_Vm_v           = snn_obj['st_mon_Fp34_neurons'].v[:]

Fp34_neu_spk_t          = snn_obj['spk_mon_Fp34_neurons'].t[:]
Fp34_neu_spk_i          = snn_obj['spk_mon_Fp34_neurons'].i[:]

real_syn_state_t        = snn_obj['st_mon_syn_neurons'].t[:]
real_syn_state_w        = snn_obj['st_mon_syn_neurons'].w[:]

Fp34_syn_state_t        = snn_obj['st_mon_Fp34_syn_neurons'].t[:]

real_syn_state_Apre     = snn_obj['st_mon_syn_neurons'].Apre[:]
real_syn_state_Apost    = snn_obj['st_mon_syn_neurons'].Apost[:]

Fp34_syn_state_Real2Fp34_lpre     = snn_obj['st_mon_Fp34_syn_neurons'].Real2Fp34_lpre[:][0]
Fp34_syn_state_Real2Fp34_qApre    = snn_obj['st_mon_Fp34_syn_neurons'].Real2Fp34_qApre[:][0]
Fp34_syn_state_Real2Fp34_qApost   = snn_obj['st_mon_Fp34_syn_neurons'].Real2Fp34_qApost[:][0]
Fp34_syn_state_Real2Fp34_w        = snn_obj['st_mon_Fp34_syn_neurons'].Real2Fp34_w[:]

Fp34_syn_state_getNearest_lpre    = snn_obj['st_mon_Fp34_syn_neurons'].getNearest_lpre[:][0]
Fp34_syn_state_getNearest_qApre   = snn_obj['st_mon_Fp34_syn_neurons'].getNearest_qApre[:][0]
Fp34_syn_state_getNearest_qApost  = snn_obj['st_mon_Fp34_syn_neurons'].getNearest_qApost[:][0]
Fp34_syn_state_getNearest_w       = snn_obj['st_mon_Fp34_syn_neurons'].getNearest_w[:]

# %% Figure Plotting Configuration (Focusing Quantized Values)

idx_post    = np.where(parameters['idxArr']==1)[0]
idx_pre     = np.where(parameters['idxArr']==0)[0]

time_post   = parameters['timeArrBr'][idx_post]/br.ms
time_pre    = parameters['timeArrBr'][idx_pre]/br.ms

Real2Fp34_qApre_post  = np.zeros(len(time_post))
Real2Fp34_qApost_pre  = np.zeros(len(time_pre))

getNearest_qApre_post  = np.zeros(len(time_post))
getNearest_qApost_pre  = np.zeros(len(time_pre))

Fp34_syn_state_t /= br.ms
for i in range(len(time_post)):
    idx = np.where(Fp34_syn_state_t == (time_post[i]+0.5))[0].item()
    Real2Fp34_qApre_post[i]     = Fp34_syn_state_Real2Fp34_qApre[idx]
    getNearest_qApre_post[i]    = Fp34_syn_state_getNearest_qApre[idx]

for i in range(len(time_pre)):
    idx = np.where(Fp34_syn_state_t == (time_pre[i]+0.5))[0].item()
    Real2Fp34_qApost_pre[i] = Fp34_syn_state_Real2Fp34_qApost[idx]
    getNearest_qApost_pre[i] = Fp34_syn_state_getNearest_qApost[idx]

input_spike_time /= br.ms

print('='*40)
print('Input Spikes')
print('-'*40)
print(f"Index \t\t:\t {parameters['idxArr']}\ntime \t\t:\t {parameters['timeArr']}\n")
print('Output Spikes')
print('-'*40)
print(f'Neu+\nIndex \t\t:\t {spike}\nTime \t\t:\t {time}\n')
print(f'Brian2\nIndex \t\t:\t {Fp34_neu_spk_i}\nTime \t\t:\t {Fp34_neu_spk_t}\n')
print('Weights')
print('-'*40)
print(f'Neu+ \t\t:\t {synDecompiledList[2][:].item()}')
print(f'Brian2 \t\t:\t {Fp34_syn_state_Real2Fp34_w[0][-2]} (Fp34)')
print(f'Brian2 \t\t:\t {Fp34_syn_state_getNearest_w[0][-2]} (Nearest)\n')
print('Parameters')
print('-'*40)
print(f"ApreMax \t:\t {parameters['ApreMax']}")
print(f"ApostMax \t:\t {synapse_param['ApostMax']}")
print(f"causalRate \t:\t {parameters['causalRate']}")
print(f"taum \t\t:\t {parameters['taum']}")
print(f"tref \t\t:\t {parameters['tref']}")
print(f"vth \t\t:\t {parameters['vth']}")
print(f"wmax \t\t:\t {parameters['wmax']}")
print(f"winit \t\t:\t {parameters['winit']}")
print('-'*40)
print(f'LTD (post at pre) \t\t:\t {Real2Fp34_qApost_pre} (Fp34)')
print(f'LTP (pre at post) \t\t:\t {Real2Fp34_qApre_post} (Fp34)')
print('-'*40)
print(f'LTD (post at pre) \t\t:\t {getNearest_qApost_pre} (Nearest)')
print(f'LTP (pre at post) \t\t:\t {getNearest_qApre_post} (Nearest)')
print('='*40)
# %% Code Endline