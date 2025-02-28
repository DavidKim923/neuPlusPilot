
"""

Neu+ training / clustering / inferencing applied by PSTDP Learning rule

@author: Lee JP

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

class NeuPlus_2Layer_SNN():
    
    def expSetup(self, experiment_time=100):
        self.testSet.setExperimentConf("EVENT_SAVE_MODE",            True)
        self.testSet.setExperimentConf("EXP_TIME",                   experiment_time)
        self.testSet.setExperimentConf("TIME_ACCEL_MODE",            True)
        self.testSet.setExperimentConf("INTERNAL_ROUTING_MODE",      True)
        self.testSet.setExperimentConf("INPUT_SPIKES_FILE",          self.expFileName['inputSpikeFileName'])
        self.testSet.setExperimentConf("SYN_TABLE_FILE_PREFIX",      self.expFileName['synTableFilePrefix'])
        self.testSet.setExperimentConf("SYN_TABLE_READ_FILE",        self.expFileName['synfname'])
        self.testSet.setExperimentConf("SYN_TABLE_READ_START",       0)
        self.testSet.setExperimentConf("SYN_TABLE_READ_COUNT",       1024)
        
    def fpgaSetup(self, inputView=False, hiddenView=False):
        self.testSet.setFPGAConfiguration("CHIP_CLOCK", 150)
        
        self.neu_input_param['viewable']    =   inputView
        self.neu_hidden_param['viewable']   =   hiddenView

        if (inputView == False):
            self.testSet.setFPGAConfiguration("ARRAY_OUTPUT_OFF", self.neu_input_param['coreNum'])
        if (hiddenView == False):
            self.testSet.setFPGAConfiguration("ARRAY_OUTPUT_OFF", self.neu_hidden_param['coreNum'])

    def learningSetup(self,
                      numCore                   = 3,

                      tauLC                     = 20.,
                      
                      wInp                      = 110,
                      wExc                      = 248.,
                      wInh                      = 16.,
                      wOut                      = 248.,
                      
                      ApreMax                   = 2.25,
                      causalRate                = -2.,
                      
                      modulation                = 0,
                      stochastic                = 0,
                      
                      trainable                 = False
                      ):
        
        numInput                                = self.neu_input_param['number']
        numHidden                               = self.neu_hidden_param['number']
        numOutput                               = self.neu_output_param['number']

        self.numOfTotalNeuron                   = int(numCore*1024)
        self.learningParam['tauLC']             = tauLC
        self.learningParam['wExc']              = wExc
        self.learningParam['wInp']              = wInp
        self.learningParam['winit']             = np.round(np.random.rand(numInput*numHidden) * wExc, decimals=2)
        self.learningParam['winit']             = self.learningParam['winit'].reshape(numInput, numHidden)
        self.learningParam['wInh']              = wInh
        self.learningParam['wOut']              = wOut
        self.learningParam['ApreMax']           = ApreMax*1.0
        self.learningParam['causalRate']        = causalRate*1.0
        self.learningParam['modulation']        = modulation
        self.learningParam['stochastic']        = stochastic
        self.learningParam['trainable']         = trainable
        
        for idx in range(numCore):
            self.testSet.setLearningEngine("LC_TIMECONSTANT",      idx, self.learningParam['tauLC'])
            self.testSet.setLearningEngine("MAX_SYN_WEIGHT",       idx, self.learningParam['wExc'])
            self.testSet.setLearningEngine("TRACE_UPDATE_AMOUNT",  idx, idx,                                self.learningParam['ApreMax'])
            self.testSet.setLearningEngine("POST_TRACE_SCALE",     idx, idx,                                self.learningParam['causalRate'])
            self.testSet.setLearningEngine("LEARNING_RULE",        idx, self.learningParam['modulation'],   0,                                0, self.learningParam['stochastic'], 0)
    
    def generateInputSpike(self):
        self.inputSpike.events_to_bytefile(events            =   self.events, 
                                           byte_file_name    =   self.expFileName['inputSpikeFileName'], 
                                           conf_events       =   [],
                                           weight_single     =   self.learningParam['wInp'])
    
    def loadH5(self, directory="None", fileName="None"):
        
        if os.path.isfile(directory+fileName) :
            with h5py.File(directory+fileName, 'r') as h5f:
                spike_time  = h5f['spike_time'][:]
                spike_idx   = h5f['spike_idx'][:]
                mnist_label = h5f['mnist_label'][:]
        
            print('----------------------------------------------------------------------------------------------')
            print(f'{len(spike_time)} Spike data loaded from {fileName}.')
            print('----------------------------------------------------------------------------------------------\n')
        
            return spike_time, spike_idx, mnist_label
    
    def inputSpikeSetup(self,
                        deltat          = 100,
                        start           = 0,
                        duration        = 1
                        ):
        
        self.duration       = duration        
        spikeLoad           = self.loadH5(directory     =   self.directory['inputSpike'], fileName      =   self.directory['inputSpikeFileName'])
        
        loaded_spike_time   = spikeLoad[0]
        loaded_spike_idx    = spikeLoad[1]
        self.loadedLabel    = spikeLoad[2][start:start+duration]
                        
        startIdx            = np.where(loaded_spike_time[:] >= start) [0][0]
        endIdx              = np.where(loaded_spike_time[:] <= (start + duration)) [0][-1]
        
        input_spike_time    = (loaded_spike_time[startIdx:endIdx] - start)
        input_spike_idx     = loaded_spike_idx[startIdx:endIdx]
        
        print(' ---------------------------------------------------------------------------------------------')
        print(f'|   input spike time    : {input_spike_time[0]:5.2f} ~ {input_spike_time[-1]:5.2f} sec')
        print(f'|   Time in Spike Train : {start+input_spike_time[0]:5.2f} ~ {start+input_spike_time[-1]:5.2f} sec')
        print(f'|   # of input spike    : {len(input_spike_idx):5d}                            ')
        print(f'|   # of Labels         : {len(self.loadedLabel):5d}                          ')
        print(' ---------------------------------------------------------------------------------------------\n')

        self.inputSpike     = inSpike.MakeNeuPLUSByteFile(deltat=deltat)
        self.events         = [input_spike_time*10, input_spike_idx]
                
    def neuronCoreSetup(self, vth=100, taum=20, vreset=0, tr=5):

        neuronNumberPerCore                  = 1024
        self.neu_input_param['size']         = int(np.sqrt(self.neu_input_param['number']))
        self.neu_input_param['vth']          = 100
        self.neu_input_param['coreNum']      = 0
        self.neu_input_param['taum']         = 0.1
        self.neu_input_param['vreset']       = 0
        self.neu_input_param['tr']           = 0
        self.neu_input_param['firstIdx']     = self.neu_input_param['coreNum'] * neuronNumberPerCore
    
        self.neu_hidden_param['size']        = int(np.sqrt(self.neu_hidden_param['number']))
        self.neu_hidden_param['vth']         = vth
        self.neu_hidden_param['coreNum']     = 1
        self.neu_hidden_param['taum']        = taum
        self.neu_hidden_param['vreset']      = vreset
        self.neu_hidden_param['tr']          = tr
        self.neu_hidden_param['firstIdx']    = self.neu_hidden_param['coreNum'] * neuronNumberPerCore
    
        self.neu_output_param['vth']         = vth
        self.neu_output_param['coreNum']     = 1
        self.neu_output_param['taum']        = taum
        self.neu_output_param['vreset']      = 0
        self.neu_output_param['tr']          = tr
        self.neu_output_param['firstIdx']    = self.neu_hidden_param['firstIdx'] + self.neu_hidden_param['number']
        
        self.testSet.setNeuronCoreConf([self.neu_input_param['coreNum']],       # Core Number
                                       [self.neu_input_param['taum']],          # Taum
                                       [self.neu_input_param['tr']],            # Refractory
                                       [self.neu_input_param['vth']],           # Vth
                                       [0],                                     # Synaptic Gain
                                       [0])                                     # Stochasticity
        
        self.testSet.setNeuronCoreConf([self.neu_hidden_param['coreNum']],
                                       [self.neu_hidden_param['taum']],
                                       [self.neu_hidden_param['tr']],
                                       [self.neu_hidden_param['vth']],
                                       [0],
                                       [0])
    
    def generateSynapticTable(self):
        
        numInput            =   self.neu_input_param ['number']
        numHidden           =   self.neu_hidden_param['number']
        numOutput           =   self.neu_output_param['number']

        layer_input         =   synTable.layer(self. neu_input_param['firstIdx'], numInput)
        layer_hidden        =   synTable.layer(self.neu_hidden_param['firstIdx'], numHidden)
        layer_output        =   synTable.layer(self.neu_output_param['firstIdx'], numOutput)

        print(f'Input  Layer  : {layer_input[0]:5d} ~ {layer_input[-1]:5d} (Reserved in Core # 0)')
        print(f'Hidden Layer  : {layer_hidden[0]:5d} ~ {layer_hidden[-1]:5d} (Reserved in Core # 1)')
        print(f'Output Layer  : {layer_output[0]:5d} ~ {layer_output[-1]:5d} (Reserved in Core # 1)')
        print("-"*50)

        # Weight Loading
        appliedWeight = []
        if (len(self.loadedWeight)):
            appliedWeight   =   self.loadedWeight
        else:
            appliedWeight   =   self.learningParam['winit']

        # Weight normalization
        connectionCount =   np.zeros(numOutput)
        normedWeight    =   np.zeros(shape=(numHidden, numOutput))
        
        if (len(self.loadedAssignment)):
            loadedAssignment                    =   self.loadedAssignment
            wmax_hidden_output                  =   self.learningParam['wOut']
            outputNeuronIdx, numConnection      =   np.unique(loadedAssignment, return_counts=True)
            connectionCount[outputNeuronIdx]    =   numConnection
            
            for source_idx in range(numHidden):
                normedWeight[source_idx, loadedAssignment[source_idx]] = wmax_hidden_output * (1./connectionCount[loadedAssignment[source_idx]])

        # Self Inhibition on hidden layer   
        neuronCoreOffset    =   1024
        
        sourceArr           =   np.arange(0, numHidden) + neuronCoreOffset
        targetArr           =   sourceArr

        sourceArrTotal      =   np.tile(sourceArr, (numHidden-1, 1)).T.flatten()
        targetArrTotal      =   np.array([])

        for i in range(numHidden):
            targetArrTotal = np.append(targetArrTotal, np.delete(targetArr, i))

        sourceArrTotal = sourceArrTotal.astype(int)
        targetArrTotal = targetArrTotal.astype(int)

        SynTable = synTable.synapticTable(4,
                                          1024,
                                          1024)
        
        # Input - Hidden Synapse
        SynTable.createFromWeightMatrix(
                                            source          =   layer_input,
                                            destination     =   layer_hidden,
                                            weights         =   appliedWeight,
                                            trainable       =   self.learningParam['trainable']
                                        )
        
        # Hidden - Hidden Synapse
        SynTable.createConnections      (   
                                            source          =   sourceArrTotal,
                                            destination     =   targetArrTotal,
                                            mode            =   'one',
                                            probability     =   1,
                                            weight          =   self.learningParam['wInh'],
                                            wmax            =   self.learningParam['wExc'],
                                            delay           =   0,
                                            dmax            =   0,
                                            synType         =   'inh',
                                            trainable       =   False
                                        )
        
        # Hidden - Output Synapse
        # SynTable.createFromWeightMatrix (
        #                                     source          =   layer_hidden,
        #                                     destination     =   layer_output,
        #                                     weights         =   normedWeight,
        #                                     trainable       =   False
        #                                 )

        SynMapCompiled = compiler.synMapCompiler(self.expFileName['synTableFilePrefix'])
        
        SynMapCompiled.generateSynMap(synTable=SynTable, 
                                      inputNeurons=[range(self.numOfTotalNeuron)]
                                      )
        
    def generateExpConfig(self):
        self.testSet.genExpConfFile(self.expFileName['conffname'])
        
    def NeuPlusRun(self, firstRun=True):
        self.generateExpConfig()
        self.generateInputSpike()
        self.generateSynapticTable()
        
        if firstRun:
            _ = param.NeuPLUSRun('-GUIModeOff', '-conf', self.expFileName['conffname'])

    #############################################
    #                                           #
    #       Analyze Spike and Weights           #

    def analyzeSpike(self,
                     printAvgSpike           =   False,
                     showAvgSpike            =   False,
                     showResultAvgSpike      =   False,

                     printAssignment         =   False,
                     showResultAssignment    =   False,
                     saveAssignment          =   False,

                     printAccuracy           =   False,
                     showResultAccuracy      =   False
                     ):
        
        event_results           = eventAnalysis.eventAnalysis(param.NeuPLUSResultFolder + '2023' + date.today().strftime("%m%d") + "/")
        
        spike                   = event_results.spikes[1]
        time                    = event_results.spikes[0]
        spikeSlice              = np.array([])
        
        firingCount             = np.zeros(self.numOfTotalNeuron)

        inputAvgFiringRateList  =   []
        hiddenAvgFiringRateList =   []
        outputAvgFiringRateList =   []

        numInput                =   self.neu_input_param['number']
        numHidden               =   self.neu_hidden_param['number']
        numOutput               =   self.neu_output_param['number']

        sizeInput               =   self.neu_input_param['size']
        sizeHidden              =   self.neu_hidden_param['size']
        
        input_layer             =   np.array([self.neu_input_param['firstIdx']    ,  self.neu_input_param['firstIdx']    + numInput]) 
        hidden_layer            =   np.array([self.neu_hidden_param['firstIdx']   ,  self.neu_hidden_param['firstIdx']   + numHidden]) 
        output_layer            =   np.array([self.neu_output_param['firstIdx']   ,  self.neu_output_param['firstIdx']   + numOutput]) 

        # Variable for clustering
        assignment              =   np.zeros(numHidden)

        resultMonitor           =   np.zeros(shape=(numHidden, numOutput))
        rateMonitor             =   np.zeros(shape=(numHidden, numOutput))

        labelCount              =   np.zeros(numOutput, dtype=int)
        connectionCount         =   np.zeros(numOutput)

        # Variable for inferencing
        accuracy                                    =   np.zeros(self.duration)
        matchCount                                  =   np.zeros_like(accuracy)

        labelList                                   =   []
        predictedList                               =   []
        numPredicted                                =   np.zeros(numOutput)
        labelCount                                  =   np.zeros_like(numPredicted)

        for i in (range(self.duration)):
            # Slice Idx
            start                                   =   np.where(time >= i)     [0][0]       
            end                                     =   np.where(time < (i+1))  [0][-1]
            label                                   =   int(self.loadedLabel[i])
            labelList.append(label)
            labelCount[label]                       +=  1
            spikeSlice                              =   spike[start:end]

            print(f'Image       :   {i}')
            print(f'Start Time  :   {start}')
            print(f'End Time    :   {end}')
            print(f'# spike     :   {len(spikeSlice)}')
            print('-'*50)

            # Count Total Firing
            firingCount                             =   np.zeros(len(firingCount))
            spikeIdx, spikeCnt                      =   np.unique(spikeSlice, return_counts=True)
            spikeIdx                                =   spikeIdx.astype(int)
            firingCount[spikeIdx]                   =   spikeCnt
                        
            # Count Firing for each layer
            if (self.neu_input_param['viewable']==True):
                inputFiring                         =   firingCount[input_layer[0]  : input_layer[1]]
            else:
                inputFiring                         =   np.ones(numInput)
                print("Not Counts Neuron Fire in Input Layer")
                print("-"*50)
            if (self.neu_hidden_param['viewable']==True):
                hiddenFiring                        =   firingCount[hidden_layer[0] : hidden_layer[1]]
                outputFiring                        =   firingCount[output_layer[0] : output_layer[1]]
            else:
                hiddenFiring                        =   np.zeros(numHidden)
                print("Not Counts Neuron Fire in Hidden Layer")
                print("-"*50)
                outputFiring                        =   np.zeros(numOutput)
                print("Not Counts Neuron Fire in Output Layer")
                print("-"*50)

            # Clustering each neurons in hidden layer
            resultMonitor[:, label]                 +=  hiddenFiring
            rateMonitor[:, label]                   =   resultMonitor[:, label] / labelCount[label]

            assignment                              =   np.argmax(rateMonitor, axis=1)
            assignedlabels, numberOfAssignments     =   np.array(np.unique(assignment, return_counts=True), dtype=int)
            connectionCount[assignedlabels]         =   numberOfAssignments

            # Calculate average firing in the layers
            inputAvgFiringRate                      =   np.mean(inputFiring[np.where(inputFiring>0)[0][:]])
            hiddenAvgFiringRate                     =   np.mean(hiddenFiring)
            outputAvgFiringRate                     =   np.mean(outputFiring)
            
            inputAvgFiringRateList.append   (inputAvgFiringRate)
            hiddenAvgFiringRateList.append  (hiddenAvgFiringRate)
            outputAvgFiringRateList.append  (outputAvgFiringRate)

            # Inference with output neurons
            predicted                               =   np.argmax(outputFiring)
            predictedList.append(predicted)
            numPredicted[predicted]                 +=  1

            if (label == predicted):
                matchCount[label] += 1

            accuracy[i] = np.sum(matchCount) / (i+1)
            
            if (showAvgSpike):
                plt.figure(figsize=(16, 8))
                plt.suptitle(f"Index : {i}, Label : {label}")

                plt.subplot(1, 3, 1)
                plt.imshow(inputFiring.reshape(sizeInput, -1), cmap='gray')
                plt.title(f"input, avg : {inputAvgFiringRate:.2f} Hz")
                
                plt.subplot(1, 3, 2)
                plt.imshow(hiddenFiring.reshape(sizeHidden, -1), cmap='gray')
                plt.title(f"Hidden, avg : {hiddenAvgFiringRate:.2f} Hz")
                
                plt.subplot(1, 3, 3)
                plt.imshow(outputFiring, cmap='gray')
                plt.title(f"Output, avg : {outputAvgFiringRate:.2f} Hz")
                plt.show()
            
            if (printAvgSpike):
                print(f"Input  Firing Rate : \n{inputFiring.reshape(sizeInput, -1).astype(int)} Hz\n")
                print(f"Hidden Firing Rate : \n{hiddenFiring.reshape(sizeHidden, -1).astype(int)} Hz\n")
                print(f"Output Firing Rate : \n{outputFiring.astype(int)} Hz\n")
                print('-'*50)

                print(f"Avg Input  Firing Rate : \t{inputAvgFiringRateList[-1]:5.2f} Hz")
                print(f"Avg Hidden Firing Rate : \t{hiddenAvgFiringRateList[-1]:5.2f} Hz")
                print(f"Avg Output Firing Rate : \t{outputAvgFiringRateList[-1]:5.2f} Hz")
                print('-'*50)

            if (printAssignment):
                print(f'resultMonitor (upto 20) : \n{resultMonitor[:20, :]}\n')
                print(f'rateMonitor   (upto 20) : \n{rateMonitor[:20, :]}\n')
                print(f'Assigned labels         : \n{assignedlabels}')
                print(f'Label Count             : \n{labelCount.astype(int)}')
                print(f'connectionCount         : \n{connectionCount.astype(int)}')
                print(f'Assignment Table        : \n{assignment.reshape(int(np.sqrt(numHidden)), -1)}')
                print('-'*50)

            if (printAccuracy):
                print('='*50)
                print(f'\t\tIndex : {i}')
                print('='*50)
                print('\n')
                print(f'Label           :   {label}')
                print(f'Predicted       :   {predicted}')
                print('-'*50)
                print(f'LabelCount      :   {labelCount.astype(int)}\n')
                print(f'numPredicted    :   {numPredicted.astype(int)}')
                print('-'*50)
                print(f'Accuracy        :   {accuracy[-1]}          ({int(sum(matchCount))}/{int(sum(labelCount))})\n')
                
        if (showResultAvgSpike):
            plt.figure(figsize=(8, 8))
            plt.plot(np.arange(0, self.duration), inputAvgFiringRateList, 'g-', label="Input Layer")
            plt.plot(np.arange(0, self.duration), hiddenAvgFiringRateList, 'r--', label="Hidden Layer")
            plt.title(f"vth : {self.neu_hidden_param['vth']}, wmax : {self.learningParam['wExc']}, inh : {self.learningParam['wInh']}, Apre : {self.learningParam['ApreMax']}, Apost : {np.sign(self.learningParam['causalRate'])*self.learningParam['ApreMax']/abs(self.learningParam['causalRate'])}")
            plt.legend(loc='best')
            plt.xticks(np.arange(0, self.duration, 1))
            plt.xlabel("Image Index")
            plt.ylabel("Average Firing Rate in [Hz]")
            learningParam = self.learningParam
            fileName = f"avgSpike_{self.duration}_img_{int(learningParam['wExc'])}_wmax_{int(learningParam['wInh'])}_inh_{int(learningParam['ApreMax'])}_Apre_{int(abs(learningParam['causalRate']))}_cr"
            plt.savefig(self.directory['result']+fileName)
            plt.show()
        
        if (showResultAssignment):
            plt.figure(figsize=(16, 8))

            plt.subplot(1, 2, 1)
            plt.bar(np.arange(0, numOutput) - 0.2, labelCount, width=0.4, color='g', label="Label Counts")
            plt.xticks(ticks=np.arange(0, numOutput))
            plt.yticks(ticks=np.arange(0, sum(labelCount)))
            plt.xlabel("Labels (Output Neuron)")
            plt.ylabel("Counts")
            plt.legend(loc='upper left')

            plt.twinx()
            plt.bar(np.arange(0, numOutput) + 0.2, connectionCount, width=0.4, color='y', label="Connection Counts")
            plt.xticks(ticks=np.arange(0, numOutput))
            plt.yticks(ticks=np.arange(0, numHidden, int(numHidden/10.)))
            plt.legend(loc='upper right')
            
            plt.subplot(1, 2, 2)
            plt.imshow(assignment.reshape(int(np.sqrt(numHidden)), -1))
            plt.xticks(ticks=np.arange(0, self.neu_hidden_param['size']))
            plt.yticks(ticks=np.arange(0, self.neu_hidden_param['size']))
            plt.colorbar(ticks=np.arange(0, 10))
            plt.title("Neural Assignment Table")

            plt.show()

        if (showResultAccuracy):
            xAxisLabel = np.arange(0, numOutput)
            xAxisimageIndex = np.arange(0, self.duration)

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(xAxisLabel, labelCount,        'r-', label='Label Appears')
            plt.plot(xAxisLabel, numPredicted,      'g-', label='Prediction')
            plt.plot(xAxisLabel, connectionCount,   'b-', label='Connection')
            plt.ylabel("Counted Number")
            plt.xlabel("Output Neurons")
            plt.xticks(xAxisLabel)
            plt.legend(loc='best')
            plt.tight_layout()

            plt.subplot(2, 1, 2)
            plt.plot(xAxisimageIndex, accuracy*100, 'k-', label='Accuracy')
            plt.xlabel("Number of Images")
            plt.xticks(xAxisimageIndex)
            plt.ylabel("%")
            plt.yticks(np.arange(0, 100, 10))
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

        if (saveAssignment):
            learningParam = self.learningParam
            
            fileName = f"assignment_{self.duration}_img_{int(learningParam['wExc'])}_wmax_{int(learningParam['wInh'])}_inh_{int(learningParam['ApreMax'])}_Apre_{int(abs(learningParam['causalRate']))}_cr"
            
            with h5py.File(self.directory['assignment']+fileName, 'w') as h5f:
                h5f.create_dataset('assignment', data=assignment.reshape(int(np.sqrt(numHidden)), -1))
            
            print('='*50)
            print(f"Assignment Saved\nFile name : {fileName}")
            print('='*50)

    def analyzeWeight(self,
                      histogram     =   False,
                      image         =   False,
                      weightValue   =   False,
                      selectivity   =   False,
                      saveWeight    =   False
                      ):
        
        synMapDecompile     =   compiler.synMapDecompiler(self.expFileName['synfname'])
        synTable_read       =   synTable.synapticTable()
        synDecompiledList   =   synMapDecompile.decompileSynMap()
        synTable_read       =   synTable_read.createFromSynapseList(synDecompiledList)

        initialWeight           =   self.learningParam['winit'].flatten()
        maximumWeight           =   self.learningParam['wExc']
        self.trainedWeight      =   np.array(synDecompiledList[2][np.where(synDecompiledList[3] == 1)[0][:]])
        trainedWeight           =   np.zeros_like(initialWeight)

        if (len(self.trainedWeight) > 0):
            trainedWeight           =   self.trainedWeight

        weightDiff              =   trainedWeight - initialWeight
        weightDiffSign          =   np.sign(weightDiff)

        numInput                =   self.neu_input_param['number']
        numHidden               =   self.neu_hidden_param['number']
        numOutput               =   self.neu_output_param['number']
        
        if (histogram):
            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.title("Initial Weight")
            plt.hist(initialWeight/maximumWeight, 100, color='green', density=True)
            plt.xlabel(f"weight / wmax ({maximumWeight})")
            plt.ylabel('Population Density')
            plt.tight_layout()
    
            plt.subplot(2, 1, 2)
            plt.title("Trained Weight")
            plt.hist(np.array(trainedWeight)/maximumWeight, 100, density=True)
            plt.xlabel(f"weight / wmax ({maximumWeight})")
            plt.ylabel('Population Density')
            plt.tight_layout()
            plt.show()
        
        if (image):
            plt.figure(figsize=(8, 8))
            plt.subplot(1, 3, 1)
            plt.imshow(trainedWeight.reshape(numInput, numHidden))
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xlabel("Target Index")
            plt.ylabel("Source Index")
            plt.title("Weight Image")
            plt.tight_layout()
            
            plt.subplot(1, 3, 2)
            plt.imshow(weightDiff.reshape(numInput, numHidden))
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xlabel("Target Index")
            plt.ylabel("Source Index")
            plt.title("Weight Increament/Decreament")
            plt.tight_layout()
            
            plt.subplot(1, 3, 3)
            plt.imshow(weightDiffSign.reshape(numInput, numHidden))
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xlabel("Target Index")
            plt.ylabel("Source Index")
            plt.title("Sign of Increament/Decreament")
            plt.tight_layout()
            plt.show()
        
        if (weightValue):
            printWithIdx = np.array([0, 150])
            print('-'*50)
            print(f'Initial Weights From {printWithIdx[0]} ~ {printWithIdx[1]} : \n{initialWeight[:printWithIdx[1]]}')
            print('-'*50)
            print(f'Trained Weights From {printWithIdx[0]} ~ {printWithIdx[1]} : \n{trainedWeight[:printWithIdx[1]]}')
            print('-'*50)
            print(f'Weight Differences From {printWithIdx[0]} ~ {printWithIdx[1]} : \n{weightDiff[:printWithIdx[1]]}')
            print('-'*50)
            print(f'Sign of Differences From {printWithIdx[0]} ~ {printWithIdx[1]} : \n{weightDiffSign[:printWithIdx[1]]}')
            print('-'*50)
        
        if (selectivity):
            sizeHidden                  =   self.neu_hidden_param['size']
            trainedWeight               =   trainedWeight.reshape(sizeHidden, sizeHidden, -1)
            hiddenSlice                 =   np.zeros(shape=(sizeHidden, sizeHidden, 9))

            hiddenSlice[:, :, 0:3]      =   trainedWeight[:, :, 0:3]
            hiddenSlice[:, :, 3:6]      =   trainedWeight[:, :, int(numHidden/2):int(numHidden/2)+3]
            hiddenSlice[:, :, 6:9]      =   trainedWeight[:, :, numHidden-3:numHidden]

            plt.figure(figsize=(12, 12))
            for row in np.arange(1, 4):
                for col in np.arange(1, 4):
                    idx = 3*(row-1)+col
                    plt.subplot(3, 3, idx)
                    plt.imshow(hiddenSlice[:, :, idx-1], cmap='gray')
                    plt.colorbar()
                    plt.xticks(np.arange(0, 29, 10))
                    plt.yticks(np.arange(0, 29, 10))
                    plt.title(f"Hidden idx {idx}")
            plt.suptitle("Row : Front, Middle, Back")
            plt.show()

        if (saveWeight):
            learningParam = self.learningParam
            
            fileName = f"weight_{self.duration}_img_{int(learningParam['wExc'])}_wmax_{int(learningParam['wInh'])}_inh_{int(learningParam['ApreMax'])}_Apre_{int(abs(learningParam['causalRate']))}_cr"
            
            with h5py.File(self.directory['weight']+fileName, 'w') as h5f:
                h5f.create_dataset('weight', data=trainedWeight.reshape(numInput, numHidden))
            
            print('='*50)
            print(f"Weight Saved\nFile name : {fileName}")
            print('='*50)


    #############################################
    #                                           #
    #       Load Weight and Assignments         #

    def loadWeight(self, fileName="None", loadWeight=False, loadLatest=False):
        if (loadWeight):

            if (loadLatest):
                folderPath = self.directory['weight']
                filePathRecentTime = []

                for fileName in os.listdir(folderPath):
                    filePath = folderPath + fileName
                    recentTime = os.path.getmtime(filePath)
                    filePathRecentTime.append((filePath, recentTime))
                
                fileName = max(filePathRecentTime, key=lambda x : x[1])[0]

            with h5py.File(fileName, 'r') as h5f:
                self.loadedWeight = h5f['weight'][:].reshape(self.neu_input_param['number'], self.neu_hidden_param['number'])
                print('-'*50)
                print(f'Weight in {fileName} Loaded')
                print(f'size : {self.loadedWeight.shape}')
                print('-'*50)
        else:
            self.loadedWeight = []

    def loadAssignment(self, fileName="None", loadAssignment=False, loadLatest=False):
        if (loadAssignment):
            
            if (loadLatest):
                folderPath = self.directory['assignment']
                filePathRecentTime = []

                for fileName in os.listdir(folderPath):
                    filePath = folderPath + fileName
                    recentTime = os.path.getmtime(filePath)
                    filePathRecentTime.append((filePath, recentTime))
                
                fileName = max(filePathRecentTime, key=lambda x : x[1])[0]

            with h5py.File(fileName, 'r') as h5f:
                self.loadedAssignment = h5f['assignment'][:].reshape(int(np.sqrt(self.neu_hidden_param['number'])), -1)
                print('-'*50)
                print('Assignment Loaded')
                print(f'size : {self.loadedAssignment}')
                print('-'*50)
        else:
            self.loadedAssignment = []

            
    def __init__(self, numInput=0, numHidden=0, numOutput=0):
        np.set_printoptions(threshold=sys.maxsize, linewidth=500)
        
        self.expFileName                        =   {}
        self.expFileName['inputSpikeFileName']  =   "testByte.nam"
        self.expFileName['synTableFilePrefix']  =   "SynTableWrite"
        self.expFileName['fname']               =   "testExpConf.exp"
        self.expFileName['nfname']              =   "testNeuronConf.nac"
        self.expFileName['conffname']           =   "neuplusconf.txt"
        self.expFileName['synfname']            =   "testRead.dat"
        
        self.directory                          =   {}
        self.directory['main']                  =   "C:/Users/user/Desktop/KIST/Project/synFlow2_project/"
        self.directory['weight']                =   self.directory['main'] + '/weights/'
        self.directory['assignment']            =   self.directory['main'] + '/assignments/'
        self.directory['result']                =   self.directory['main'] + '/result/'
        self.directory['inputSpike']            =   "C:/Users/user/Desktop/KIST/Project/synFlow2_project/"      
        # 'C:/Users/User/Desktop/LJP/PSTDP_2_LAYER_SNN/data/poi_input/'
        self.directory['inputSpikeFileName']    =   'MNIST_POI_0_TO_10'
        
        self.testSet                            =   compiler.expSetupCompiler(self.expFileName['fname'], self.expFileName['nfname'])
        self.inputSpike                         =   []
        
        self.numOfTotalNeuron                   =   0
        self.duration                           =   0
        
        self.learningParam                      =   {}
        
        self.image                              =   []
        self.label                              =   []
        self.events                             =   []
        
        self.neu_input_param                    =   {}
        self.neu_input_param['number']          =   numInput
        self.neu_input_param['viewable']        =   True
        
        self.neu_hidden_param                   =   {}
        self.neu_hidden_param['number']         =   numHidden
        self.neu_hidden_param['viewable']       =   True
        
        self.neu_output_param                   =   {}
        self.neu_output_param['number']         =   numOutput
        self.neu_output_param['viewable']       =   True
        
        self.initialWeight                      =   []
        self.trainedWeight                      =   []
        self.loadedWeight                       =   []
        self.loadedLabel                        =   []
        self.loadedAssignment                   =   []
        
snn = NeuPlus_2Layer_SNN(
                            numInput            =   784,
                            numHidden           =   400,
                            numOutput           =   10
                        )

snn.expSetup            (
                            experiment_time     =   1000
                        )
snn.neuronCoreSetup     (
                            vth                 =   248,
                            taum                =   20,
                            vreset              =   0,
                            tr                  =   5
                        )
snn.fpgaSetup           (
                            inputView           =   False,
                            hiddenView          =   True
                        )
snn.learningSetup       (
                            numCore             =   2,
                            wExc                =   248.,
                            wInh                =   16.,
                            wOut                =   16.,
                            
                            ApreMax             =   32.,
                            causalRate          =   -16.,
                            stochastic          =   1.,
                            
                            trainable           =   True
                        )
snn.inputSpikeSetup     (
                            deltat              =   10,
                            start               =   0,
                            duration            =   2
                        )
snn.loadWeight          (
                            fileName            =   ' ',
                            loadWeight          =   False,
                            loadLatest          =   False
                        )
snn.loadAssignment      (
                            fileName            =   ' ',
                            loadAssignment      =   False,
                            loadLatest          =   False
                        )
snn.NeuPlusRun          (
                            firstRun            =   True
                        )
snn.analyzeSpike        (
                            printAvgSpike       =   False,
                            showAvgSpike        =   False,
                            showResultAvgSpike  =   True,

                            printAssignment     =   False,
                            saveAssignment      =   False,
                            showResultAssignment=   False,

                            printAccuracy       =   False,
                            showResultAccuracy  =   False
                        )
snn.analyzeWeight       (
                            histogram           =   False,
                            image               =   False,
                            weightValue         =   False,
                            selectivity         =   False,
                            saveWeight          =   False
                        )