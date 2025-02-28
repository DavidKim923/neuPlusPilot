# neuPLUS tutorial
Welcome to the neuPLUS neuromorphic hardware testing repository. This repository is a collection of **neuralSim**-based codes designed specifically for evaluating and validating the performance of neuPLUS neuromorphic hardware. It provides a flexible and user-friendly simulation environment that allows researchers and engineers to test hardware functionality.

### neuPlus_rate.py
**neuPlus_rate.py** is a script designed to verify the proper operation of the neuron core in the neuPLUS neuromorphic hardware. The script sets fixed input weights and then examines the relationship between the output firing rate and variations in the input firing rate and neuron threshold.

With a neuron core time constant of 20ms and a refractory period of 5ms, the input weight is fixed at 14, and the input firing rate is varied from 0 to 3000Hz. The graph below shows the output firing rate for neuron core thresholds of 32, 64, 128, and 256.

![Image](https://github.com/user-attachments/assets/d8c6a578-9a29-4285-9118-f4c4bca00d1d)

### neuPlus_weight.py

**neuPlus_rate.py** is a script designed to verify the correct operation of the neuron core in the neuPLUS neuromorphic hardware. The code sets a fixed input firing rate and then examines the relationship between the output firing rate and variations in the input weight and neuron threshold.

With a neuron core time constant of 20ms and a refractory period of 5ms, the input firing rate is fixed at 1500Hz, and the input weight is varied from 1 to 248. The graph below shows the output firing rate for neuron core thresholds of 32, 64, 128, and 256.


![Image](https://github.com/user-attachments/assets/9d5fbb27-378c-4a60-8cb8-9d7d30222ee3)

### neuPlus_weight_rate.py
**neuPlus_weight_rate.py** is a script designed to verify the correct operation of the neuron core in the neuPLUS neuromorphic hardware. The script keeps only the neuron core settings fixed while examining the relationship between the output firing rate and changes in both the input firing rate and weight.

The neuron core is configured with a time constant of 20ms, a refractory period of 5ms, and a threshold of 256. Under conditions where the input firing rate varies from 0 to 3000Hz and the input weight varies from 1 to 248, the output firing rate is visualized using a 2D heatmap, as shown below.

![Image](https://github.com/user-attachments/assets/0bff7d03-b843-4f36-aa73-7222af1ec670)