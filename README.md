# DomEye

## Code
The Code folder contains the source code of DomEye detector and its counterparts.

Execution order：features.py ─> train.py ─> test_tpr.py ─> test_fpr.py ─> test_overhead.py

The features.py file is used to calculate the statistical features used by each detector.

The train.py file is used to train each detector and test their overall performance.

The test_tpr.py file is used to test the sensitivity performance of each detector.

The test_fpr.py file is used to test the false alarm performance of each detector.

The test_overhead.py file is used to test the performance overhead of each detector.

:warning: The code is intended for RESEARCH PURPOSES ONLY!


## Dataset
The Dataset folder contains all the traffic data used in the experiments, for training and testing DomEye detector and its peers.

The experimental section consists of four different experiments, each corresponding to a dataset. That is to say, we used four different datasets.

The dataset used for detector training is TRAIN.  
File format: pcap  
Link: https://pan.baidu.com/s/11ihJGKaC32JMUNQQE48ZIw  
Code: ifxq

The dataset used for sensitivity testing is TEST_TPR.  
File format: pcap  
Link: https://pan.baidu.com/s/1kiQXG8BG95pxn1M3HBll9Q  
Code: gb9h

The dataset used for false alarm testing is TEST_FPR.  
File format: pcap  
Link: https://pan.baidu.com/s/136FTeLBE11ovBIrjeZeX7w  
Code: 0li0  
Link: https://pan.baidu.com/s/1AZKTpqhdLxvC_-PBvBuJrw  
Code: qou2

The dataset used for overhead testing is TEST_OVERHEAD.  
File format: pcap  
Link: https://pan.baidu.com/s/1t_AC36xlVTCL8v5qEK4siA  
Code: eoq2