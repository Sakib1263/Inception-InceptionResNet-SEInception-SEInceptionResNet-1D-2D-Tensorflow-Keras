# Inception-Model-Builder-Tensorflow-Keras
Inception Models supported: 
1. Inception_v3 [Reference: https://arxiv.org/abs/1512.00567]  
2. Inception_v4 [Reference: https://arxiv.org/abs/1602.07261]  
3. Inception_ResNet_v1 [Reference: https://arxiv.org/abs/1602.07261]  
4. Inception_ResNet_v2 [Reference: https://arxiv.org/abs/1602.07261]  

## Inception-v3
Inception_v3 is a more efficient version of Inception_v1 (or GoogLeNet) and Inception_v2 (BN-Inception), so they were not implemented.  
Inception_v3 architecture is as follows:  
![Inception_v3 Architecture Params](https://github.com/Sakib1263/Inception-Model-Builder-Tensorflow-Keras/blob/main/Documents/Images/Inception_v3.png "Inception_v3 Architecture")  
Inception-v3 Architecture (Batch Norm and ReLU are used after Conv) [1]


After the Stem blocks, the Inception_v3 architecture contains 5x Inception-A Modules followed by a Grid Size Reduction Block, then 4x Inception-B Modules followed by another Reduction Block. In the end, before the final MLP layer, there are 2x Inception-C Modules. Each type of Incception Module has been depicted in the image below for Inception_v4. The MLP layer contains Global Pooling, Flattening and final Dense layer selective for Classification or Regression tasks.  
## Inception-v4  
The primary differences between Inception v3 and v4 are in the Stem structure and the number of Inception (A, B and C) modules in each step. The Inception_v4 architecture along with the three modules types are as follows:  
![Inception_v4 Architecture Params](https://github.com/Sakib1263/Inception-Model-Builder-Tensorflow-Keras/blob/main/Documents/Images/Inception_v4.png "Inception_v4 Architecture")  Inception-v4: Whole Network Schema (Leftmost), Stem (2nd Left), Inception-A (Middle), Inception-B (2nd Right), Inception-C (Rightmost) [2]

So, in Inception_v4, Inception Module-A is being used 4 times, Module-B 7 times and Module-C 3 times. As seen in the figure, the modified stem now has branch filter concatenation similar to the Inception Modules.  

## Inception-ResNet-v1  
Inception_ResNet_v1, as shown in the figure below, consists of modfied Inception Modules. The main difference is the skip connections like that of ResNets. Its Stem is similar to Inception_v3. In case of Inception_ResNet_v1, Module-A gets repeated 5 times, Module-B 10 times and Module-C 5 times.  
![Inception_ResNet_v1 Architecture Params](https://github.com/Sakib1263/Inception-Model-Builder-Tensorflow-Keras/blob/main/Documents/Images/Inception_ResNet_v1.png "Inception_ResNet_v1  Architecture")  Inception-ResNet-v1: Whole Network Schema (Leftmost), Stem (2nd Left), Inception-A (Middle), Inception-B (2nd Right), Inception-C (Rightmost) [2]  

## Inception-ResNet-v2  
The Stem of the Inception_ResNet_v2 is exactly same as that of Inception_v4, so it is not mentioned again in the figure below. Its Inception Modules are similar in structure that of Inception_ResNet_v1 but heavier. But in this case, Module-A gets repeated 10 times, Module-B 20 times and Module-C 10 times.  

![Inception_ResNet_v2 Architecture Params](https://github.com/Sakib1263/Inception-Model-Builder-Tensorflow-Keras/blob/main/Documents/Images/Inception_ResNet_v2.png "Inception_ResNet_v2  Architecture") Inception-ResNet-v2: Inception-A (Leftmost), Inception-B (Middle), Inception-C (Rightmost) [2]  

## Supported Features
The speciality about this model is its flexibility. The user has the option for: 
1. Choosing any of 4 available Inception models for either 1D or 2D tasks.
2. Varying number of input kernel/filter, commonly known as the Width of the model. Default is 32, which is better to use following the paper.
3. Varying number of classes for Classification tasks and number of extracted features for Regression tasks.
4. Varying number of Channels in the Input Dataset.
5. Auxilliary Outputs: Two optional Auxilliary Outputs are available for each Inception Model, each one is set before the Grid Size Reduction Blocks, respectively. If set 'TRUE', the output will contain 3 columns as predictions. Mentionable that this concept of having two auxilliary outputs, which act similar to 'Deep Supervision' for segmentation models such as UNet, has been adopted from Inception_v1 (GoogLeNet) architecture as provided below.  

![Inception_v1 Architecture Params](https://github.com/Sakib1263/Inception-Model-Builder-Tensorflow-Keras/blob/main/Documents/Images/Inception_v1.png "Inception_v1 Architecture") GoogLeNet Network (From Left to Right) [3]  

Details of the implementation process are available in the Jupyter Notebook containing the DEMO provided in the codes section. The datasets used in the DEMO are also available in the 'Documents' folder.  


## References
[1] Review: Inception-v3 — 1st Runner Up (Image Classification) in ILSVRC 2015. Medium. (2021). Retrieved 22 August 2021, from https://sh-tsang.medium.com/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c.  
[2] Review: Inception-v4 — Evolved From GoogLeNet, Merged with ResNet Idea (Image Classification). Medium. (2021). Retrieved 22 August 2021, from https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc.  
[3] Review: GoogLeNet (Inception v1)— Winner of ILSVRC 2014 (Image Classification). Medium. (2021). Retrieved 22 August 2021, from https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7.  
