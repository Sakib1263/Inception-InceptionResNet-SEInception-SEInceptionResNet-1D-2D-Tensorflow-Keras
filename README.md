# Inception-Model-Builder-Tensorflow-Keras
Inception Models supported: Inception_v3, Inception_v4, Inception_ResNet_v1, Inception_ResNet_v2  
## Inception-v3
Inception_v3 is a more efficient version of Inception_v1 (or GoogLeNet) and Inception_v2 (BN-Inception), so they were not implemented.  
Inception_v3 architecture is as follows:  
![Inception_v3 Architecture Params](https://github.com/Sakib1263/Inception-Model-Builder-Tensorflow-Keras/blob/main/Documents/Images/Inception_v3.png "Inception_v3 Architecture")  
Inception-v3 Architecture (Batch Norm and ReLU are used after Conv)  


After the Stem blocks, the Inception_v3 architecture contains 5x Inception-A Modules followed by a Grid Size Reduction Block, then 4x Inception-B Modules followed by another Reduction Block. In the end, before the final MLP layer, there are 2x Inception-C Modules. Each type of Incception Module has been depicted in the image below for Inception_v4. The MLP layer contains Global Pooling, Flattening and final Dense layer selective for Classification or Regression tasks.  
## Inception-v4  
The primary differences between Inception v3 and v4 are in the Stem structure and the number of Inception (A,B and C) modules in each step. The Inception_v4 architecture along with the three modules types are as follows:  
![Inception_v4 Architecture Params](https://github.com/Sakib1263/Inception-Model-Builder-Tensorflow-Keras/blob/main/Documents/Images/Inception_v4.png "Inception_v4 Architecture")  Inception-v4: Whole Network Schema (Leftmost), Stem (2nd Left), Inception-A (Middle), Inception-B (2nd Right), Inception-C (Rightmost)



