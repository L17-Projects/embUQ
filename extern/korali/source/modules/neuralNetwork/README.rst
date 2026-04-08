***************
Neural Network
***************

This module combines the given :ref:`layers <module-neuralnetwork-layer>` to a Neural Network. The Neural Network can be used, for example, as a function approximator in a :ref:`Supervised Learning <module-problem-supervisedlearning>` problem. 

The user can choose between different backends to perform the computations. Besides the Korali lightweight implementation, we support

+ Intel's `oneDNN <https://github.com/oneapi-src/oneDNN>`_ library.
+ Nvidia's `cuDNN <https://developer.nvidia.com/cudnn>`_ library.

This module enables two *Operation* types:

 + **Training**: In this operation, the weights and biases are optimized to minimize a given loss function. The initial guess is chosen according to the specified *Weight Initialization*. 
 
 + **Inference**: In this operation, weights and biases are either provided by the user, or obtained by the training operation. This configuration is only used to compute the output for a given input.
 
