******
Layers
******

This module contains the descriptions of the layers to form a :ref:`Neural Network <module-neuralnetwork>`. 

We support

+ Intel's `oneDNN <https://github.com/oneapi-src/oneDNN>`_ library.
+ Nvidia's `cuDNN <https://developer.nvidia.com/cudnn>`_ library.

Each layer consists of a vector :math:`\mathbf{z}^l\in\mathbb{R}^{n_l}` with *Node Count* elements. Each layer is connected to the previous layer by *Weights*, a *Bias* and an activation function *Activation Function*.

