********************
Deconvolution Layer
********************

Specialization of the Layer for Deconvolution in Convolutional Neural Networks. Should contain the same arguments for kernel/padding/stride as the respective Convolution layer that is inverted. The resulting image-size is given as :math:`IH\times IW`. The number of 
_outputChannels=:math:`IH\cdot IW\cdot IC` must be specified such that the number of input channels :math:`IC` takes the wished value.