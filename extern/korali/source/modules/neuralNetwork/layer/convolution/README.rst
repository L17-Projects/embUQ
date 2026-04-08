********************
Convolutional Layer
********************

Specialization of the Layer for Convolution in Convolutional Neural Networks. The input image of size :math:`IH\times IW` with :math:`IC` channels is padded using :math:`PT` zeros from the top, :math:`PL` left, :math:`PB` bottom and :math:`PR` rights, respectively. The trainable convolution has size :math:`KH\times KW` and :math:`OC` channels and performs the operation using a stride of :math:`SV` in vertical and :math:`SH` in horizontal direction respectively.

The dimension :math:`OC\times OH\times OW` of the resulting convoluted image can be computed as

.. math::

	OH = \lfloor \frac{IH - (KH - (PR + PL))}{SH}\rfloor + 1
	OW = \lfloor \frac{IW - (KW - (PT + PB))}{SV}\rfloor + 1
	OC = \frac{\text{_outputChannels}}{OH \cdot OW}

We note that the _outputChannels must be specified such that :math:`OC` takes the wished value.