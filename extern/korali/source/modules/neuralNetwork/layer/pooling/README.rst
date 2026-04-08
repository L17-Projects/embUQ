***************
Pooling Layer
***************

Specialization of the Layer for Pooling in Convolutional Neural Networks. The input image of size :math:`IH\times IW` is padded using :math:`PT` zeros from the top, :math:`PL` left, :math:`PB` bottom and :math:`PR` rights, respectively. The pooling kernel then beforms the selected function over a Kernel with size :math:`KH\times KW` and a stride of :math:`SV` in vertical and :math:`SH` in horizontal direction respectively.

The dimension :math:`OH\times OW` of the resulting pooled image can be computed as

.. math::

	OH = \lfloor \frac{IH - (KH - (PR + PL))}{SH}\rfloor + 1
	OW = \lfloor \frac{IW - (KW - (PT + PB))}{SV}\rfloor + 1

Note that the _outputChannels must be specified such that it equals :math:`OH\cdot OW\cdot IC`, where :math:`IC` is the number of channels in the original image. This guarantees the the image has the same number of channels in the output.