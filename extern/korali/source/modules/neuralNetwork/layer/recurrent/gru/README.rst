*************
GRU Layer
*************

Specialization of the Recurrent Layer for Gated Recurrent Units (GRU). The input sequence entry :math:`t\in\{0,\dots,T\}` denoted by :math:`\mathbf{z}^{t}\in\mathbb{R}^{n_{l-1}}` is processed to the output state :math:`\mathbf{h}^{t}\in\mathbb{R}^{n_{l}}` by

.. math::

	\mathbf{z}_t &= \sigma_g(W_{z} \mathbf{x}_t + U_{z} \mathbf{h}_{t-1} + \mathbf{b}_z) \\
	\mathbf{r}_t &= \sigma_g(W_{r} \mathbf{x}_t + U_{r} \mathbf{h}_{t-1} + \mathbf{b}_r) \\
	\hat{\mathbf{h}}_t &= \phi_h(W_{h} \mathbf{x}_t + U_{h} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \\
	\mathbf{h}_t &=  (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \hat{\mathbf{h}}_t

We note that for :math:`t=0` the needed vector :math:`\mathbf{h}_{t-1}` is a zero vector. From the input and the hidden state we compute the gate :math:`\mathbf{z}_t`, and reset :math:`\mathbf{r}_t` vector using the respective Weights :math:`W_z,W_r,U_z,U_r` and Biases :math:`\mathbf{b}_z,\mathbf{b}_r`. These are used to compute the output state via the Weights :math:`W_h,U_h` and Bias :math:`\mathbf{b}_h`. The used component-wise non-linearitites are sigmoid for :math:`\sigma_g` and hyperbolic tangent functions for :math:`\phi_h`. As an illustration we attach a visual representation of the data-flow through an LSTM layer.