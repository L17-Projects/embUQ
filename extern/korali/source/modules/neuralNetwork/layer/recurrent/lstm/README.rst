*************
LSTM Layer
*************

Specialization of the Recurrent Layer for Long Short Term Memory (LSTM). The input sequence entries :math:`t\in\{0,\dots,T\}` denoted by :math:`\mathbf{x}^{t}\in\mathbb{R}^{n_{l-1}}` is processed to the output state :math:`\mathbf{h}^{t-1}\in\mathbb{R}^{n_{l}}` by

.. math::

	\mathbf{f}_t &= \sigma_g(W_{f} \mathbf{x}^{t} + U_{f} \mathbf{h}^{t-1} + \mathbf{b}_f)
	\mathbf{i}_t &= \sigma_g(W_{i} \mathbf{x}^{t} + U_{i} \mathbf{h}^{t-1} + \mathbf{b}_i)
	\mathbf{o}_t &= \sigma_g(W_{o} \mathbf{x}^{t} + U_{o} \mathbf{h}^{t-1} + \mathbf{b}_o)
	\tilde{\mathbf{c}}_t &= \sigma_c(W_{c} \mathbf{x}^{t} + U_{c} \mathbf{h}^{t-1} + \mathbf{b}_c)
	\mathbf{c}_t &= f_t \circ \mathbf{c}_{t-1} + \mathbf{i}_t \circ \tilde{\mathbf{c}}_t \\
	\mathbf{h}_t &= \mathbf{o}_t \circ \sigma_h(\mathbf{c}_t)

We note that for :math:`t=0` the needed vectors :math:`\mathbf{h}_{t-1}` and :math:`\mathbf{c}_{t-1}` are zero vectors. From the input and the hidden state we compute the forget :math:`\mathbf{f}_t`, the input :math:`\mathbf{i}_t`, and output :math:`\mathbf{o}_t` activation vector using the respective Weights :math:`W_f,W_i,W_o,U_f,U_i,U_o` and Biases :math:`\mathbf{b}_f,\mathbf{b}_i,\mathbf{b}_o`. These are used to compute the cell :math:`\tilde{\mathbf{c}}_t` activation vector via the Weights :math:`W_c` and Bias :math:`\mathbf{b}_c`. The cell state :math:`\mathbf{c}_t` is now computed by combining the cell state with the forget and input activation vectors. The output state is obtained by combining the cell state with the output activation vector. The used component-wise non-linearitites are sigmoid for :math:`\sigma_g` and hyperbolic tangent functions for :math:`\sigma_c` and :math:`\sigma_h`. As an illustration we attach a visual representation of the data-flow through an LSTM layer.