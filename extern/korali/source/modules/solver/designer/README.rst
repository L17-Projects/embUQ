*********
Designer
*********

This solver determines the optimal experimental design for a given measurement model. In a first step, we sample the prior distribution of the parameters :math:`\vartheta^{(i)}\sim p(\vartheta)` for :math:`i=1,\dots,N_{\vartheta}`. Given the samples, we evaluate the model :math:`F(\vartheta^{(i)},s)` and save the results for the candidate locations of the design parameters :math:`F(\vartheta^{(i)},s)`. In the last step, the likelihood :math:`p(y|\vartheta,s)` is sampled for each of the parameters. Using the created samples :math:`y^{(i,j)}\sim p(y|\vartheta^{(i)},s)` for :math:`i=1,\dots,N_y` we approximate the utility using numerical integration

.. math::

  \hat U(s)=\sum\limits_{i=1}^{N_{\vartheta}}\sum\limits_{j=1}^{N_y}w_{ij}\left[\ln p(y^{(i,j)}|\vartheta^{(i)},s)-\ln\left(\sum\limits_{k=1}^{N_{\vartheta}} w_{k} p(y^{(i,j)}|\vartheta^{(k)},s) \right) \right]
  

and perform the optimization over the space of design space is performed in order to determine the optimal design.
