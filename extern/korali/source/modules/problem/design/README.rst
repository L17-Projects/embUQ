******
Design
******

The design problem considers the expected information gain of measurements for the experimental design :math:`s`, given by

.. math::

  U(s)=\int_{\mathcal{Y}} \int_{\mathcal{T}} \ln \frac{p(y \mid \vartheta, s)}{p(y \mid s)} p(y \mid \vartheta, s) p(\vartheta) \mathrm{d} \vartheta \mathrm{d} y

where :math:`y` denotes the measurements for parameters :math:`\vartheta`. The goal is to determine the optimal experimental design

.. math::

  s^\star = \arg\max_s U(s)