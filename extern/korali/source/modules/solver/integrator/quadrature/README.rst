**************************
Quadrature
**************************

This solver performs the numerical integration of scalar functions :math:`f:\mathbb{R}^D\to \mathbb{R}` for the "Integration" problem. Provided evalutation points :math:`\mathbf{x}_i\in\mathbb{R}^D` with :math:`i=0,\dots,N-1` the function is evaluated, giving :math:`f(\mathbf{x}_i)`. Using the provided weights :math:`w_d\in\mathbb{R}` with :math:`d=0,\dots,D-1` for one dimension, the cartesian product :math:`w_i=\prod\limits_{d=0}^{D-1}w_{i,d}` with :math:`i=0,\dots,D` is computed. Using these quantities the numerical approximation of the integral is performed 

.. math::

  I=\int\limits_{a_0}^{b_0}\cdots\int \int\limits_{a_{D-1}}^{b_{D-1}} f(x)\mathrm{d}^Dx \approx \sum\limits_{i=0}^N w_i f(\mathbf{x}_i)

is performed. The supported quadrature rules are Rectangle, Trapezoidal, and Simpson.
