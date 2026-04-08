**************************
Monte Carlo
**************************

This solver performs the numerical integration of scalar functions :math:`f:\mathbb{R}^D\to \mathbb{R}` using Monte-Carlo Integration. Here, the evalutation points :math:`\mathbf{x}_i\in\mathbb{R}^D` with :math:`i=0,\dots,N-1` are sampled uniformly at random. For the sampled points, the function is evaluated, giving :math:`f(\mathbf{x}_i)`. Using these quantities the numerical approximation of the integral is performed 

.. math::

  I=\int\limits_{a_0}^{b_0}\cdots\int \int\limits_{a_{D-1}}^{b_{D-1}} f(x)\mathrm{d}^Dx \approx \frac{1}{N}\sum\limits_{i=0}^N f(\mathbf{x}_i)

is performed.
