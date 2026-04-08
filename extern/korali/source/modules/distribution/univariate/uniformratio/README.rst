**********************************
Ratio of Two Uniform Distribution
**********************************

UniformRatio
------------

The UniformRatio distribution is the distribution of :math:`Z=X/Y`, where :math:`X` and :math:`Y` are two uniformly distributed random variables. The probability density function is:

.. math::

    f(z;minX,maxX,minY,maxY)=
    \begin{cases}
        (min(maxY, maxX / z)^2 - max(minY, minX / z)^2) * C &  z \in [ minX/maxY, maxX/minY ]\,, \\
    0, & \text{otherwise,}
    \end{cases}
    
where :math:`minX` and :math:`maxX` are the bounds of the random variable X, and :math:`minY` and :math:`maxY` are the bounds of the random variable Y. :math:`C` is a normalization constant.
